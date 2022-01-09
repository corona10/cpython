/*

  Reference Cycle Garbage Collection
  ==================================

  Neil Schemenauer <nas@arctrix.com>

  Based on a post on the python-dev list.  Ideas from Guido van Rossum,
  Eric Tiedemann, and various others.

  http://www.arctrix.com/nas/python/gc/

  The following mailing list threads provide a historical perspective on
  the design of this module.  Note that a fair amount of refinement has
  occurred since those discussions.

  http://mail.python.org/pipermail/python-dev/2000-March/002385.html
  http://mail.python.org/pipermail/python-dev/2000-March/002434.html
  http://mail.python.org/pipermail/python-dev/2000-March/002497.html

  For a highlevel view of the collection process, read the collect
  function.

*/

#include "Python.h"
#include "pycore_context.h"
#include "pycore_initconfig.h"
#include "pycore_interp.h"      // PyInterpreterState.gc
#include "pycore_object.h"
#include "pycore_pyerrors.h"
#include "pycore_pystate.h"     // _PyThreadState_GET()
#include "pydtrace.h"
#include <stdbool.h>

typedef struct _gc_runtime_state GCState;

/*[clinic input]
module gc
[clinic start generated code]*/
/*[clinic end generated code: output=da39a3ee5e6b4b0d input=b5c9690ecc842d79]*/


#ifdef Py_DEBUG
#  define GC_DEBUG
#endif

#define GC_NEXT _PyGCHead_NEXT
#define GC_PREV _PyGCHead_PREV

/* Get an object's GC head */
#define AS_GC(o) ((PyGC_Head *)(((char *)(o))-sizeof(PyGC_Head)))

/* Get the object given the GC head */
#define FROM_GC(g) ((PyObject *)(((char *)(g))+sizeof(PyGC_Head)))

/* Queue to avoid stack overflow when marking reachable objects */
#define MARK_QUEUE_SIZE 1000

struct mark_state {
    PyGC_Head *reachable;
    int generation;
    //PyObject *queue[MARK_QUEUE_SIZE];
    int queue_depth;
    int aborted; /* true if queue depth would have been exceeded, restart */
    bool finalizers; /* we are marking finalizers */
    PyObject *ref; /* for verbose debugging */
    Py_ssize_t found;
};

#define COLOR_BLACK 1 // alive
#define COLOR_GREY  2 // alive but needs traverse
#define COLOR_WHITE 3 // dead (or potentially dead before traversal)

#define GET_COLOR(g) ((g)->gc_color)
#define SET_COLOR(g, v) ((g)->gc_color = v)

#define IS_BLACK(g) (GET_COLOR(g) == COLOR_BLACK)
#define IS_GREY(g) (GET_COLOR(g) == COLOR_GREY)
#define IS_WHITE(g) (GET_COLOR(g) == COLOR_WHITE)

#define SET_GEN(g, v) ((g)->gc_gen = v)
#define GET_GEN(g) ((g)->gc_gen)

// oldest generation that we move objects to on normal collection
#define OLD_GENERATION (NUM_GENERATIONS-1)
// generation used by freeze (next oldest from OLD_GENERATION)
#define PERMANENT_GENERATION NUM_GENERATIONS

bool debug_verbose;

static inline int
_gc_is_finalized(PyObject *obj)
{
    return _PyGC_FINALIZED(obj);
}

static inline void
gc_decref(PyObject *obj)
{
    // Note: ob_refcnt is negative at this point
    _PyObject_ASSERT_WITH_MSG(obj,
                              Py_REFCNT(obj) < 0,
                              "refcount is too small");
    obj->ob_refcnt += 1;
}

/* set for debugging information */
#define DEBUG_STATS             (1<<0) /* print collection statistics */
#define DEBUG_COLLECTABLE       (1<<1) /* print collectable objects */
#define DEBUG_UNCOLLECTABLE     (1<<2) /* print uncollectable objects */
#define DEBUG_SAVEALL           (1<<5) /* save all garbage in gc.garbage */
#define DEBUG_VERBOSE           (1<<6) /* verbose debuggin output */
#define DEBUG_LEAK              DEBUG_COLLECTABLE | \
                DEBUG_UNCOLLECTABLE | \
                DEBUG_SAVEALL

static GCState *
get_gc_state(void)
{
    PyInterpreterState *interp = _PyInterpreterState_GET();
    return &interp->gc;
}

#define HEAD(state) (&(state)->gc_head)

PyStatus
_PyGC_Init(PyInterpreterState *interp)
{
    GCState *gcstate = &interp->gc;

    gcstate->garbage = PyList_New(0);
    if (gcstate->garbage == NULL) {
        return _PyStatus_NO_MEMORY();
    }

    gcstate->callbacks = PyList_New(0);
    if (gcstate->callbacks == NULL) {
        return _PyStatus_NO_MEMORY();
    }

    return _PyStatus_OK();
}


/*
_gc_prev values
---------------

    used for doubly linked list

gc_refs
------
    At the start of a collection, update_refs() copies the true refcount
    to gc_refs, for each object in the generation being collected.
    subtract_refs() then adjusts gc_refs so that it equals the number of
    times an object is referenced directly from outside the generation
    being collected.

_gc_next values
---------------

0
    The object is not tracked

!= 0
    Pointer to the next object in the GC list.

*/

/*** list functions ***/

static inline void
gc_list_init(PyGC_Head *list)
{
    // List header must not have flags.
    // We can assign pointer by simple cast.
    list->_gc_prev = list;
    list->_gc_next = list;
}

/* Remove `node` from the gc list it's currently in. */
static inline void
gc_list_remove(PyGC_Head *node)
{
    PyGC_Head *prev = GC_PREV(node);
    PyGC_Head *next = GC_NEXT(node);

    _PyGCHead_SET_NEXT(prev, next);
    _PyGCHead_SET_PREV(next, prev);

    node->_gc_next = 0; /* object is not currently tracked */
}

/* Collection state while collecting.  Stores a list of object being examined
 * and the original ref counts, allocated on heap. */
typedef struct _gc_collection_state {
    Py_ssize_t max_size;
    Py_ssize_t size;
    // objects currently being examined for cycles
    void **objects;
    // saved reference counts, will be restored after cycle detection done
    Py_ssize_t *refs;
    // flags for objects, 1 == REACHABLE and objects[i] can be invalid pointer
    //uint8_t *flags;
} cstate_t;

static cstate_t *
gc_cstate_new(Py_ssize_t size)
{
    //fprintf(stderr, "allocate %ld gc saved vector\n", size);
    cstate_t *cstate = PyMem_Malloc(sizeof(cstate_t));
    if (cstate == NULL) {
        Py_FatalError("out of memory in GC allocating cstate"); // FIXME
        return NULL;
    }
    cstate->size = 0;
    cstate->max_size = size;
    cstate->objects = PyMem_Malloc(sizeof(void*) * size);
    if (cstate->objects == NULL) {
        PyMem_Free(cstate);
        Py_FatalError("out of memory in GC allocating cstate"); // FIXME
        return NULL;
    }
    cstate->refs = PyMem_Malloc(sizeof(Py_ssize_t) * size);
    if (cstate->refs == NULL) {
        PyMem_Free(cstate->objects);
        PyMem_Free(cstate);
        Py_FatalError("out of memory in GC allocating cstate"); // FIXME
        return NULL;
    }
#if 0
    cstate->flags = PyMem_Malloc(sizeof(uint8_t) * size);
    if (cstate->flags == NULL) {
        PyMem_Free(cstate->refs);
        PyMem_Free(cstate->objects);
        PyMem_Free(cstate);
        assert(0); // FIXME: handle error
        return NULL;
    }
#endif
    return cstate;
}

static bool
gc_cstate_grow(cstate_t *cstate)
{
    Py_ssize_t n = cstate->max_size;
    n += (n >> 2) + 16;
    cstate->objects = PyMem_Realloc(cstate->objects, n * sizeof(void*));
    assert(cstate->objects); // FIXME: check errors
    cstate->refs = PyMem_Realloc(cstate->refs, n * sizeof(Py_ssize_t));
    assert(cstate->refs); // FIXME: check errors
#if 0
    cstate->flags = PyMem_Realloc(cstate->flags, n * sizeof(uint8_t));
    assert(cstate->flags); // FIXME: check errors
#endif
    cstate->max_size = n;
    //fprintf(stderr, "grow cstate %ld\n", cstate->max_size);
    return true; // FIXME: return error
}

static void
gc_cstate_free(cstate_t *cstate)
{
    PyMem_Free(cstate->objects);
    PyMem_Free(cstate->refs);
    PyMem_Free(cstate);
}

/* Add an object to the cstate list */
static Py_ssize_t
gc_cstate_add(cstate_t *cstate, PyObject *op)
{
    /* update saved info */
    Py_ssize_t i = cstate->size++;
    if (i >= cstate->max_size) {
        gc_cstate_grow(cstate); // FIXME: check error
    }
    assert(i < cstate->max_size);
    //cstate->flags[i] = 0;
    cstate->objects[i] = op;
    /* object memory will not be freed when this is set, objects[i] pointer
     * will remain valid, even while cycles are broken and finalizers run */
    _PyGC_SET_FLAG(AS_GC(op), GC_FLAG_IN_CSTATE);
    /* set default flags */
    _PyGC_CLEAR_FLAG(AS_GC(op), GC_FLAG_FINIALIZER_REACHABLE);
    SET_COLOR(AS_GC(op), COLOR_WHITE);
    return i;
}

/* Append objects in a GC list to a Python list.
 * Return 0 if all OK, < 0 if error (out of memory for list)
 */
static int
append_objects(PyObject *py_list, PyGC_Head *gc_list, int generation)
{
    PyGC_Head *gc;
    for (gc = GC_NEXT(gc_list); gc != gc_list; gc = GC_NEXT(gc)) {
        if (generation != -1 && GET_GEN(gc) != generation) {
            continue;
        }
        PyObject *op = FROM_GC(gc);
        if (op != py_list) {
            if (PyList_Append(py_list, op)) {
                return -1; /* exception */
            }
        }
    }
    return 0;
}

// Constants for validate_list's flags argument.
enum flagstates {collecting_clear_unreachable_clear,
                 collecting_clear_unreachable_set,
                 collecting_set_unreachable_clear,
                 collecting_set_unreachable_set};


void
_PyGC_InitState(GCState *gcstate)
{
    gcstate->enabled = 1; /* automatic collection enabled? */

    struct gc_generation generations[NUM_GENERATIONS] = {
        /* threshold,    count */
        {700,        0},
        {10,         0},
        {10,         0},
    };
    for (int i = 0; i < NUM_GENERATIONS; i++) {
        gcstate->generations[i] = generations[i];
    };
    gc_list_init(&gcstate->gc_head);
    struct gc_generation permanent_generation = {
          0, 0
    };
    gcstate->permanent_generation = permanent_generation;
}

static bool
gc_in_cstate(PyGC_Head *gc)
{
    return _PyGC_HAVE_FLAG(gc, GC_FLAG_IN_CSTATE);
}

static bool
object_in_cstate(PyObject *op)
{
    return PyObject_IS_GC(op) && gc_in_cstate(AS_GC(op));
}

/* Create cstate list for collected generation */
static cstate_t *
gc_build_cstate(GCState *gcstate, int generation)
{
    cstate_t *cstate = gc_cstate_new(1000);
    PyGC_Head *head = HEAD(gcstate);
    for (PyGC_Head *gc = GC_NEXT(head); gc != head; gc = GC_NEXT(gc)) {
        if (GET_GEN(gc) > generation) {
            assert(IS_BLACK(gc));
            continue;
        }
        if (!_PyObject_GC_IS_TRACKED(FROM_GC(gc))) {
            assert(IS_BLACK(gc));
            continue;
        }

        PyObject *op = FROM_GC(gc);
        /* Python's cyclic gc should never see an incoming refcount
         * of 0:  if something decref'ed to 0, it should have been
         * deallocated immediately at that time.
         * Possible cause (if the assert triggers):  a tp_dealloc
         * routine left a gc-aware object tracked during its teardown
         * phase, and did something-- or allowed something to happen --
         * that called back into Python.  gc can trigger then, and may
         * see the still-tracked dying object.  Before this assert
         * was added, such mistakes went on to allow gc to try to
         * delete the object again.  In a debug build, that caused
         * a mysterious segfault, when _Py_ForgetReference tried
         * to remove the object from the doubly-linked list of all
         * objects a second time.  In a release build, an actual
         * double deallocation occurred, which leads to corruption
         * of the allocator's internal bookkeeping pointers.  That's
         * so serious that maybe this should be a release-build
         * check instead of an assert?
         */
        _PyObject_ASSERT(op, Py_REFCNT(op) > 0);
        gc_cstate_add(cstate, op);
    }
    //fprintf(stderr, "cstate size %ld\n", cstate->size);
    return cstate;
}

/* Save current refcnt of objects, done before subtracting refs */
static void
save_refs(cstate_t *cstate)
{
    for (Py_ssize_t i = 0; i < cstate->size; i++) {
        PyObject *op = cstate->objects[i];
        Py_ssize_t count = Py_REFCNT(op);
        assert(count >= 0);
        cstate->refs[i] = count;
        // negative value marks object for visit_decref()
        op->ob_refcnt = -count;
    }
}

/* A traversal callback for subtract_refs. */
static int
visit_decref(PyObject *op, void *parent)
{
    assert(op != NULL);
    if (op->ob_refcnt < 0) {
        gc_decref(op);
    }
    return 0;
}

/* Subtract internal references from gc_refs.  After this, gc_refs is >= 0
 * for all objects in containers, and is GC_REACHABLE for all tracked gc
 * objects not in containers.  The ones with gc_refs > 0 are directly
 * reachable from outside containers, and so can't be collected.
 */
static void
subtract_refs(GCState *gcstate, cstate_t *cstate)
{
    traverseproc traverse;
    for (Py_ssize_t i = 0; i < cstate->size; i++) {
        PyObject *op = cstate->objects[i];
        if (!IS_WHITE(AS_GC(op))) {
            continue;
        }
#if 0
        if (state && state->debug & DEBUG_VERBOSE) {
            fprintf(stderr, "subtract op=%p gen=%d ", op, GET_GEN(gc));
            //dump_obj(op);
        }
#endif
        traverse = Py_TYPE(op)->tp_traverse;
        (void) traverse(op,
                        (visitproc)visit_decref,
                        op);
    }
}


/* A traversal callback for mark_reachable. */
static int
visit_reachable(PyObject *op, struct mark_state *state)
{
    PyGC_Head *gc = AS_GC(op);
    if (!object_in_cstate(op)) {
        return 0;
    }
    // Ignore untracked objects
    if (gc->_gc_next == 0) {
        return 0;
    }

    switch GET_COLOR(gc) {
        case COLOR_WHITE:
            // We thought this object was dead but it turns out to be alive.
            // Mark as grey because we have to traverse it yet.
#if 0
            if (debug_verbose) {
                fprintf(stderr, "became reachable %p->%p\n", state->ref,
                        FROM_GC(gc));
            }
#endif
            state->found++;
            SET_COLOR(gc, COLOR_GREY);
            /* no break, continue to grey case */
            __attribute__ ((fallthrough));
        case COLOR_GREY:
            // Try to traverse all objects reachable from this one
            if (state->queue_depth >= MARK_QUEUE_SIZE) {
                /* we can't recurse further, leave it grey, next loop will
                 * get it */
                state->aborted = 1;
            }
            else {
                /* mark it black as it is proven reachable and we don't have
                 * to look at it anymore.  Note that every loop in
                 * mark_reachable() must at least turn some GREY to BLACK in
                 * order for us to make progress. */
                SET_COLOR(gc, COLOR_BLACK);
                if (state->finalizers) {
                    _PyGC_SET_FLAG(gc, GC_FLAG_FINIALIZER_REACHABLE);
                }
                state->queue_depth += 1;
                state->ref = FROM_GC(gc);
                traverseproc traverse = Py_TYPE(op)->tp_traverse;
                traverse(op, (visitproc)visit_reachable, state);
                state->queue_depth -= 1;
                assert(state->queue_depth >= 0);
            }
            break;
        case COLOR_BLACK:
            // Object is alive, either not in collected generation or
            // referenced from other alive objects
            break;
        default:
            assert(0); /* invalid GET_COLOR() value */
    }
    return 0;
}

static Py_ssize_t
propagate_reachable(GCState *gcstate, cstate_t *cstate, bool finalizers)
{
    PyGC_Head *head = HEAD(gcstate);
    struct mark_state mstate;
    int done = 0;
    int mark_loops = 0;
    mstate.found = 0;
    mstate.reachable = head;
    mstate.finalizers = finalizers;

    while (!done) {
        mark_loops += 1;
        mstate.aborted = 0;
        mstate.queue_depth = 0;
        /* turn grey to black and all reachable from those also black */
        for (Py_ssize_t i = 0; i < cstate->size; i++) {
            PyObject *op = cstate->objects[i];
            PyGC_Head *gc = AS_GC(op);
            if (IS_GREY(gc)) {
                assert(_PyObject_GC_IS_TRACKED(op));
                visit_reachable(op, &mstate);
            }
        }
        /* if queue depth was exceeded, need another pass */
        done = !mstate.aborted;
    }
    if (debug_verbose && mark_loops > 0) {
        fprintf(stderr, "mark loops needed %d\n", mark_loops);
    }
#if Py_DEBUG
    // done propagate, there must be no grey left at this point
    for (Py_ssize_t i = 0; i < cstate->size; i++) {
        PyObject *op = cstate->objects[i];
        PyGC_Head *gc = AS_GC(op);
        if (!IS_WHITE(gc)) {
            assert(IS_BLACK(gc));
        }
    }
#endif
    return mstate.found;
}

static void
restore_refs(GCState *gcstate, cstate_t *cstate)
{
    // restore original refcnt and update reachable flag
    for (Py_ssize_t i = 0; i < cstate->size; i++) {
        PyObject *op = cstate->objects[i];
        op->ob_refcnt = cstate->refs[i];
        assert(op->ob_refcnt >= 0);
    }
}

// Mark all objects that are alive as grey, potential garbage will remain white
static Py_ssize_t
mark_reachable(GCState *gcstate, cstate_t *cstate)
{
    Py_ssize_t n = 0;
    for (Py_ssize_t i = 0; i < cstate->size; i++) {
        PyObject *op = cstate->objects[i];
        PyGC_Head *gc = AS_GC(op);
        if (!IS_WHITE(gc)) {
            continue;
        }
        if (Py_REFCNT(op)) {
            /* gc is definitely reachable from outside */
            /* mark grey since we still need to mark all objects reachable from
             * it */
            SET_COLOR(gc, COLOR_GREY);
            n++;
        }
    }
    n += propagate_reachable(gcstate, cstate, false);
    return n;
}

static Py_ssize_t
find_reachable(GCState *gcstate, cstate_t *cstate)
{
    save_refs(cstate);
    subtract_refs(gcstate, cstate);
    Py_ssize_t revived = mark_reachable(gcstate, cstate);
    restore_refs(gcstate, cstate);
    return revived;
}

static void
increment_generation(GCState *gcstate, cstate_t *cstate, int generation)
{
    if (generation < OLD_GENERATION) {
        generation +=1;
    }
    //fprintf(stderr, "set gen %d\n", generation);
    for (Py_ssize_t i = 0; i < cstate->size; i++) {
        PyObject *op = cstate->objects[i];
        PyGC_Head *gc = AS_GC(op);
        SET_GEN(gc, generation);
    }
}

static void
untrack_tuples(GCState *gcstate, cstate_t *cstate)
{
    for (Py_ssize_t i = 0; i < cstate->size; i++) {
        PyObject *op = cstate->objects[i];
        if (PyTuple_CheckExact(op) && IS_BLACK(AS_GC(op))) {
            _PyTuple_MaybeUntrack(op);
            if (!_PyObject_GC_IS_TRACKED(op)) {
                SET_COLOR(AS_GC(op), COLOR_BLACK);
            }
        }
    }
}

/* Try to untrack all currently tracked dictionaries */
static void
untrack_dicts(GCState *gcstate, cstate_t *cstate)
{
    for (Py_ssize_t i = 0; i < cstate->size; i++) {
        PyObject *op = cstate->objects[i];
        if (PyDict_CheckExact(op) && IS_BLACK(AS_GC(op))) {
            _PyDict_MaybeUntrack(op);
        }
    }
}

/* Return true if object has a pre-PEP 442 finalization method. */
static int
has_legacy_finalizer(PyObject *op)
{
    return Py_TYPE(op)->tp_del != NULL;
}

/* Mark unreachable objects that have tp_del slots as grey.  Traverse those
 * objects and mark everything reachable as black and also set the
 * GC_FLAG_FINIALIZER_REACHABLE flag.
 */
static void
mark_legacy_finalizers(GCState *gcstate, cstate_t *cstate)
{

    bool need_propogate = false;

    for (Py_ssize_t i = 0; i < cstate->size; i++) {
        PyObject *op = cstate->objects[i];
        PyGC_Head *gc = AS_GC(op);
        if (IS_BLACK(gc)) {
            continue;
        }
        if (has_legacy_finalizer(op)) {
            //fprintf(stderr, "move legacy %p\n", op);
            SET_COLOR(gc, COLOR_GREY);
            need_propogate = true;
        }
    }
    if (need_propogate) {
        propagate_reachable(gcstate, cstate, true);
    }
}

/* Clear all weakrefs to unreachable objects, and if such a weakref has a
 * callback, invoke it if necessary.  Note that it's possible for such
 * weakrefs to be outside the unreachable set -- indeed, those are precisely
 * the weakrefs whose callbacks must be invoked.  See gc_weakref.txt for
 * overview & some details.  Some weakrefs with callbacks may be reclaimed
 * directly by this routine; the number reclaimed is the return value.  Other
 * weakrefs with callbacks may be moved into the `old` generation.  Objects
 * moved into `old` have gc_refs set to GC_REACHABLE; the objects remaining in
 * unreachable are left at GC_TENTATIVELY_UNREACHABLE.  When this returns,
 * no object in `unreachable` is weakly referenced anymore.
 */
static int
handle_weakrefs(GCState *gcstate, cstate_t *cstate)
{
    PyObject *op;               /* generally FROM_GC(gc) */
    PyWeakReference *wr;        /* generally a cast of op */
    PyObject *wrcb_to_call;     /* weakrefs with callbacks to call */
    int num_freed = 0;
    wrcb_to_call = PyList_New(0);
    assert(wrcb_to_call != NULL);
    _PyObject_GC_UNTRACK(wrcb_to_call);
    /* Clear all weakrefs to the objects in unreachable.  If such a weakref
     * also has a callback, move it into `wrcb_to_call` if the callback
     * needs to be invoked.  Note that we cannot invoke any callbacks until
     * all weakrefs to unreachable objects are cleared, lest the callback
     * resurrect an unreachable object via a still-active weakref.  We
     * make another pass over wrcb_to_call, invoking callbacks, after this
     * pass completes.
     */
    for (Py_ssize_t i = 0; i < cstate->size; i++) {
        PyObject *op = cstate->objects[i];
        PyGC_Head *gc = AS_GC(op);
        if (IS_BLACK(gc)) {
            continue;
        }
        PyWeakReference **wrlist;

        op = FROM_GC(gc);

        if (PyWeakref_Check(op)) {
            /* A weakref inside the unreachable set must be cleared.  If we
             * allow its callback to execute inside delete_garbage(), it
             * could expose objects that have tp_clear already called on
             * them.  Or, it could resurrect unreachable objects.  One way
             * this can happen is if some container objects do not implement
             * tp_traverse.  Then, wr_object can be outside the unreachable
             * set but can be deallocated as a result of breaking the
             * reference cycle.  If we don't clear the weakref, the callback
             * will run and potentially cause a crash.  See bpo-38006 for
             * one example.
             */
            _PyWeakref_ClearRef((PyWeakReference *)op);
        }

        if (! PyType_SUPPORTS_WEAKREFS(Py_TYPE(op)))
            continue;

        /* It supports weakrefs.  Does it have any? */
        wrlist = (PyWeakReference **)
                                _PyObject_GET_WEAKREFS_LISTPTR(op);

        /* `op` may have some weakrefs.  March over the list, clear
         * all the weakrefs, and move the weakrefs with callbacks
         * that must be called into wrcb_to_call.
         */
        for (wr = *wrlist; wr != NULL; wr = *wrlist) {
            PyGC_Head *wrasgc;                  /* AS_GC(wr) */

            /* _PyWeakref_ClearRef clears the weakref but leaves
             * the callback pointer intact.  Obscure:  it also
             * changes *wrlist.
             */
            _PyObject_ASSERT((PyObject *)wr, wr->wr_object == op);
            //fprintf(stderr, "clear weakref %p %p\n", wr, wr->wr_object);
            _PyWeakref_ClearRef(wr);
            _PyObject_ASSERT((PyObject *)wr, wr->wr_object == Py_None);
            if (wr->wr_callback == NULL) {
                /* no callback */
                continue;
            }

            /* Headache time.  `op` is going away, and is weakly referenced by
             * `wr`, which has a callback.  Should the callback be invoked?  If wr
             * is also trash, no:
             *
             * 1. There's no need to call it.  The object and the weakref are
             *    both going away, so it's legitimate to pretend the weakref is
             *    going away first.  The user has to ensure a weakref outlives its
             *    referent if they want a guarantee that the wr callback will get
             *    invoked.
             *
             * 2. It may be catastrophic to call it.  If the callback is also in
             *    cyclic trash (CT), then although the CT is unreachable from
             *    outside the current generation, CT may be reachable from the
             *    callback.  Then the callback could resurrect insane objects.
             *
             * Since the callback is never needed and may be unsafe in this case,
             * wr is simply left in the unreachable set.  Note that because we
             * already called _PyWeakref_ClearRef(wr), its callback will never
             * trigger.
             *
             * OTOH, if wr isn't part of CT, we should invoke the callback:  the
             * weakref outlived the trash.  Note that since wr isn't CT in this
             * case, its callback can't be CT either -- wr acted as an external
             * root to this generation, and therefore its callback did too.  So
             * nothing in CT is reachable from the callback either, so it's hard
             * to imagine how calling it later could create a problem for us.  wr
             * is moved to wrcb_to_call in this case.
             */
            wrasgc = AS_GC(wr);
            if (IS_BLACK(wrasgc)) {
                /* Move wr to wrcb_to_call, for the next pass. */
                PyList_Append(wrcb_to_call, (PyObject *)wr);
            }
        }
    }

    /* Invoke the callbacks we decided to honor.  It's safe to invoke them
     * because they can't reference unreachable objects.
     */
    Py_ssize_t len = PyList_GET_SIZE(wrcb_to_call);
    for (Py_ssize_t i = 0; i < len; i++) {
        PyObject *temp;
        PyObject *callback;

        op = PyList_GET_ITEM(wrcb_to_call, i);
        _PyObject_ASSERT(op, PyWeakref_Check(op));
        wr = (PyWeakReference *)op;
        callback = wr->wr_callback;
        _PyObject_ASSERT(op, callback != NULL);

        /* copy-paste of weakrefobject.c's handle_callback() */
        //PySys_WriteStderr("calling wr callback %p\n", callback);
        temp = PyObject_CallOneArg(callback, (PyObject *)wr);
        if (temp == NULL)
            PyErr_WriteUnraisable(callback);
        else
            Py_DECREF(temp);

        /* Give up the reference we created in the first pass.  When
         * op's refcount hits 0 (which it may or may not do right now),
         * op's tp_dealloc will decref op->wr_callback too.  Note
         * that the refcount probably will hit 0 now, and because this
         * weakref was reachable to begin with, gc didn't already
         * add it to its count of freed objects.  Example:  a reachable
         * weak value dict maps some key to this reachable weakref.
         * The callback removes this key->weakref mapping from the
         * dict, leaving no other references to the weakref (excepting
         * ours).
         */
        PyList_SetItem(wrcb_to_call, i, Py_None);
#if 0 // FIXME
        if (wrcb_to_call._gc_next == gc) {
            /* object is still alive */
            SET_COLOR(gc, COLOR_BLACK);
        }
        else {
            ++num_freed;
        }
#endif
    }

    return num_freed;
}

static void
debug_cycle(const char *msg, PyObject *op)
{
    PySys_FormatStderr("gc: %s <%s %p>\n",
                       msg, Py_TYPE(op)->tp_name, op);
}

/* Handle uncollectable garbage (cycles with tp_del slots, and stuff reachable
 * only from such cycles).
 * If DEBUG_SAVEALL, all objects in finalizers are appended to the module
 * garbage list (a Python list), else only the objects in finalizers with
 * __del__ methods are appended to garbage.  All objects in finalizers are
 * merged into the old list regardless.
 */
static void
handle_legacy_finalizers(PyThreadState *tstate, GCState *gcstate, cstate_t *cstate)
{
    assert(!_PyErr_Occurred(tstate));
    assert(gcstate->garbage != NULL);

    for (Py_ssize_t i = 0; i < cstate->size; i++) {
        PyObject *op = cstate->objects[i];
        PyGC_Head *gc = AS_GC(op);
        if (_PyGC_HAVE_FLAG(gc, GC_FLAG_NEED_FREE)) {
            continue;
        }
        if (!_PyGC_HAVE_FLAG(gc, GC_FLAG_FINIALIZER_REACHABLE)) {
            continue;
        }
        if (!_PyObject_GC_IS_TRACKED(op)) {
            continue; // might be freed but on freelist
        }
        if ((gcstate->debug & DEBUG_SAVEALL) || has_legacy_finalizer(op)) {
            assert(IS_BLACK(gc)); // set with GC_FLAG_FINIALIZER_REACHABLE
            if (PyList_Append(gcstate->garbage, op) < 0) {
                _PyErr_Clear(tstate);
                break;
            }
        }
    }
}

/* Run first-time finalizers (if any) on all the objects in cstate->objects.
 * Some of the items in cstate->objects might become invalid if refcnt goes to
 * zero.
 */
static bool
finalize_garbage(PyThreadState *tstate, GCState *gcstate, cstate_t *cstate)
{
    bool have_finalizers = false;
    destructor finalize;
    for (Py_ssize_t i = 0; i < cstate->size; i++) {
        PyObject *op = cstate->objects[i];
        PyGC_Head *gc = AS_GC(op);
        if (!IS_WHITE(gc)) {
            continue;
        }
        if (!_PyObject_GC_IS_TRACKED(FROM_GC(gc))) {
            continue;
        }
        assert(!_PyGC_HAVE_FLAG(AS_GC(op), GC_FLAG_NEED_FREE));
        if (!_gc_is_finalized(op) &&
                (finalize = Py_TYPE(op)->tp_finalize) != NULL) {
            _PyGC_SET_FINALIZED(op);
            Py_INCREF(op);
            finalize(op);
            Py_DECREF(op);
            have_finalizers = true;
        }
    }
    return have_finalizers;
}

/* Break reference cycles by clearing the containers involved.
 */
static void
delete_garbage(PyThreadState *tstate, GCState *gcstate, cstate_t *cstate)
{
    assert(!_PyErr_Occurred(tstate));

    for (Py_ssize_t i = 0; i < cstate->size; i++) {
        PyObject *op = cstate->objects[i];
        PyGC_Head *gc = AS_GC(op);
        if (!IS_WHITE(gc)) {
            continue;
        }
        if (!_PyObject_GC_IS_TRACKED(FROM_GC(gc))) {
            continue; // might be freed but on freelist
        }
        assert(!_PyGC_HAVE_FLAG(AS_GC(op), GC_FLAG_NEED_FREE));

        if (gcstate->debug & DEBUG_SAVEALL) {
            assert(gcstate->garbage != NULL);
            if (PyList_Append(gcstate->garbage, op) < 0) {
                _PyErr_Clear(tstate);
            }
            SET_COLOR(gc, COLOR_BLACK); // is reachable now
        }
        else if (Py_REFCNT(op) > 0) {
            inquiry clear;
            if ((clear = Py_TYPE(op)->tp_clear) != NULL) {
                Py_INCREF(op);
                (void) clear(op);
                Py_DECREF(op);
                if (_PyErr_Occurred(tstate)) {
                    _PyErr_WriteUnraisableMsg("in tp_clear of",
                                              (PyObject*)Py_TYPE(op));
                }
            }
        }
    }
}

static void
gc_free_deferred(PyThreadState *tstate, GCState *gcstate, cstate_t *cstate)
{
    Py_ssize_t num_freed = 0;
    for (Py_ssize_t i = 0; i < cstate->size; i++) {
        PyObject *op = cstate->objects[i];
        PyGC_Head *gc = AS_GC(op);
        if (_PyGC_HAVE_FLAG(gc, GC_FLAG_NEED_FREE)) {
            PyObject_FREE(gc);
            num_freed++;
        }
        else {
            _PyGC_CLEAR_FLAG(gc, GC_FLAG_IN_CSTATE);
            SET_COLOR(gc, COLOR_BLACK); // FIXME: needed?
        }
    }
#if 0
    if (num_freed > 0) {
        fprintf(stderr, "deferred num freed %ld\n", num_freed);
    }
#endif
}

/* Clear all free lists
 * All free lists are cleared during the collection of the highest generation.
 * Allocated items in the free list may keep a pymalloc arena occupied.
 * Clearing the free lists may give back memory to the OS earlier.
 */
static void
clear_freelists(PyInterpreterState *interp)
{
    _PyTuple_ClearFreeList(interp);
    _PyFloat_ClearFreeList(interp);
    _PyList_ClearFreeList(interp);
    _PyDict_ClearFreeList(interp);
    _PyAsyncGen_ClearFreeLists(interp);
    _PyContext_ClearFreeList(interp);
}

void
tally_gen_counts(GCState *gcstate, Py_ssize_t *counts)
{
    PyGC_Head *head = HEAD(gcstate);
    for (PyGC_Head *gc = GC_NEXT(head); gc != head; gc = GC_NEXT(gc)) {
        counts[GET_GEN(gc)]++;
    }
}

// Show stats for objects in each generations
static void
show_stats_each_generations(GCState *gcstate)
{
    char buf[100];
    size_t pos = 0;
    Py_ssize_t counts[NUM_GENERATIONS+1];

    tally_gen_counts(gcstate, counts);
    for (int i = 0; i < NUM_GENERATIONS && pos < sizeof(buf); i++) {
        pos += PyOS_snprintf(buf+pos, sizeof(buf)-pos,
                             " %zd",
                             counts[i]);
    }

    PySys_FormatStderr(
        "gc: objects in each generation:%s\n"
        "gc: objects in permanent generation: %zd\n",
        buf, counts[PERMANENT_GENERATION]);
}

/* This is the main function.  Read this to understand how the
 * collection process works. */
static Py_ssize_t
gc_collect_main(PyThreadState *tstate, int generation,
                Py_ssize_t *n_collected, Py_ssize_t *n_uncollectable,
                int nofail)
{
    int i;
    Py_ssize_t m = 0; /* # objects collected */
    Py_ssize_t n = 0; /* # unreachable objects that couldn't be collected */
    _PyTime_t t1 = 0;   /* initialize to prevent a compiler warning */
    GCState *gcstate = &tstate->interp->gc;

    // gc_collect_main() must not be called before _PyGC_Init
    // or after _PyGC_Fini()
    assert(gcstate->garbage != NULL);
    assert(!_PyErr_Occurred(tstate));

#ifdef EXPERIMENTAL_ISOLATED_SUBINTERPRETERS
    if (tstate->interp->config._isolated_interpreter) {
        // bpo-40533: The garbage collector must not be run on parallel on
        // Python objects shared by multiple interpreters.
        return 0;
    }
#endif

    if (gcstate->debug & DEBUG_STATS) {
        PySys_WriteStderr("gc: collecting generation %d...\n", generation);
        show_stats_each_generations(gcstate);
        t1 = _PyTime_GetPerfCounter();
    }

    if (PyDTrace_GC_START_ENABLED())
        PyDTrace_GC_START(generation);

    /* update collection and allocation counters */
    if (generation+1 < NUM_GENERATIONS)
        gcstate->generations[generation+1].count += 1;
    for (i = 0; i <= generation; i++)
        gcstate->generations[i].count = 0;

    cstate_t *cstate = gc_build_cstate(gcstate, generation);
    assert(cstate);
    increment_generation(gcstate, cstate, generation);
    find_reachable(gcstate, cstate);

    untrack_tuples(gcstate, cstate);
    /* Move reachable objects to next generation. */
    if (generation < NUM_GENERATIONS - 1) {
        if (generation == NUM_GENERATIONS - 2) {
            gcstate->long_lived_pending += 0; // FIXME
        }
    }
    else {
        /* We only un-track dicts in full collections, to avoid quadratic
           dict build-up. See issue #14775. */
        untrack_dicts(gcstate, cstate);
        gcstate->long_lived_pending = 0;
        gcstate->long_lived_total = 0; // FIXME
    }

    /* All objects in unreachable are trash, but objects reachable from
     * legacy finalizers (e.g. tp_del) can't safely be deleted.
     */
    mark_legacy_finalizers(gcstate, cstate);
    /* Collect statistics on collectable and uncollectable objects found and
     * print debugging information.
     */
    for (Py_ssize_t i = 0; i < cstate->size; i++) {
        PyObject *op = cstate->objects[i];
        PyGC_Head *gc = AS_GC(op);
        if (IS_WHITE(gc)) {
            m++;
            if (gcstate->debug & DEBUG_COLLECTABLE) {
                debug_cycle("collectable", op);
            }
        }
        if (_PyGC_HAVE_FLAG(gc, GC_FLAG_FINIALIZER_REACHABLE)) {
            n++;
            if (gcstate->debug & DEBUG_UNCOLLECTABLE) {
                debug_cycle("uncollectable", op);
            }
        }
    }

    /* Clear weakrefs and invoke callbacks as necessary. */
    m += handle_weakrefs(gcstate, cstate);

    if (m > 0) {
        /* Call tp_finalize on objects which have one. */
        bool have_finalizers = finalize_garbage(tstate, gcstate, cstate);

        if (have_finalizers) {
            /* Second pass of garbage cycle finding. Start with set of
             * previously unreachable objects.  Check what is still unreachable
             * after running finalizers. */
            cstate_t *garbage_cstate = gc_cstate_new(m);
            assert(cstate); // FIXME
            for (Py_ssize_t i = 0; i < cstate->size; i++) {
                PyObject *op = cstate->objects[i];
                if (IS_BLACK(AS_GC(op))) {
                    continue;
                }
                if (!_PyObject_GC_IS_TRACKED(op)) {
                    continue; // might be freed but on freelist
                }
                gc_cstate_add(garbage_cstate, op);
            }
            Py_ssize_t revived = find_reachable(gcstate, garbage_cstate);
#if 0
            fprintf(stderr, "size = %ld m = %ld revived = %ld\n",
                    garbage_cstate->size, m, revived);
#endif
            m -= revived;
            gc_cstate_free(garbage_cstate);
        }
        if (m > 0) {
           /* Call tp_clear on objects in the unreachable set.  This will
             * cause the reference cycles to be broken.  It may also cause some
             * objects in finalizers to be freed.
             */
            delete_garbage(tstate, gcstate, cstate);
        }
    }

    /* Append instances in the uncollectable set to a Python
     * reachable list of garbage.  The programmer has to deal with
     * this if they insist on creating this type of structure.
     */
    if (n > 0) {
        handle_legacy_finalizers(tstate, gcstate, cstate);
    }

    gc_free_deferred(tstate, gcstate, cstate);
    gc_cstate_free(cstate);
    if (gcstate->debug & DEBUG_STATS) {
        double d = _PyTime_AsSecondsDouble(_PyTime_GetPerfCounter() - t1);
        PySys_WriteStderr(
            "gc: done, %zd unreachable, %zd uncollectable, %.4fs elapsed\n",
            n+m, n, d);
    }

    /* Clear free list only during the collection of the highest
     * generation */
    if (generation == NUM_GENERATIONS-1) {
        clear_freelists(tstate->interp);
    }

    if (_PyErr_Occurred(tstate)) {
        if (nofail) {
            _PyErr_Clear(tstate);
        }
        else {
            _PyErr_WriteUnraisableMsg("in garbage collection", NULL);
        }
    }

    /* Update stats */
    if (n_collected) {
        *n_collected = m;
    }
    if (n_uncollectable) {
        *n_uncollectable = n;
    }

    struct gc_generation_stats *stats = &gcstate->generation_stats[generation];
    stats->collections++;
    stats->collected += m;
    stats->uncollectable += n;

    if (PyDTrace_GC_DONE_ENABLED()) {
        PyDTrace_GC_DONE(n + m);
    }

    assert(!_PyErr_Occurred(tstate));
    return n + m;
}

/* Invoke progress callbacks to notify clients that garbage collection
 * is starting or stopping
 */
static void
invoke_gc_callback(PyThreadState *tstate, const char *phase,
                   int generation, Py_ssize_t collected,
                   Py_ssize_t uncollectable)
{
    assert(!_PyErr_Occurred(tstate));

    /* we may get called very early */
    GCState *gcstate = &tstate->interp->gc;
    if (gcstate->callbacks == NULL) {
        return;
    }

    /* The local variable cannot be rebound, check it for sanity */
    assert(PyList_CheckExact(gcstate->callbacks));
    PyObject *info = NULL;
    if (PyList_GET_SIZE(gcstate->callbacks) != 0) {
        info = Py_BuildValue("{sisnsn}",
            "generation", generation,
            "collected", collected,
            "uncollectable", uncollectable);
        if (info == NULL) {
            PyErr_WriteUnraisable(NULL);
            return;
        }
    }
    for (Py_ssize_t i=0; i<PyList_GET_SIZE(gcstate->callbacks); i++) {
        PyObject *r, *cb = PyList_GET_ITEM(gcstate->callbacks, i);
        Py_INCREF(cb); /* make sure cb doesn't go away */
        r = PyObject_CallFunction(cb, "sO", phase, info);
        if (r == NULL) {
            PyErr_WriteUnraisable(cb);
        }
        else {
            Py_DECREF(r);
        }
        Py_DECREF(cb);
    }
    Py_XDECREF(info);
    assert(!_PyErr_Occurred(tstate));
}

/* Perform garbage collection of a generation and invoke
 * progress callbacks.
 */
static Py_ssize_t
gc_collect_with_callback(PyThreadState *tstate, int generation)
{
    assert(!_PyErr_Occurred(tstate));
    Py_ssize_t result, collected, uncollectable;
    invoke_gc_callback(tstate, "start", generation, 0, 0);
    result = gc_collect_main(tstate, generation, &collected, &uncollectable, 0);
    invoke_gc_callback(tstate, "stop", generation, collected, uncollectable);
    assert(!_PyErr_Occurred(tstate));
    return result;
}

static Py_ssize_t
gc_collect_generations(PyThreadState *tstate)
{
    GCState *gcstate = &tstate->interp->gc;
    /* Find the oldest generation (highest numbered) where the count
     * exceeds the threshold.  Objects in the that generation and
     * generations younger than it will be collected. */
    Py_ssize_t n = 0;
    for (int i = NUM_GENERATIONS-1; i >= 0; i--) {
        if (gcstate->generations[i].count > gcstate->generations[i].threshold) {
            /* Avoid quadratic performance degradation in number
               of tracked objects (see also issue #4074):

               To limit the cost of garbage collection, there are two strategies;
                 - make each collection faster, e.g. by scanning fewer objects
                 - do less collections
               This heuristic is about the latter strategy.

               In addition to the various configurable thresholds, we only trigger a
               full collection if the ratio

                long_lived_pending / long_lived_total

               is above a given value (hardwired to 25%).

               The reason is that, while "non-full" collections (i.e., collections of
               the young and middle generations) will always examine roughly the same
               number of objects -- determined by the aforementioned thresholds --,
               the cost of a full collection is proportional to the total number of
               long-lived objects, which is virtually unbounded.

               Indeed, it has been remarked that doing a full collection every
               <constant number> of object creations entails a dramatic performance
               degradation in workloads which consist in creating and storing lots of
               long-lived objects (e.g. building a large list of GC-tracked objects would
               show quadratic performance, instead of linear as expected: see issue #4074).

               Using the above ratio, instead, yields amortized linear performance in
               the total number of objects (the effect of which can be summarized
               thusly: "each full garbage collection is more and more costly as the
               number of objects grows, but we do fewer and fewer of them").

               This heuristic was suggested by Martin von Löwis on python-dev in
               June 2008. His original analysis and proposal can be found at:
               http://mail.python.org/pipermail/python-dev/2008-June/080579.html
            */
            if (i == NUM_GENERATIONS - 1
                && gcstate->long_lived_pending < gcstate->long_lived_total / 4)
                continue;
            n = gc_collect_with_callback(tstate, i);
            break;
        }
    }
    return n;
}

#include "clinic/gcmodule.c.h"

/*[clinic input]
gc.enable

Enable automatic garbage collection.
[clinic start generated code]*/

static PyObject *
gc_enable_impl(PyObject *module)
/*[clinic end generated code: output=45a427e9dce9155c input=81ac4940ca579707]*/
{
    PyGC_Enable();
    Py_RETURN_NONE;
}

/*[clinic input]
gc.disable

Disable automatic garbage collection.
[clinic start generated code]*/

static PyObject *
gc_disable_impl(PyObject *module)
/*[clinic end generated code: output=97d1030f7aa9d279 input=8c2e5a14e800d83b]*/
{
    PyGC_Disable();
    Py_RETURN_NONE;
}

/*[clinic input]
gc.isenabled -> bool

Returns true if automatic garbage collection is enabled.
[clinic start generated code]*/

static int
gc_isenabled_impl(PyObject *module)
/*[clinic end generated code: output=1874298331c49130 input=30005e0422373b31]*/
{
    return PyGC_IsEnabled();
}

/*[clinic input]
gc.collect -> Py_ssize_t

    generation: int(c_default="NUM_GENERATIONS - 1") = 2

Run the garbage collector.

With no arguments, run a full collection.  The optional argument
may be an integer specifying which generation to collect.  A ValueError
is raised if the generation number is invalid.

The number of unreachable objects is returned.
[clinic start generated code]*/

static Py_ssize_t
gc_collect_impl(PyObject *module, int generation)
/*[clinic end generated code: output=b697e633043233c7 input=40720128b682d879]*/
{
    PyThreadState *tstate = _PyThreadState_GET();

    if (generation < 0 || generation >= NUM_GENERATIONS) {
        _PyErr_SetString(tstate, PyExc_ValueError, "invalid generation");
        return -1;
    }

    GCState *gcstate = &tstate->interp->gc;
    Py_ssize_t n;
    if (gcstate->collecting) {
        /* already collecting, don't do anything */
        n = 0;
    }
    else {
        gcstate->collecting = 1;
        n = gc_collect_with_callback(tstate, generation);
        gcstate->collecting = 0;
    }
    return n;
}

/*[clinic input]
gc.set_debug

    flags: int
        An integer that can have the following bits turned on:
          DEBUG_STATS - Print statistics during collection.
          DEBUG_COLLECTABLE - Print collectable objects found.
          DEBUG_UNCOLLECTABLE - Print unreachable but uncollectable objects
            found.
          DEBUG_SAVEALL - Save objects to gc.garbage rather than freeing them.
          DEBUG_LEAK - Debug leaking programs (everything but STATS).
    /

Set the garbage collection debugging flags.

Debugging information is written to sys.stderr.
[clinic start generated code]*/

static PyObject *
gc_set_debug_impl(PyObject *module, int flags)
/*[clinic end generated code: output=7c8366575486b228 input=5e5ce15e84fbed15]*/
{
    GCState *gcstate = get_gc_state();
    gcstate->debug = flags;
    debug_verbose = flags & DEBUG_VERBOSE;
    Py_RETURN_NONE;
}

/*[clinic input]
gc.get_debug -> int

Get the garbage collection debugging flags.
[clinic start generated code]*/

static int
gc_get_debug_impl(PyObject *module)
/*[clinic end generated code: output=91242f3506cd1e50 input=91a101e1c3b98366]*/
{
    GCState *gcstate = get_gc_state();
    return gcstate->debug;
}

PyDoc_STRVAR(gc_set_thresh__doc__,
"set_threshold(threshold0, [threshold1, threshold2]) -> None\n"
"\n"
"Sets the collection thresholds.  Setting threshold0 to zero disables\n"
"collection.\n");

static PyObject *
gc_set_threshold(PyObject *self, PyObject *args)
{
    GCState *gcstate = get_gc_state();
    if (!PyArg_ParseTuple(args, "i|ii:set_threshold",
                          &gcstate->generations[0].threshold,
                          &gcstate->generations[1].threshold,
                          &gcstate->generations[2].threshold))
        return NULL;
    for (int i = 3; i < NUM_GENERATIONS; i++) {
        /* generations higher than 2 get the same threshold */
        gcstate->generations[i].threshold = gcstate->generations[2].threshold;
    }
    Py_RETURN_NONE;
}

/*[clinic input]
gc.get_threshold

Return the current collection thresholds.
[clinic start generated code]*/

static PyObject *
gc_get_threshold_impl(PyObject *module)
/*[clinic end generated code: output=7902bc9f41ecbbd8 input=286d79918034d6e6]*/
{
    GCState *gcstate = get_gc_state();
    return Py_BuildValue("(iii)",
                         gcstate->generations[0].threshold,
                         gcstate->generations[1].threshold,
                         gcstate->generations[2].threshold);
}

/*[clinic input]
gc.get_count

Return a three-tuple of the current collection counts.
[clinic start generated code]*/

static PyObject *
gc_get_count_impl(PyObject *module)
/*[clinic end generated code: output=354012e67b16398f input=a392794a08251751]*/
{
    GCState *gcstate = get_gc_state();
    return Py_BuildValue("(iii)",
                         gcstate->generations[0].count,
                         gcstate->generations[1].count,
                         gcstate->generations[2].count);
}

static int
referrersvisit(PyObject* obj, PyObject *objs)
{
    Py_ssize_t i;
    for (i = 0; i < PyTuple_GET_SIZE(objs); i++)
        if (PyTuple_GET_ITEM(objs, i) == obj)
            return 1;
    return 0;
}

static int
gc_referrers_for(PyObject *objs, PyGC_Head *list, PyObject *resultlist)
{
    PyGC_Head *gc;
    PyObject *obj;
    traverseproc traverse;
    for (gc = GC_NEXT(list); gc != list; gc = GC_NEXT(gc)) {
        obj = FROM_GC(gc);
        traverse = Py_TYPE(obj)->tp_traverse;
        if (obj == objs || obj == resultlist)
            continue;
        if (traverse(obj, (visitproc)referrersvisit, objs)) {
            if (PyList_Append(resultlist, obj) < 0)
                return 0; /* error */
        }
    }
    return 1; /* no error */
}

PyDoc_STRVAR(gc_get_referrers__doc__,
"get_referrers(*objs) -> list\n\
Return the list of objects that directly refer to any of objs.");

static PyObject *
gc_get_referrers(PyObject *self, PyObject *args)
{
    if (PySys_Audit("gc.get_referrers", "(O)", args) < 0) {
        return NULL;
    }

    PyObject *result = PyList_New(0);
    if (!result) {
        return NULL;
    }

    GCState *gcstate = get_gc_state();
    if (!(gc_referrers_for(args, HEAD(gcstate), result))) {
        Py_DECREF(result);
        return NULL;
    }
    return result;
}

/* Append obj to list; return true if error (out of memory), false if OK. */
static int
referentsvisit(PyObject *obj, PyObject *list)
{
    return PyList_Append(list, obj) < 0;
}

PyDoc_STRVAR(gc_get_referents__doc__,
"get_referents(*objs) -> list\n\
Return the list of objects that are directly referred to by objs.");

static PyObject *
gc_get_referents(PyObject *self, PyObject *args)
{
    Py_ssize_t i;
    if (PySys_Audit("gc.get_referents", "(O)", args) < 0) {
        return NULL;
    }
    PyObject *result = PyList_New(0);

    if (result == NULL)
        return NULL;

    for (i = 0; i < PyTuple_GET_SIZE(args); i++) {
        traverseproc traverse;
        PyObject *obj = PyTuple_GET_ITEM(args, i);

        if (!_PyObject_IS_GC(obj))
            continue;
        traverse = Py_TYPE(obj)->tp_traverse;
        if (! traverse)
            continue;
        if (traverse(obj, (visitproc)referentsvisit, result)) {
            Py_DECREF(result);
            return NULL;
        }
    }
    return result;
}

/*[clinic input]
gc.get_objects
    generation: Py_ssize_t(accept={int, NoneType}, c_default="-1") = None
        Generation to extract the objects from.

Return a list of objects tracked by the collector (excluding the list returned).

If generation is not None, return only the objects tracked by the collector
that are in that generation.
[clinic start generated code]*/

static PyObject *
gc_get_objects_impl(PyObject *module, Py_ssize_t generation)
/*[clinic end generated code: output=48b35fea4ba6cb0e input=ef7da9df9806754c]*/
{
    PyThreadState *tstate = _PyThreadState_GET();
    PyObject* result;
    GCState *gcstate = &tstate->interp->gc;

    if (PySys_Audit("gc.get_objects", "n", generation) < 0) {
        return NULL;
    }

    result = PyList_New(0);
    if (result == NULL) {
        return NULL;
    }

    /* If generation is passed, we extract only that generation */
    if (generation != -1) {
        if (generation >= NUM_GENERATIONS) {
            _PyErr_Format(tstate, PyExc_ValueError,
                          "generation parameter must be less than the number of "
                          "available generations (%i)",
                           NUM_GENERATIONS);
            goto error;
        }

        if (generation < 0) {
            _PyErr_SetString(tstate, PyExc_ValueError,
                             "generation parameter cannot be negative");
            goto error;
        }

    }
    if (append_objects(result, HEAD(gcstate), generation)) {
        goto error;
    }
    return result;

error:
    Py_DECREF(result);
    return NULL;
}

/*[clinic input]
gc.get_stats

Return a list of dictionaries containing per-generation statistics.
[clinic start generated code]*/

static PyObject *
gc_get_stats_impl(PyObject *module)
/*[clinic end generated code: output=a8ab1d8a5d26f3ab input=1ef4ed9d17b1a470]*/
{
    int i;
    struct gc_generation_stats stats[NUM_GENERATIONS], *st;

    /* To get consistent values despite allocations while constructing
       the result list, we use a snapshot of the running stats. */
    GCState *gcstate = get_gc_state();
    for (i = 0; i < NUM_GENERATIONS; i++) {
        stats[i] = gcstate->generation_stats[i];
    }

    PyObject *result = PyList_New(0);
    if (result == NULL)
        return NULL;

    for (i = 0; i < NUM_GENERATIONS; i++) {
        PyObject *dict;
        st = &stats[i];
        dict = Py_BuildValue("{snsnsn}",
                             "collections", st->collections,
                             "collected", st->collected,
                             "uncollectable", st->uncollectable
                            );
        if (dict == NULL)
            goto error;
        if (PyList_Append(result, dict)) {
            Py_DECREF(dict);
            goto error;
        }
        Py_DECREF(dict);
    }
    return result;

error:
    Py_XDECREF(result);
    return NULL;
}


/*[clinic input]
gc.is_tracked

    obj: object
    /

Returns true if the object is tracked by the garbage collector.

Simple atomic objects will return false.
[clinic start generated code]*/

static PyObject *
gc_is_tracked(PyObject *module, PyObject *obj)
/*[clinic end generated code: output=14f0103423b28e31 input=d83057f170ea2723]*/
{
    PyObject *result;

    if (_PyObject_IS_GC(obj) && _PyObject_GC_IS_TRACKED(obj))
        result = Py_True;
    else
        result = Py_False;
    Py_INCREF(result);
    return result;
}

/*[clinic input]
gc.is_finalized

    obj: object
    /

Returns true if the object has been already finalized by the GC.
[clinic start generated code]*/

static PyObject *
gc_is_finalized(PyObject *module, PyObject *obj)
/*[clinic end generated code: output=e1516ac119a918ed input=201d0c58f69ae390]*/
{
    if (_gc_is_finalized(obj)) {
         Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

/*[clinic input]
gc.freeze

Freeze all current tracked objects and ignore them for future collections.

This can be used before a POSIX fork() call to make the gc copy-on-write friendly.
Note: collection before a POSIX fork() call may free pages for future allocation
which can cause copy-on-write.
[clinic start generated code]*/

static PyObject *
gc_freeze_impl(PyObject *module)
/*[clinic end generated code: output=502159d9cdc4c139 input=b602b16ac5febbe5]*/
{
    GCState *gcstate = get_gc_state();
    PyGC_Head *head = HEAD(gcstate);
    for (PyGC_Head *gc = GC_NEXT(head); gc != head; gc=GC_NEXT(gc)) {
        int i = GET_GEN(gc);
        if (i < PERMANENT_GENERATION) {
            SET_GEN(gc, PERMANENT_GENERATION);
        }
    }
    for (int i = 0; i < NUM_GENERATIONS; i++) {
        gcstate->generations[i].count = 0;
    }
    Py_RETURN_NONE;
}

/*[clinic input]
gc.unfreeze

Unfreeze all objects in the permanent generation.

Put all objects in the permanent generation back into oldest generation.
[clinic start generated code]*/

static PyObject *
gc_unfreeze_impl(PyObject *module)
/*[clinic end generated code: output=1c15f2043b25e169 input=2dd52b170f4cef6c]*/
{
    GCState *gcstate = get_gc_state();
    PyGC_Head *head = HEAD(gcstate);
    for (PyGC_Head *gc = GC_NEXT(head); gc != head; gc=GC_NEXT(gc)) {
        int i = GET_GEN(gc);
        if (i == PERMANENT_GENERATION) {
            SET_GEN(gc, OLD_GENERATION);
        }
    }
    Py_RETURN_NONE;
}

/*[clinic input]
gc.get_freeze_count -> Py_ssize_t

Return the number of objects in the permanent generation.
[clinic start generated code]*/

static Py_ssize_t
gc_get_freeze_count_impl(PyObject *module)
/*[clinic end generated code: output=61cbd9f43aa032e1 input=45ffbc65cfe2a6ed]*/
{
    GCState *gcstate = get_gc_state();
    Py_ssize_t counts[NUM_GENERATIONS+1];
    memset(counts, 0, sizeof(counts));
    tally_gen_counts(gcstate, counts);
    return counts[PERMANENT_GENERATION];
}


PyDoc_STRVAR(gc__doc__,
"This module provides access to the garbage collector for reference cycles.\n"
"\n"
"enable() -- Enable automatic garbage collection.\n"
"disable() -- Disable automatic garbage collection.\n"
"isenabled() -- Returns true if automatic collection is enabled.\n"
"collect() -- Do a full collection right now.\n"
"get_count() -- Return the current collection counts.\n"
"get_stats() -- Return list of dictionaries containing per-generation stats.\n"
"set_debug() -- Set debugging flags.\n"
"get_debug() -- Get debugging flags.\n"
"set_threshold() -- Set the collection thresholds.\n"
"get_threshold() -- Return the current the collection thresholds.\n"
"get_objects() -- Return a list of all objects tracked by the collector.\n"
"is_tracked() -- Returns true if a given object is tracked.\n"
"is_finalized() -- Returns true if a given object has been already finalized.\n"
"get_referrers() -- Return the list of objects that refer to an object.\n"
"get_referents() -- Return the list of objects that an object refers to.\n"
"freeze() -- Freeze all tracked objects and ignore them for future collections.\n"
"unfreeze() -- Unfreeze all objects in the permanent generation.\n"
"get_freeze_count() -- Return the number of objects in the permanent generation.\n");

static PyMethodDef GcMethods[] = {
    GC_ENABLE_METHODDEF
    GC_DISABLE_METHODDEF
    GC_ISENABLED_METHODDEF
    GC_SET_DEBUG_METHODDEF
    GC_GET_DEBUG_METHODDEF
    GC_GET_COUNT_METHODDEF
    {"set_threshold",  gc_set_threshold, METH_VARARGS, gc_set_thresh__doc__},
    GC_GET_THRESHOLD_METHODDEF
    GC_COLLECT_METHODDEF
    GC_GET_OBJECTS_METHODDEF
    GC_GET_STATS_METHODDEF
    GC_IS_TRACKED_METHODDEF
    GC_IS_FINALIZED_METHODDEF
    {"get_referrers",  gc_get_referrers, METH_VARARGS,
        gc_get_referrers__doc__},
    {"get_referents",  gc_get_referents, METH_VARARGS,
        gc_get_referents__doc__},
    GC_FREEZE_METHODDEF
    GC_UNFREEZE_METHODDEF
    GC_GET_FREEZE_COUNT_METHODDEF
    {NULL,      NULL}           /* Sentinel */
};

static int
gcmodule_exec(PyObject *module)
{
    GCState *gcstate = get_gc_state();

    /* garbage and callbacks are initialized by _PyGC_Init() early in
     * interpreter lifecycle. */
    assert(gcstate->garbage != NULL);
    if (PyModule_AddObjectRef(module, "garbage", gcstate->garbage) < 0) {
        return -1;
    }
    assert(gcstate->callbacks != NULL);
    if (PyModule_AddObjectRef(module, "callbacks", gcstate->callbacks) < 0) {
        return -1;
    }

#define ADD_INT(NAME) if (PyModule_AddIntConstant(module, #NAME, NAME) < 0) { return -1; }
    ADD_INT(DEBUG_STATS);
    ADD_INT(DEBUG_COLLECTABLE);
    ADD_INT(DEBUG_UNCOLLECTABLE);
    ADD_INT(DEBUG_SAVEALL);
    ADD_INT(DEBUG_LEAK);
    ADD_INT(DEBUG_VERBOSE);
#undef ADD_INT
    return 0;
}

static PyModuleDef_Slot gcmodule_slots[] = {
    {Py_mod_exec, gcmodule_exec},
    {0, NULL}
};

static struct PyModuleDef gcmodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "gc",
    .m_doc = gc__doc__,
    .m_size = 0,  // per interpreter state, see: get_gc_state()
    .m_methods = GcMethods,
    .m_slots = gcmodule_slots
};

PyMODINIT_FUNC
PyInit_gc(void)
{
    return PyModuleDef_Init(&gcmodule);
}

/* C API for controlling the state of the garbage collector */
int
PyGC_Enable(void)
{
    GCState *gcstate = get_gc_state();
    int old_state = gcstate->enabled;
    gcstate->enabled = 1;
    return old_state;
}

int
PyGC_Disable(void)
{
    GCState *gcstate = get_gc_state();
    int old_state = gcstate->enabled;
    gcstate->enabled = 0;
    return old_state;
}

int
PyGC_IsEnabled(void)
{
    GCState *gcstate = get_gc_state();
    return gcstate->enabled;
}

/* Public API to invoke gc.collect() from C */
Py_ssize_t
PyGC_Collect(void)
{
    PyThreadState *tstate = _PyThreadState_GET();
    GCState *gcstate = &tstate->interp->gc;

    if (!gcstate->enabled) {
        return 0;
    }

    Py_ssize_t n;
    if (gcstate->collecting) {
        /* already collecting, don't do anything */
        n = 0;
    }
    else {
        PyObject *exc, *value, *tb;
        gcstate->collecting = 1;
        _PyErr_Fetch(tstate, &exc, &value, &tb);
        n = gc_collect_with_callback(tstate, NUM_GENERATIONS - 1);
        _PyErr_Restore(tstate, exc, value, tb);
        gcstate->collecting = 0;
    }

    return n;
}

Py_ssize_t
_PyGC_CollectNoFail(PyThreadState *tstate)
{
    /* Ideally, this function is only called on interpreter shutdown,
       and therefore not recursively.  Unfortunately, when there are daemon
       threads, a daemon thread can start a cyclic garbage collection
       during interpreter shutdown (and then never finish it).
       See http://bugs.python.org/issue8713#msg195178 for an example.
       */
    GCState *gcstate = &tstate->interp->gc;
    if (gcstate->collecting) {
        return 0;
    }

    Py_ssize_t n;
    gcstate->collecting = 1;
    n = gc_collect_main(tstate, NUM_GENERATIONS - 1, NULL, NULL, 1);
    gcstate->collecting = 0;
    return n;
}

void
_PyGC_DumpShutdownStats(PyInterpreterState *interp)
{
    GCState *gcstate = &interp->gc;
    if (!(gcstate->debug & DEBUG_SAVEALL)
        && gcstate->garbage != NULL && PyList_GET_SIZE(gcstate->garbage) > 0) {
        const char *message;
        if (gcstate->debug & DEBUG_UNCOLLECTABLE)
            message = "gc: %zd uncollectable objects at " \
                "shutdown";
        else
            message = "gc: %zd uncollectable objects at " \
                "shutdown; use gc.set_debug(gc.DEBUG_UNCOLLECTABLE) to list them";
        /* PyErr_WarnFormat does too many things and we are at shutdown,
           the warnings module's dependencies (e.g. linecache) may be gone
           already. */
        if (PyErr_WarnExplicitFormat(PyExc_ResourceWarning, "gc", 0,
                                     "gc", NULL, message,
                                     PyList_GET_SIZE(gcstate->garbage)))
            PyErr_WriteUnraisable(NULL);
        if (gcstate->debug & DEBUG_UNCOLLECTABLE) {
            PyObject *repr = NULL, *bytes = NULL;
            repr = PyObject_Repr(gcstate->garbage);
            if (!repr || !(bytes = PyUnicode_EncodeFSDefault(repr)))
                PyErr_WriteUnraisable(gcstate->garbage);
            else {
                PySys_WriteStderr(
                    "      %s\n",
                    PyBytes_AS_STRING(bytes)
                    );
            }
            Py_XDECREF(repr);
            Py_XDECREF(bytes);
        }
    }
}

void
_PyGC_Fini(PyInterpreterState *interp)
{
    GCState *gcstate = &interp->gc;
    Py_CLEAR(gcstate->garbage);
    Py_CLEAR(gcstate->callbacks);
}

/* for debugging */
void
_PyGC_Dump(PyGC_Head *g)
{
    _PyObject_Dump(FROM_GC(g));
}


#ifdef Py_DEBUG
static int
visit_validate(PyObject *op, void *parent_raw)
{
    PyObject *parent = _PyObject_CAST(parent_raw);
    if (_PyObject_IsFreed(op)) {
        _PyObject_ASSERT_FAILED_MSG(parent,
                                    "PyObject_GC_Track() object is not valid");
    }
    return 0;
}
#endif


/* extension modules might be compiled with GC support so these
   functions must always be available */

void
PyObject_GC_Track(void *op_raw)
{
    PyObject *op = _PyObject_CAST(op_raw);
    if (_PyObject_GC_IS_TRACKED(op)) {
        _PyObject_ASSERT_FAILED_MSG(op,
                                    "object already tracked "
                                    "by the garbage collector");
    }
    _PyObject_GC_TRACK(op);

#ifdef Py_DEBUG
    /* Check that the object is valid: validate objects traversed
       by tp_traverse() */
    traverseproc traverse = Py_TYPE(op)->tp_traverse;
    (void)traverse(op, visit_validate, op);
#endif
}

void
PyObject_GC_UnTrack(void *op_raw)
{
    PyObject *op = _PyObject_CAST(op_raw);
    /* Obscure:  the Py_TRASHCAN mechanism requires that we be able to
     * call PyObject_GC_UnTrack twice on an object.
     */
    if (_PyObject_GC_IS_TRACKED(op)) {
        _PyObject_GC_UNTRACK(op);
    }
}

int
PyObject_IS_GC(PyObject *obj)
{
    return _PyObject_IS_GC(obj);
}

void
_PyObject_GC_Link(PyObject *op)
{
    PyGC_Head *g = AS_GC(op);
    assert(((uintptr_t)g & (sizeof(uintptr_t)-1)) == 0);  // g must be correctly aligned

    PyThreadState *tstate = _PyThreadState_GET();
    GCState *gcstate = &tstate->interp->gc;
    g->_gc_next = 0;
    g->_gc_prev = 0;
    gcstate->generations[0].count++; /* number of allocated GC objects */
    if (gcstate->generations[0].count > gcstate->generations[0].threshold &&
        gcstate->enabled &&
        gcstate->generations[0].threshold &&
        !gcstate->collecting &&
        !_PyErr_Occurred(tstate))
    {
        gcstate->collecting = 1;
        gc_collect_generations(tstate);
        gcstate->collecting = 0;
    }
}

static PyObject *
gc_alloc(size_t basicsize, size_t presize)
{
    PyThreadState *tstate = _PyThreadState_GET();
    if (basicsize > PY_SSIZE_T_MAX - presize) {
        return _PyErr_NoMemory(tstate);
    }
    size_t size = presize + basicsize;
    char *mem = PyObject_Malloc(size);
    if (mem == NULL) {
        return _PyErr_NoMemory(tstate);
    }
    ((PyObject **)mem)[0] = NULL;
    ((PyObject **)mem)[1] = NULL;
    PyObject *op = (PyObject *)(mem + presize);
    _PyObject_GC_Link(op);
    return op;
}

PyObject *
_PyObject_GC_New(PyTypeObject *tp)
{
    size_t presize = _PyType_PreHeaderSize(tp);
    PyObject *op = gc_alloc(_PyObject_SIZE(tp), presize);
    if (op == NULL) {
        return NULL;
    }
    _PyObject_Init(op, tp);
    return op;
}

PyVarObject *
_PyObject_GC_NewVar(PyTypeObject *tp, Py_ssize_t nitems)
{
    size_t size;
    PyVarObject *op;

    if (nitems < 0) {
        PyErr_BadInternalCall();
        return NULL;
    }
    size_t presize = _PyType_PreHeaderSize(tp);
    size = _PyObject_VAR_SIZE(tp, nitems);
    op = (PyVarObject *)gc_alloc(size, presize);
    if (op == NULL) {
        return NULL;
    }
    _PyObject_InitVar(op, tp, nitems);
    return op;
}

PyVarObject *
_PyObject_GC_Resize(PyVarObject *op, Py_ssize_t nitems)
{
    const size_t basicsize = _PyObject_VAR_SIZE(Py_TYPE(op), nitems);
    _PyObject_ASSERT((PyObject *)op, !_PyObject_GC_IS_TRACKED(op));
    if (basicsize > PY_SSIZE_T_MAX - sizeof(PyGC_Head)) {
        return (PyVarObject *)PyErr_NoMemory();
    }

    PyGC_Head *g = AS_GC(op);
    g = (PyGC_Head *)PyObject_Realloc(g,  sizeof(PyGC_Head) + basicsize);
    if (g == NULL)
        return (PyVarObject *)PyErr_NoMemory();
    op = (PyVarObject *) FROM_GC(g);
    Py_SET_SIZE(op, nitems);
    return op;
}

void
PyObject_GC_Del(void *op)
{
    size_t presize = _PyType_PreHeaderSize(((PyObject *)op)->ob_type);
    PyGC_Head *g = AS_GC(op);
    if (_PyObject_GC_IS_TRACKED(op)) {
        gc_list_remove(g);
    }
    GCState *gcstate = get_gc_state();
    if (gcstate->generations[0].count > 0) {
        gcstate->generations[0].count--;
    }
    if (gc_in_cstate(g)) {
        _PyGC_SET_FLAG(g, GC_FLAG_NEED_FREE);
        SET_COLOR(g, COLOR_BLACK);
    }
    else {
        PyObject_Free(((char *)op)-presize);
    }
}

int
PyObject_GC_IsTracked(PyObject* obj)
{
    if (_PyObject_IS_GC(obj) && _PyObject_GC_IS_TRACKED(obj)) {
        return 1;
    }
    return 0;
}

int
PyObject_GC_IsFinalized(PyObject *obj)
{
    return _gc_is_finalized(obj);
}
