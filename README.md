# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py



MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/blackpearl/Documents/MLE/mod_3/minitorch/fast_ops.py (164)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/blackpearl/Documents/MLE/mod_3/minitorch/fast_ops.py (164) 
------------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                             | 
        out: Storage,                                                                     | 
        out_shape: Shape,                                                                 | 
        out_strides: Strides,                                                             | 
        in_storage: Storage,                                                              | 
        in_shape: Shape,                                                                  | 
        in_strides: Strides,                                                              | 
    ) -> None:                                                                            | 
        stride_aligned = (                                                                | 
            len(out_shape) == len(in_shape)                                               | 
            and np.array_equal(out_shape, in_shape)                                       | 
            and len(out_strides) == len(in_strides)                                       | 
            and np.array_equal(out_strides, in_strides)                                   | 
        )                                                                                 | 
        total_size = len(out)                                                             | 
                                                                                          | 
        # If stride-aligned, avoid indexing calculations                                  | 
        if stride_aligned:                                                                | 
            for i in prange(total_size):--------------------------------------------------| #1
                out[i] = fn(in_storage[i])                                                | 
            return                                                                        | 
                                                                                          | 
        all_indices = np.zeros((total_size, 2, MAX_DIMS), np.int32)-----------------------| #0
        for i in prange(total_size):------------------------------------------------------| #2
            to_index(i, out_shape, all_indices[i, 0])                                     | 
            broadcast_index(all_indices[i, 0], out_shape, in_shape, all_indices[i, 1])    | 
            in_ordinal = index_to_position(all_indices[i, 1], in_strides)                 | 
            out[i] = fn(in_storage[in_ordinal])                                           | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #1, #0, #2).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/blackpearl/Documents/MLE/mod_3/minitorch/fast_ops.py (219)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/blackpearl/Documents/MLE/mod_3/minitorch/fast_ops.py (219) 
-----------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                            | 
        out: Storage,                                                                    | 
        out_shape: Shape,                                                                | 
        out_strides: Strides,                                                            | 
        a_storage: Storage,                                                              | 
        a_shape: Shape,                                                                  | 
        a_strides: Strides,                                                              | 
        b_storage: Storage,                                                              | 
        b_shape: Shape,                                                                  | 
        b_strides: Strides,                                                              | 
    ) -> None:                                                                           | 
        stride_aligned = (                                                               | 
            (len(out_shape) == len(a_shape) == len(b_shape))                             | 
            and np.array_equal(out_shape, a_shape)                                       | 
            and np.array_equal(out_shape, b_shape)                                       | 
            and (len(out_strides) == len(a_strides) == len(b_strides))                   | 
            and np.array_equal(out_strides, a_strides)                                   | 
            and np.array_equal(out_strides, b_strides)                                   | 
        )                                                                                | 
        total_size = len(out)                                                            | 
                                                                                         | 
        # If stride-aligned, avoid indexing calculations                                 | 
        if stride_aligned:                                                               | 
            for i in prange(total_size):-------------------------------------------------| #4
                out[i] = fn(a_storage[i], b_storage[i])                                  | 
            return                                                                       | 
                                                                                         | 
        all_indices = np.zeros((total_size, 3, MAX_DIMS), np.int32)----------------------| #3
                                                                                         | 
        for i in prange(total_size):-----------------------------------------------------| #5
            to_index(i, out_shape, all_indices[i, 0])                                    | 
                                                                                         | 
            broadcast_index(all_indices[i, 0], out_shape, a_shape, all_indices[i, 1])    | 
            a_ordinal = index_to_position(all_indices[i, 1], a_strides)                  | 
                                                                                         | 
            broadcast_index(all_indices[i, 0], out_shape, b_shape, all_indices[i, 2])    | 
            b_ordinal = index_to_position(all_indices[i, 2], b_strides)                  | 
                                                                                         | 
            out[i] = fn(a_storage[a_ordinal], b_storage[b_ordinal])                      | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #4, #3, #5).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/blackpearl/Documents/MLE/mod_3/minitorch/fast_ops.py (283)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/blackpearl/Documents/MLE/mod_3/minitorch/fast_ops.py (283) 
----------------------------------------------------------------------------|loop #ID
    def _reduce(                                                            | 
        out: Storage,                                                       | 
        out_shape: Shape,                                                   | 
        out_strides: Strides,                                               | 
        a_storage: Storage,                                                 | 
        a_shape: Shape,                                                     | 
        a_strides: Strides,                                                 | 
        reduce_dim: int,                                                    | 
    ) -> None:                                                              | 
        reduce_size = a_shape[reduce_dim]                                   | 
        reduce_step = a_strides[reduce_dim]                                 | 
        total_size = len(out)                                               | 
        out_index_array = np.zeros((total_size, MAX_DIMS), np.int32)--------| #6
                                                                            | 
        for i in prange(total_size):----------------------------------------| #7
            to_index(i, out_shape, out_index_array[i])                      | 
            a_ordinal = index_to_position(out_index_array[i], a_strides)    | 
            accum = out[i]                                                  | 
            for _ in range(reduce_size):                                    | 
                accum = fn(accum, a_storage[a_ordinal])                     | 
                a_ordinal += reduce_step                                    | 
            out[i] = accum                                                  | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #6, #7).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/blackpearl/Documents/MLE/mod_3/minitorch/fast_ops.py (309)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/blackpearl/Documents/MLE/mod_3/minitorch/fast_ops.py (309) 
---------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                               | 
    out: Storage,                                                          | 
    out_shape: Shape,                                                      | 
    out_strides: Strides,                                                  | 
    a_storage: Storage,                                                    | 
    a_shape: Shape,                                                        | 
    a_strides: Strides,                                                    | 
    b_storage: Storage,                                                    | 
    b_shape: Shape,                                                        | 
    b_strides: Strides,                                                    | 
) -> None:                                                                 | 
    """NUMBA tensor matrix multiply function.                              | 
                                                                           | 
    Should work for any tensor shapes that broadcast as long as            | 
                                                                           | 
    ```                                                                    | 
    assert a_shape[-1] == b_shape[-2]                                      | 
    ```                                                                    | 
                                                                           | 
    Optimizations:                                                         | 
                                                                           | 
    * Outer loop in parallel                                               | 
    * No index buffers or function calls                                   | 
    * Inner loop should have no global writes, 1 multiply.                 | 
                                                                           | 
                                                                           | 
    Args:                                                                  | 
    ----                                                                   | 
        out (Storage): storage for `out` tensor                            | 
        out_shape (Shape): shape for `out` tensor                          | 
        out_strides (Strides): strides for `out` tensor                    | 
        a_storage (Storage): storage for `a` tensor                        | 
        a_shape (Shape): shape for `a` tensor                              | 
        a_strides (Strides): strides for `a` tensor                        | 
        b_storage (Storage): storage for `b` tensor                        | 
        b_shape (Shape): shape for `b` tensor                              | 
        b_strides (Strides): strides for `b` tensor                        | 
                                                                           | 
    Returns:                                                               | 
    -------                                                                | 
        None : Fills in `out`                                              | 
                                                                           | 
    """                                                                    | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                 | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                 | 
    o_batch_stride = out_strides[0]                                        | 
    batch_count = out_shape[0]                                             | 
                                                                           | 
    for batch_i in prange(batch_count):------------------------------------| #8
        o_batch_offset = batch_i * o_batch_stride                          | 
        a_batch_offset = batch_i * a_batch_stride                          | 
        b_batch_offset = batch_i * b_batch_stride                          | 
                                                                           | 
        for a_row_i in range(a_shape[-2]):                                 | 
            a_row_offset = a_batch_offset + a_row_i * a_strides[-2]        | 
            o_row_offset = o_batch_offset + a_row_i * out_strides[-2]      | 
            for b_col_j in range(b_shape[-1]):                             | 
                b_col_offset = b_batch_offset + b_col_j * b_strides[-1]    | 
                o_col_offset = b_col_j * out_strides[-1]                   | 
                dot_sum = 0.0                                              | 
                for k in range(a_shape[-1]):                               | 
                    a_idx = a_row_offset + k * a_strides[-1]               | 
                    b_idx = b_col_offset + k * b_strides[-2]               | 
                    dot_sum += a_storage[a_idx] * b_storage[b_idx]         | 
                out[o_row_offset + o_col_offset] = dot_sum                 | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #8).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None