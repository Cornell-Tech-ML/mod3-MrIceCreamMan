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

# Diagnostics Output
Result of running `python project/parallel_check.py`
```
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
```

# Proof of CUDA matmul is faster than CPU matmul
The following is the result of running `timing.py` given by https://gist.github.com/justinchiu/e153cbfa667ee8212c5fe40e12252c8a
```
Running size 64
/usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 8 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
{'fast': np.float64(0.003639856974283854), 'gpu': np.float64(0.029495716094970703)}
Running size 128
/usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
{'fast': np.float64(0.01656182607014974), 'gpu': np.float64(0.016972064971923828)}
Running size 256
{'fast': np.float64(0.10535868008931477), 'gpu': np.float64(0.05886228879292806)}
Running size 512
{'fast': np.float64(1.1741130352020264), 'gpu': np.float64(0.2879769802093506)}
Running size 1024
{'fast': np.float64(9.245623668034872), 'gpu': np.float64(1.003982384999593)}

Timing summary
Size: 64
    fast: 0.00364
    gpu: 0.02950
Size: 128
    fast: 0.01656
    gpu: 0.01697
Size: 256
    fast: 0.10536
    gpu: 0.05886
Size: 512
    fast: 1.17411
    gpu: 0.28798
Size: 1024
    fast: 9.24562
    gpu: 1.00398
```

# Training result of regular size models on GPU
Result for running `python run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05`
```
Epoch  0  loss  4.425310182899812 correct 43 time per epoch 5.802286863327026
Epoch  10  loss  1.146739949447818 correct 47 time per epoch 1.9099524021148682
Epoch  20  loss  1.0920680971593688 correct 48 time per epoch 1.890258550643921
Epoch  30  loss  0.1381095533238562 correct 48 time per epoch 1.8939740657806396
Epoch  40  loss  2.397249962318893 correct 49 time per epoch 2.705644130706787
Epoch  50  loss  0.7175278404178144 correct 49 time per epoch 1.8811743259429932
Epoch  60  loss  0.9970706730565515 correct 48 time per epoch 1.867917776107788
Epoch  70  loss  0.9687434846680212 correct 49 time per epoch 1.9410827159881592
Epoch  80  loss  0.8824980663508651 correct 49 time per epoch 1.9433367252349854
Epoch  90  loss  0.09039074058189642 correct 49 time per epoch 2.6076948642730713
Epoch  100  loss  0.6442387109082192 correct 49 time per epoch 1.9633276462554932
Epoch  110  loss  0.8500924506093516 correct 49 time per epoch 1.8455405235290527
Epoch  120  loss  0.7259760253526997 correct 48 time per epoch 2.1178274154663086
Epoch  130  loss  0.07786807751478649 correct 48 time per epoch 1.8615734577178955
Epoch  140  loss  0.8645413287288227 correct 50 time per epoch 2.3066930770874023
Epoch  150  loss  0.120569963241615 correct 48 time per epoch 1.878885269165039
Epoch  160  loss  0.5016689563246577 correct 48 time per epoch 1.9484009742736816
Epoch  170  loss  1.1743557465989924 correct 50 time per epoch 2.297710418701172
Epoch  180  loss  1.6865480073456118 correct 48 time per epoch 1.965811014175415
Epoch  190  loss  0.833488936705183 correct 50 time per epoch 1.8809444904327393
Epoch  200  loss  0.936631394953281 correct 50 time per epoch 1.9454457759857178
Epoch  210  loss  1.185487665541291 correct 50 time per epoch 1.8632431030273438
Epoch  220  loss  0.7578062471828556 correct 50 time per epoch 2.682478904724121
Epoch  230  loss  0.021095822146426864 correct 48 time per epoch 1.8584721088409424
Epoch  240  loss  0.9927421628140296 correct 50 time per epoch 1.8671660423278809
Epoch  250  loss  0.715458243463477 correct 50 time per epoch 1.866140365600586
Epoch  260  loss  1.5013706950777483 correct 48 time per epoch 1.8791122436523438
Epoch  270  loss  0.054428303561044646 correct 50 time per epoch 2.365098476409912
Epoch  280  loss  0.5601499030374257 correct 50 time per epoch 1.8726441860198975
Epoch  290  loss  1.7244668349487027 correct 48 time per epoch 1.8887982368469238
Epoch  300  loss  0.009651417158117473 correct 50 time per epoch 2.2128257751464844
Epoch  310  loss  0.9495524459604222 correct 50 time per epoch 1.8831098079681396
Epoch  320  loss  0.004991058595098929 correct 48 time per epoch 1.8955905437469482
Epoch  330  loss  0.5294679100065558 correct 50 time per epoch 1.8499374389648438
Epoch  340  loss  1.8080630826672772 correct 48 time per epoch 1.8512349128723145
Epoch  350  loss  0.009836989918120327 correct 50 time per epoch 2.6831746101379395
Epoch  360  loss  0.003191008866457444 correct 49 time per epoch 1.866661548614502
Epoch  370  loss  0.0054857007319960415 correct 49 time per epoch 1.8590381145477295
Epoch  380  loss  0.31885309252342436 correct 50 time per epoch 1.8496580123901367
Epoch  390  loss  0.01078542794403995 correct 50 time per epoch 1.8559575080871582
Epoch  400  loss  0.7086347767577871 correct 50 time per epoch 2.696899890899658
Epoch  410  loss  0.8677751912435875 correct 50 time per epoch 1.9580364227294922
Epoch  420  loss  0.7367462451937269 correct 50 time per epoch 1.8511333465576172
Epoch  430  loss  0.669838311667682 correct 50 time per epoch 1.9593455791473389
Epoch  440  loss  0.0014518135940028627 correct 49 time per epoch 1.8690576553344727
Epoch  450  loss  0.337769162377658 correct 49 time per epoch 2.5609560012817383
Epoch  460  loss  0.9330044402794455 correct 50 time per epoch 1.8763322830200195
Epoch  470  loss  1.061104057944014 correct 50 time per epoch 1.9644317626953125
Epoch  480  loss  0.005540109772876927 correct 49 time per epoch 2.176276922225952
Epoch  490  loss  1.2139284771450902 correct 49 time per epoch 1.9576680660247803
```

Result for running `python run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05`
```
Epoch  0  loss  6.642088643839312 correct 35 time per epoch 4.826858043670654
Epoch  10  loss  7.612508812034084 correct 38 time per epoch 1.8584206104278564
Epoch  20  loss  5.708705228090032 correct 40 time per epoch 1.8698151111602783
Epoch  30  loss  4.0562864840951205 correct 44 time per epoch 1.8624191284179688
Epoch  40  loss  3.8039613034386033 correct 43 time per epoch 2.7022640705108643
Epoch  50  loss  4.331630223640514 correct 43 time per epoch 1.8769948482513428
Epoch  60  loss  2.1989058841488482 correct 43 time per epoch 1.8762667179107666
Epoch  70  loss  2.4814890588648866 correct 47 time per epoch 2.0887820720672607
Epoch  80  loss  4.41137240786972 correct 45 time per epoch 1.9605202674865723
Epoch  90  loss  2.923742014981793 correct 49 time per epoch 1.984046220779419
Epoch  100  loss  2.177838748275439 correct 49 time per epoch 1.9676156044006348
Epoch  110  loss  2.1524256705289035 correct 49 time per epoch 1.90199613571167
Epoch  120  loss  1.866968781622714 correct 47 time per epoch 2.793445110321045
Epoch  130  loss  1.9868995414484965 correct 49 time per epoch 1.9142367839813232
Epoch  140  loss  1.2435346587941802 correct 48 time per epoch 1.9864494800567627
Epoch  150  loss  1.1213440647829325 correct 49 time per epoch 2.1528160572052
Epoch  160  loss  0.8658943888074099 correct 49 time per epoch 1.9554381370544434
Epoch  170  loss  0.8938655169325675 correct 48 time per epoch 1.9275362491607666
Epoch  180  loss  1.3447077853160512 correct 49 time per epoch 1.9743778705596924
Epoch  190  loss  1.5443294432566759 correct 49 time per epoch 1.9068925380706787
Epoch  200  loss  0.20274720490374384 correct 49 time per epoch 2.8123202323913574
Epoch  210  loss  1.6716937246200934 correct 49 time per epoch 1.8813636302947998
Epoch  220  loss  3.3367913514309095 correct 49 time per epoch 1.8978796005249023
Epoch  230  loss  1.3210224001235 correct 49 time per epoch 1.910351037979126
Epoch  240  loss  0.7076778922497851 correct 49 time per epoch 1.8992788791656494
Epoch  250  loss  0.8524299995288788 correct 49 time per epoch 2.1733853816986084
Epoch  260  loss  0.5413200920760878 correct 50 time per epoch 1.8989286422729492
Epoch  270  loss  0.8108711776352141 correct 49 time per epoch 1.8814985752105713
Epoch  280  loss  0.4701982503948955 correct 49 time per epoch 2.7159485816955566
Epoch  290  loss  0.9415639673050833 correct 50 time per epoch 1.8895857334136963
Epoch  300  loss  1.511818674826271 correct 49 time per epoch 1.868889331817627
Epoch  310  loss  1.5082768918187244 correct 49 time per epoch 1.9207494258880615
Epoch  320  loss  1.6196209834592799 correct 48 time per epoch 1.8811628818511963
Epoch  330  loss  0.5385332625305405 correct 49 time per epoch 2.214721441268921
Epoch  340  loss  1.5561427879631062 correct 48 time per epoch 1.9111356735229492
Epoch  350  loss  0.24242495039374595 correct 49 time per epoch 1.9593195915222168
Epoch  360  loss  1.0942093125327657 correct 49 time per epoch 2.7072768211364746
Epoch  370  loss  0.6014547553527319 correct 49 time per epoch 1.8920502662658691
Epoch  380  loss  0.9107187243324767 correct 49 time per epoch 1.9043617248535156
Epoch  390  loss  1.6553024385888024 correct 46 time per epoch 1.9043848514556885
Epoch  400  loss  0.04289809539293849 correct 49 time per epoch 1.8711345195770264
Epoch  410  loss  0.3316444859554835 correct 50 time per epoch 2.3944263458251953
Epoch  420  loss  0.0899274353508734 correct 49 time per epoch 1.8833584785461426
Epoch  430  loss  0.3571143446859053 correct 50 time per epoch 1.98030686378479
Epoch  440  loss  0.1178057528694861 correct 50 time per epoch 2.6238908767700195
Epoch  450  loss  0.14496641971519372 correct 50 time per epoch 1.9563367366790771
Epoch  460  loss  1.2015025327663793 correct 49 time per epoch 1.8702912330627441
Epoch  470  loss  0.7259421264745877 correct 49 time per epoch 1.958768367767334
Epoch  480  loss  0.45783432751629527 correct 49 time per epoch 1.860485553741455
Epoch  490  loss  1.3330999690552918 correct 49 time per epoch 2.4594969749450684
```

Result for running `python run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05`
```
Epoch  0  loss  6.546706087870305 correct 32 time per epoch 5.556763410568237
Epoch  10  loss  4.281535406906728 correct 45 time per epoch 1.8978140354156494
Epoch  20  loss  3.685716508256242 correct 47 time per epoch 1.902047872543335
Epoch  30  loss  1.4209525283139954 correct 45 time per epoch 2.756239891052246
Epoch  40  loss  4.049296774218354 correct 46 time per epoch 1.9195806980133057
Epoch  50  loss  2.7157550790230904 correct 47 time per epoch 1.8913929462432861
Epoch  60  loss  1.6309243150885937 correct 49 time per epoch 2.1671299934387207
Epoch  70  loss  1.194015195057057 correct 49 time per epoch 1.9215145111083984
Epoch  80  loss  1.3376108618691662 correct 47 time per epoch 1.9687588214874268
Epoch  90  loss  2.337513131915451 correct 49 time per epoch 1.9735956192016602
Epoch  100  loss  1.72133293835343 correct 49 time per epoch 1.9838416576385498
Epoch  110  loss  1.799928282171041 correct 49 time per epoch 2.3599400520324707
Epoch  120  loss  0.9017937517663879 correct 49 time per epoch 2.0168018341064453
Epoch  130  loss  1.5874744938350043 correct 49 time per epoch 1.8998668193817139
Epoch  140  loss  1.8142668609835155 correct 49 time per epoch 2.754263401031494
Epoch  150  loss  1.555611289750229 correct 49 time per epoch 1.9569547176361084
Epoch  160  loss  0.49189588569098336 correct 49 time per epoch 1.9689948558807373
Epoch  170  loss  1.5167801024527057 correct 49 time per epoch 1.9828567504882812
Epoch  180  loss  2.392242137567045 correct 49 time per epoch 2.0546939373016357
Epoch  190  loss  2.4417248792580297 correct 46 time per epoch 1.9268109798431396
Epoch  200  loss  1.150622826293472 correct 49 time per epoch 1.981842279434204
Epoch  210  loss  0.29793520200230234 correct 50 time per epoch 1.9428226947784424
Epoch  220  loss  1.102923803288364 correct 49 time per epoch 2.55735182762146
Epoch  230  loss  0.8267215321806514 correct 50 time per epoch 1.8804481029510498
Epoch  240  loss  0.7733229839704656 correct 50 time per epoch 1.9446520805358887
Epoch  250  loss  0.5176587536741334 correct 49 time per epoch 2.399007797241211
Epoch  260  loss  0.047599179533677746 correct 50 time per epoch 1.8862802982330322
Epoch  270  loss  0.5006626844638545 correct 48 time per epoch 1.9358665943145752
Epoch  280  loss  0.6612768659316677 correct 50 time per epoch 1.8876402378082275
Epoch  290  loss  0.6080726748726362 correct 50 time per epoch 1.880208969116211
Epoch  300  loss  0.4987777123009661 correct 50 time per epoch 2.3210065364837646
Epoch  310  loss  0.9372522214542871 correct 48 time per epoch 1.8824760913848877
Epoch  320  loss  0.35747318153545904 correct 50 time per epoch 1.897374153137207
Epoch  330  loss  0.6954092879630231 correct 50 time per epoch 2.709073305130005
Epoch  340  loss  0.2774906664481861 correct 50 time per epoch 1.8763127326965332
Epoch  350  loss  0.7691227072212509 correct 50 time per epoch 1.9691660404205322
Epoch  360  loss  0.7830900136669972 correct 50 time per epoch 2.0368499755859375
Epoch  370  loss  0.44081594895528564 correct 50 time per epoch 1.897937297821045
Epoch  380  loss  0.5018458457416846 correct 50 time per epoch 2.07533597946167
Epoch  390  loss  0.42413336967916754 correct 50 time per epoch 1.9177191257476807
Epoch  400  loss  0.19489767743192984 correct 50 time per epoch 1.8800036907196045
Epoch  410  loss  0.48695431394530153 correct 50 time per epoch 2.7491230964660645
Epoch  420  loss  0.4060317645205694 correct 50 time per epoch 1.912114143371582
Epoch  430  loss  0.2701790946260617 correct 50 time per epoch 1.9565694332122803
Epoch  440  loss  0.314736246230366 correct 50 time per epoch 2.2635650634765625
Epoch  450  loss  0.37755486633601687 correct 50 time per epoch 2.018341541290283
Epoch  460  loss  0.48930422912525223 correct 50 time per epoch 1.87557053565979
Epoch  470  loss  0.7396447660385529 correct 50 time per epoch 1.9569206237792969
Epoch  480  loss  0.13301753602974295 correct 50 time per epoch 1.959768533706665
Epoch  490  loss  0.6592243010582469 correct 50 time per epoch 2.4020588397979736
```

# Training result of regular size models on CPU
Result for running `python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05`
```
Epoch  0  loss  6.1610841201706394 correct 43 time per epoch 28.151917934417725
Epoch  10  loss  1.933561568250816 correct 50 time per epoch 0.11862993240356445
Epoch  20  loss  1.2698393259843228 correct 50 time per epoch 0.12018799781799316
Epoch  30  loss  0.5194077832059514 correct 50 time per epoch 0.11577272415161133
Epoch  40  loss  0.41232973150563307 correct 50 time per epoch 0.11723780632019043
Epoch  50  loss  0.12578864586213678 correct 50 time per epoch 0.11853480339050293
Epoch  60  loss  0.3574774993863874 correct 50 time per epoch 0.11579155921936035
Epoch  70  loss  0.124682195064558 correct 50 time per epoch 0.11454153060913086
Epoch  80  loss  0.6130579845698984 correct 50 time per epoch 0.1267223358154297
Epoch  90  loss  0.5148693854147104 correct 50 time per epoch 0.12622737884521484
Epoch  100  loss  0.13003559408688584 correct 50 time per epoch 0.14166522026062012
Epoch  110  loss  0.4494788345415134 correct 50 time per epoch 0.20275568962097168
Epoch  120  loss  0.26139278297883406 correct 50 time per epoch 0.11464524269104004
Epoch  130  loss  0.3582860166637293 correct 50 time per epoch 0.11644768714904785
Epoch  140  loss  0.24327520055004428 correct 50 time per epoch 0.11707353591918945
Epoch  150  loss  0.23355890260173795 correct 50 time per epoch 0.11410236358642578
Epoch  160  loss  0.012315403079563915 correct 50 time per epoch 0.11393308639526367
Epoch  170  loss  0.23177601155963978 correct 50 time per epoch 0.11472058296203613
Epoch  180  loss  0.09145115537855228 correct 50 time per epoch 0.11686134338378906
Epoch  190  loss  0.0874881242018702 correct 50 time per epoch 0.11898541450500488
Epoch  200  loss  0.022665590170170707 correct 50 time per epoch 0.15008044242858887
Epoch  210  loss  0.014113731337630452 correct 50 time per epoch 0.18830084800720215
Epoch  220  loss  0.1587721468995593 correct 50 time per epoch 0.12795281410217285
Epoch  230  loss  0.07299287532189368 correct 50 time per epoch 0.11282682418823242
Epoch  240  loss  0.034948950776634014 correct 50 time per epoch 0.11232471466064453
Epoch  250  loss  0.0722402556095794 correct 50 time per epoch 0.11458683013916016
Epoch  260  loss  0.1479106985347374 correct 50 time per epoch 0.11328744888305664
Epoch  270  loss  0.09121223948908122 correct 50 time per epoch 0.11455535888671875
Epoch  280  loss  0.0011500019075520698 correct 50 time per epoch 0.11386394500732422
Epoch  290  loss  0.08100526623009228 correct 50 time per epoch 0.11830854415893555
Epoch  300  loss  0.02854517387244567 correct 50 time per epoch 0.11800909042358398
Epoch  310  loss  0.0011454103064337255 correct 50 time per epoch 0.18401503562927246
Epoch  320  loss  0.04218112510001067 correct 50 time per epoch 0.11473345756530762
Epoch  330  loss  0.16250201911796686 correct 50 time per epoch 0.11447453498840332
Epoch  340  loss  0.05111499839170552 correct 50 time per epoch 0.12375807762145996
Epoch  350  loss  0.034788346184135205 correct 50 time per epoch 0.12606120109558105
Epoch  360  loss  0.053842581454039466 correct 50 time per epoch 0.11269903182983398
Epoch  370  loss  0.11008421628214446 correct 50 time per epoch 0.12317109107971191
Epoch  380  loss  0.06272791793180131 correct 50 time per epoch 0.11796236038208008
Epoch  390  loss  0.14615854007008391 correct 50 time per epoch 0.12224578857421875
Epoch  400  loss  0.04012052480115601 correct 50 time per epoch 0.11823391914367676
Epoch  410  loss  0.022180320288918084 correct 50 time per epoch 0.27239441871643066
Epoch  420  loss  0.07986493806994284 correct 50 time per epoch 0.1198122501373291
Epoch  430  loss  0.004958298366101095 correct 50 time per epoch 0.12407374382019043
Epoch  440  loss  0.055461731417022875 correct 50 time per epoch 0.1306922435760498
Epoch  450  loss  0.08543513510003478 correct 50 time per epoch 0.11565494537353516
Epoch  460  loss  0.03560836596458526 correct 50 time per epoch 0.1146697998046875
Epoch  470  loss  0.04220135174664121 correct 50 time per epoch 0.11635589599609375
Epoch  480  loss  0.0010869722757897813 correct 50 time per epoch 0.1132209300994873
Epoch  490  loss  0.02982727304965447 correct 50 time per epoch 0.11481094360351562
```

Result for running `python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05`
```
Epoch  0  loss  6.732476552307876 correct 34 time per epoch 27.55536460876465
Epoch  10  loss  5.851196894118081 correct 38 time per epoch 0.11670207977294922
Epoch  20  loss  5.6263894539880095 correct 38 time per epoch 0.13901734352111816
Epoch  30  loss  4.695452314242541 correct 38 time per epoch 0.1814126968383789
Epoch  40  loss  4.006521481733707 correct 40 time per epoch 0.18302273750305176
Epoch  50  loss  3.5728816499576213 correct 42 time per epoch 0.11956310272216797
Epoch  60  loss  3.8673799730868494 correct 45 time per epoch 0.11786818504333496
Epoch  70  loss  3.764644454628034 correct 45 time per epoch 0.11553120613098145
Epoch  80  loss  4.717371903845236 correct 46 time per epoch 0.11900973320007324
Epoch  90  loss  1.9661028482667042 correct 45 time per epoch 0.12273144721984863
Epoch  100  loss  3.4251952767993976 correct 46 time per epoch 0.1328444480895996
Epoch  110  loss  2.4564382422662394 correct 47 time per epoch 0.12290310859680176
Epoch  120  loss  2.2363825469983842 correct 46 time per epoch 0.1182546615600586
Epoch  130  loss  2.6728601439680535 correct 48 time per epoch 0.11940288543701172
Epoch  140  loss  2.75034647250863 correct 48 time per epoch 0.21689105033874512
Epoch  150  loss  2.7761147912140114 correct 49 time per epoch 0.11223363876342773
Epoch  160  loss  2.530881761239122 correct 46 time per epoch 0.12705397605895996
Epoch  170  loss  0.8936640935998911 correct 48 time per epoch 0.11786222457885742
Epoch  180  loss  1.7980634609192072 correct 47 time per epoch 0.11634421348571777
Epoch  190  loss  0.8972940657942137 correct 48 time per epoch 0.11598944664001465
Epoch  200  loss  2.4214421031628843 correct 49 time per epoch 0.1304328441619873
Epoch  210  loss  1.0133600154634823 correct 49 time per epoch 0.12940216064453125
Epoch  220  loss  3.034120639141407 correct 48 time per epoch 0.1150963306427002
Epoch  230  loss  1.8513994979109416 correct 49 time per epoch 0.11427569389343262
Epoch  240  loss  1.0126140013398002 correct 48 time per epoch 0.22017884254455566
Epoch  250  loss  1.5065545709726467 correct 48 time per epoch 0.11535930633544922
Epoch  260  loss  3.4022585200431137 correct 46 time per epoch 0.11350703239440918
Epoch  270  loss  2.640377874822029 correct 48 time per epoch 0.11842918395996094
Epoch  280  loss  1.5620019891246457 correct 48 time per epoch 0.11361241340637207
Epoch  290  loss  0.9747723540398413 correct 49 time per epoch 0.11482620239257812
Epoch  300  loss  1.0994171723691908 correct 48 time per epoch 0.11511850357055664
Epoch  310  loss  1.0378671763025902 correct 49 time per epoch 0.12806200981140137
Epoch  320  loss  0.35208898763487834 correct 50 time per epoch 0.11559081077575684
Epoch  330  loss  0.35814467973432595 correct 48 time per epoch 0.11671638488769531
Epoch  340  loss  1.8146541588611864 correct 49 time per epoch 0.2501661777496338
Epoch  350  loss  1.5868745813440308 correct 49 time per epoch 0.11643552780151367
Epoch  360  loss  0.979679429767284 correct 50 time per epoch 0.1163322925567627
Epoch  370  loss  0.3970124076367918 correct 49 time per epoch 0.11474227905273438
Epoch  380  loss  2.1956865568859283 correct 49 time per epoch 0.11431503295898438
Epoch  390  loss  0.29421824690016124 correct 49 time per epoch 0.11570882797241211
Epoch  400  loss  0.37104759378920843 correct 50 time per epoch 0.11426138877868652
Epoch  410  loss  0.7267673943901265 correct 48 time per epoch 0.11562561988830566
Epoch  420  loss  1.4943481870417639 correct 47 time per epoch 0.11503100395202637
Epoch  430  loss  0.36360139436814287 correct 50 time per epoch 0.12908434867858887
Epoch  440  loss  0.21350875747469705 correct 47 time per epoch 0.2542431354522705
Epoch  450  loss  1.5494149119602185 correct 50 time per epoch 0.11519885063171387
Epoch  460  loss  1.3424536548845267 correct 49 time per epoch 0.11486601829528809
Epoch  470  loss  0.20144103848369607 correct 48 time per epoch 0.11547303199768066
Epoch  480  loss  1.4897330987558448 correct 49 time per epoch 0.1173865795135498
Epoch  490  loss  0.929129921571074 correct 49 time per epoch 0.11592674255371094
```

Result for running `python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05`
```
Epoch  0  loss  6.900165572925429 correct 23 time per epoch 27.754984378814697
Epoch  10  loss  6.723273935211606 correct 41 time per epoch 0.12819552421569824
Epoch  20  loss  4.7659902713747515 correct 41 time per epoch 0.1139523983001709
Epoch  30  loss  3.172736719295947 correct 43 time per epoch 0.11464333534240723
Epoch  40  loss  2.616841192931276 correct 42 time per epoch 0.11676764488220215
Epoch  50  loss  3.2644049263324866 correct 46 time per epoch 0.11430549621582031
Epoch  60  loss  1.872038939752086 correct 46 time per epoch 0.20544099807739258
Epoch  70  loss  1.3443822600283197 correct 49 time per epoch 0.2438957691192627
Epoch  80  loss  2.169070720875715 correct 47 time per epoch 0.11474084854125977
Epoch  90  loss  2.5257748386058574 correct 48 time per epoch 0.12905144691467285
Epoch  100  loss  1.614105567935457 correct 49 time per epoch 0.11915135383605957
Epoch  110  loss  1.3960181687724933 correct 50 time per epoch 0.12450170516967773
Epoch  120  loss  1.6196363540634926 correct 50 time per epoch 0.11649298667907715
Epoch  130  loss  1.2768209354103224 correct 49 time per epoch 0.1164388656616211
Epoch  140  loss  0.9597409000865323 correct 49 time per epoch 0.11676836013793945
Epoch  150  loss  0.9647230870669886 correct 49 time per epoch 0.11501312255859375
Epoch  160  loss  1.1004100287741163 correct 49 time per epoch 0.20094966888427734
Epoch  170  loss  1.2032656275452434 correct 48 time per epoch 0.25922679901123047
Epoch  180  loss  1.185950404024116 correct 50 time per epoch 0.1165471076965332
Epoch  190  loss  1.6539177659307984 correct 50 time per epoch 0.11563634872436523
Epoch  200  loss  0.565682536431298 correct 49 time per epoch 0.11903762817382812
Epoch  210  loss  0.7684725049833742 correct 50 time per epoch 0.11585688591003418
Epoch  220  loss  1.4429843260248014 correct 49 time per epoch 0.11644196510314941
Epoch  230  loss  1.324389834582235 correct 50 time per epoch 0.11458587646484375
Epoch  240  loss  0.9112931991288917 correct 50 time per epoch 0.11625123023986816
Epoch  250  loss  1.0758539386129966 correct 50 time per epoch 0.11467790603637695
Epoch  260  loss  0.8778161936184463 correct 49 time per epoch 0.18583989143371582
Epoch  270  loss  0.7266778149553736 correct 50 time per epoch 0.2918717861175537
Epoch  280  loss  1.3799654734869975 correct 50 time per epoch 0.12390017509460449
Epoch  290  loss  0.09824723286436396 correct 50 time per epoch 0.12194132804870605
Epoch  300  loss  0.2742352993773475 correct 50 time per epoch 0.12421607971191406
Epoch  310  loss  0.6254635804109847 correct 50 time per epoch 0.12098836898803711
Epoch  320  loss  0.58384992956322 correct 49 time per epoch 0.12034082412719727
Epoch  330  loss  0.11738089298948676 correct 50 time per epoch 0.13416457176208496
Epoch  340  loss  0.8659480351704897 correct 50 time per epoch 0.12218761444091797
Epoch  350  loss  0.3263452372714901 correct 50 time per epoch 0.11573362350463867
Epoch  360  loss  0.40317518133145663 correct 50 time per epoch 0.19086813926696777
Epoch  370  loss  0.5220826771592323 correct 50 time per epoch 0.11950254440307617
Epoch  380  loss  0.8963233400598862 correct 50 time per epoch 0.11645340919494629
Epoch  390  loss  0.9661513098290497 correct 50 time per epoch 0.11776256561279297
Epoch  400  loss  0.12831181942715345 correct 50 time per epoch 0.11569523811340332
Epoch  410  loss  0.4970116580160524 correct 50 time per epoch 0.11840033531188965
Epoch  420  loss  0.21896080591781444 correct 50 time per epoch 0.11727023124694824
Epoch  430  loss  0.7837990428799015 correct 50 time per epoch 0.12534284591674805
Epoch  440  loss  0.5859786967863078 correct 50 time per epoch 0.12255644798278809
Epoch  450  loss  0.14440010428341507 correct 50 time per epoch 0.11547231674194336
Epoch  460  loss  0.5574307000254981 correct 50 time per epoch 0.16771483421325684
Epoch  470  loss  0.6369839045936916 correct 50 time per epoch 0.23931360244750977
Epoch  480  loss  0.4595414761422066 correct 50 time per epoch 0.11479759216308594
Epoch  490  loss  0.16175665520776777 correct 50 time per epoch 0.11522912979125977
```

# Training result of large size models on CPU
Result for running `python run_fast_tensor.py --BACKEND gpu --HIDDEN 200 --DATASET split --RATE 0.05`
```
Epoch  0  loss  5.149477005898868 correct 40 time per epoch 5.849961996078491
Epoch  10  loss  3.726717207374521 correct 48 time per epoch 1.9245738983154297
Epoch  20  loss  2.754787190347509 correct 48 time per epoch 1.936750888824463
Epoch  30  loss  1.2774405715816965 correct 48 time per epoch 1.9239726066589355
Epoch  40  loss  0.9424112653288195 correct 49 time per epoch 1.9457883834838867
Epoch  50  loss  2.3077425742302693 correct 49 time per epoch 1.957155704498291
Epoch  60  loss  0.8380496023084956 correct 50 time per epoch 1.9361672401428223
Epoch  70  loss  0.22446786941002456 correct 50 time per epoch 1.9943675994873047
Epoch  80  loss  0.8028016459790274 correct 50 time per epoch 2.033174514770508
Epoch  90  loss  0.43676113386376125 correct 50 time per epoch 1.9493827819824219
Epoch  100  loss  0.2700933704795736 correct 50 time per epoch 2.1108522415161133
Epoch  110  loss  0.31250479512837553 correct 50 time per epoch 1.9773669242858887
Epoch  120  loss  0.5227508782756202 correct 50 time per epoch 2.04774808883667
Epoch  130  loss  0.3670741798600807 correct 50 time per epoch 2.2209692001342773
Epoch  140  loss  0.3565362326512417 correct 50 time per epoch 2.0363683700561523
Epoch  150  loss  0.2626755054257597 correct 50 time per epoch 1.9374027252197266
Epoch  160  loss  0.18354567367328514 correct 50 time per epoch 2.473013162612915
Epoch  170  loss  0.14033618514636714 correct 50 time per epoch 1.9167938232421875
Epoch  180  loss  0.13172668686018202 correct 50 time per epoch 2.0814154148101807
Epoch  190  loss  0.19329573695573318 correct 50 time per epoch 2.621274709701538
Epoch  200  loss  0.15613612865067067 correct 50 time per epoch 2.0276434421539307
Epoch  210  loss  0.05807470349506262 correct 50 time per epoch 1.990060567855835
Epoch  220  loss  0.12625307878382955 correct 50 time per epoch 2.7665793895721436
Epoch  230  loss  0.07730292098365728 correct 50 time per epoch 1.9547462463378906
Epoch  240  loss  0.1909669184658459 correct 50 time per epoch 2.0320355892181396
Epoch  250  loss  0.07665807555656173 correct 50 time per epoch 2.6568753719329834
Epoch  260  loss  0.06927453811691335 correct 50 time per epoch 1.9375479221343994
Epoch  270  loss  0.06847964735170999 correct 50 time per epoch 1.9598476886749268
Epoch  280  loss  0.14268608164561758 correct 50 time per epoch 2.542421817779541
Epoch  290  loss  0.15686098699840526 correct 50 time per epoch 1.9859068393707275
Epoch  300  loss  0.05850661145172087 correct 50 time per epoch 1.9529364109039307
Epoch  310  loss  0.09879013611362056 correct 50 time per epoch 2.479795455932617
Epoch  320  loss  0.16170462761802532 correct 50 time per epoch 1.973942756652832
Epoch  330  loss  0.059128498307203674 correct 50 time per epoch 1.948439598083496
Epoch  340  loss  0.04255794493781022 correct 50 time per epoch 2.3738396167755127
Epoch  350  loss  0.08099819894680414 correct 50 time per epoch 2.07798433303833
Epoch  360  loss  0.007617351100577814 correct 50 time per epoch 1.9275953769683838
Epoch  370  loss  0.08468889813739727 correct 50 time per epoch 2.1440467834472656
Epoch  380  loss  0.11400712229032406 correct 50 time per epoch 1.940352439880371
Epoch  390  loss  0.0522904661370582 correct 50 time per epoch 1.9486784934997559
Epoch  400  loss  0.10419909265352713 correct 50 time per epoch 1.9860498905181885
Epoch  410  loss  0.015051714580236913 correct 50 time per epoch 2.037952184677124
Epoch  420  loss  0.08371089170296318 correct 50 time per epoch 1.9366796016693115
Epoch  430  loss  0.02899827916452554 correct 50 time per epoch 2.0168213844299316
Epoch  440  loss  0.07221165811964471 correct 50 time per epoch 1.9380340576171875
Epoch  450  loss  0.09896201334161836 correct 50 time per epoch 2.026855707168579
Epoch  460  loss  0.04863281335286463 correct 50 time per epoch 1.937967300415039
Epoch  470  loss  0.06823895597194635 correct 50 time per epoch 2.0491247177124023
Epoch  480  loss  0.08028063182632031 correct 50 time per epoch 1.985868215560913
Epoch  490  loss  0.07968320979444923 correct 50 time per epoch 2.05904221534729
```

Result for running `python run_fast_tensor.py --BACKEND cpu --HIDDEN 200 --DATASET split --RATE 0.05`
```
Epoch  0  loss  6.1262810333287465 correct 31 time per epoch 28.6850745677948
Epoch  10  loss  10.08611454385301 correct 22 time per epoch 0.23846197128295898
Epoch  20  loss  3.2854564456772377 correct 49 time per epoch 0.22823357582092285
Epoch  30  loss  2.616636185337869 correct 45 time per epoch 0.24227261543273926
Epoch  40  loss  2.2688683821795985 correct 48 time per epoch 0.4501800537109375
Epoch  50  loss  1.371849129031032 correct 47 time per epoch 0.2322063446044922
Epoch  60  loss  2.397817845868659 correct 45 time per epoch 0.23090624809265137
Epoch  70  loss  1.2195419611675549 correct 50 time per epoch 0.22929048538208008
Epoch  80  loss  1.455974531669368 correct 50 time per epoch 0.24265599250793457
Epoch  90  loss  1.3708732036806344 correct 50 time per epoch 0.3947103023529053
Epoch  100  loss  1.5635330836234764 correct 46 time per epoch 0.22842955589294434
Epoch  110  loss  1.0824081386604725 correct 49 time per epoch 0.23432612419128418
Epoch  120  loss  1.0544165357968862 correct 50 time per epoch 0.2285144329071045
Epoch  130  loss  0.7091076339276094 correct 50 time per epoch 0.22821426391601562
Epoch  140  loss  0.8233798711692394 correct 50 time per epoch 0.23898649215698242
Epoch  150  loss  0.7447079324935632 correct 50 time per epoch 0.22665834426879883
Epoch  160  loss  0.8491984673809203 correct 50 time per epoch 0.2313065528869629
Epoch  170  loss  0.9352962945423329 correct 50 time per epoch 0.22755980491638184
Epoch  180  loss  0.3025752180801282 correct 50 time per epoch 0.22767066955566406
Epoch  190  loss  0.6835760784527021 correct 50 time per epoch 0.23097443580627441
Epoch  200  loss  0.5678189495855697 correct 49 time per epoch 0.24388551712036133
Epoch  210  loss  1.0984323124444828 correct 49 time per epoch 0.22539567947387695
Epoch  220  loss  0.18519794509846252 correct 49 time per epoch 0.228745698928833
Epoch  230  loss  0.7313116896520779 correct 50 time per epoch 0.23305916786193848
Epoch  240  loss  0.4427499915084566 correct 50 time per epoch 0.2471637725830078
Epoch  250  loss  0.2647927364585604 correct 50 time per epoch 0.2381000518798828
Epoch  260  loss  0.35622300940456253 correct 50 time per epoch 0.2236933708190918
Epoch  270  loss  0.3174594825033068 correct 50 time per epoch 0.24106955528259277
Epoch  280  loss  0.25149109663354363 correct 50 time per epoch 0.224961519241333
Epoch  290  loss  0.2752593625547975 correct 50 time per epoch 0.22905588150024414
Epoch  300  loss  0.3089024923093687 correct 50 time per epoch 0.24311089515686035
Epoch  310  loss  0.1803973121858805 correct 50 time per epoch 0.23340845108032227
Epoch  320  loss  0.4616166459990253 correct 50 time per epoch 0.2332615852355957
Epoch  330  loss  0.774683911008103 correct 49 time per epoch 0.2296431064605713
Epoch  340  loss  0.6379835454206645 correct 50 time per epoch 0.23067378997802734
Epoch  350  loss  0.13391008904607102 correct 50 time per epoch 0.24434566497802734
Epoch  360  loss  0.9837789919141621 correct 50 time per epoch 0.2235400676727295
Epoch  370  loss  0.033515461844967225 correct 50 time per epoch 0.22951817512512207
Epoch  380  loss  0.04591189076274042 correct 50 time per epoch 0.2303321361541748
Epoch  390  loss  0.41134510099971633 correct 50 time per epoch 0.22723674774169922
Epoch  400  loss  0.23000229967281421 correct 50 time per epoch 0.2284245491027832
Epoch  410  loss  0.10477062569793649 correct 50 time per epoch 0.24272727966308594
Epoch  420  loss  0.15951924263651998 correct 50 time per epoch 0.2308812141418457
Epoch  430  loss  0.27040564719434457 correct 50 time per epoch 0.23323535919189453
Epoch  440  loss  0.0955462494146589 correct 50 time per epoch 0.2277359962463379
Epoch  450  loss  0.10510125710118447 correct 50 time per epoch 0.3397669792175293
Epoch  460  loss  0.27775918510126907 correct 50 time per epoch 0.22763347625732422
Epoch  470  loss  0.016792314470310574 correct 50 time per epoch 0.24110984802246094
Epoch  480  loss  0.22151748748347444 correct 50 time per epoch 0.2348334789276123
Epoch  490  loss  0.023394060187058467 correct 50 time per epoch 0.24062490463256836
```
