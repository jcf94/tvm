# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Auto-scheduling Sparse Matrix Multiplication on CPU with Custom Sketch Rule
===========================================================================
**Author**: `Chengfan Jia <https://github.com/jcf94/>`_

This is a tutorial on how to use the auto-scheduler to tune a sparse matrix multiplication for
CPUs.

Auto-scheduler is designed to explore the schedule with best performance for a given computation
declaration automatically. While sometimes, we may have a demand to try some special ops which may
not been well-supported by auto-scheduler's default sketch rules and result in poor performance.
Fortunately, auto-scheduler currently allows user to provide a CustomSketch to cover these cases.

We use sparse matrix multiplication as an example in this tutorial to demonstrate how to implement
and plug a custom sketch rule to the auto-scheduler search policy.

Note that this tutorial will not run on Windows or recent versions of macOS. To
get it to run, you will need to wrap the body of this tutorial in a :code:`if
__name__ == "__main__":` block.
"""

import os
import itertools

import numpy as np
import tvm
from tvm import te, auto_scheduler, runtime, topi
from tvm.auto_scheduler import _ffi_api
from tvm.topi.utils import get_const_tuple

import scipy.sparse as sp

######################################################################
# Define the computation
# ^^^^^^^^^^^^^^^^^^^^^^
# To begin with, let us define the computation of a sparse matmul with several relu and bias add.
# The function should return the list of input/output tensors.
# From these tensors, the auto-scheduler can get the whole computational graph.

# We use this function to generate a random bsr matrix
def random_bsr_matrix(M, N, BS_R, BS_C, density, dtype):
    import itertools

    Y = np.zeros((M, N), dtype=dtype)
    assert M % BS_R == 0
    assert N % BS_C == 0
    nnz = int(density * M * N)
    num_blocks = int(nnz / (BS_R * BS_C)) + 1
    candidate_blocks = np.asarray(list(itertools.product(range(0, M, BS_R), range(0, N, BS_C))))
    assert candidate_blocks.shape[0] == M // BS_R * N // BS_C
    chosen_blocks = candidate_blocks[
        np.random.choice(candidate_blocks.shape[0], size=num_blocks, replace=False)
    ]
    for i in range(len(chosen_blocks)):
        r, c = chosen_blocks[i]
        Y[r : r + BS_R, c : c + BS_C] = np.random.randn(BS_R, BS_C)
    s = sp.bsr_matrix(Y, blocksize=(BS_R, BS_C))
    assert s.data.shape == (num_blocks, BS_R, BS_C)
    assert s.indices.shape == (num_blocks,)
    assert s.indptr.shape == (M // BS_R + 1,)
    return s

@auto_scheduler.register_workload
def sparse_dense(M, N, K, w_data_shape, w_indices_shape, w_indptr_shape, dtype):
    X = te.placeholder(shape=(M, K), dtype=dtype)
    W_data = te.placeholder(shape=w_data_shape, dtype=dtype)
    W_indices = te.placeholder(shape=w_indices_shape, dtype="int32")
    W_indptr = te.placeholder(shape=w_indptr_shape, dtype="int32")
    B = te.placeholder(shape=(M, N), dtype=dtype)

    out = topi.nn.sparse_dense(
        topi.nn.relu(X), W_data, W_indices, W_indptr
    )
    out = te.compute((M, N), lambda i, j: out[i, j] + B[i, j], name="BiasAdd")
    out = topi.nn.relu(out)

    return [X, W_data, W_indices, W_indptr, B, out]

######################################################################
# Special step for sparse workload
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# During schedule tuning, auto-scheduler will use random inputs to measure the performance of a
# generated schedule. While we cannot directly use a random array as the input of a sparse op, for
# the "indices" and "indptr" array are meaningful for the computation.
#
# To solve this problem, we register these as special buffers, and load them when process program
# measuring.
# See the :any:`auto_scheduler.measure` code for more details.

# Define the basic shapes of this sparse computation
M = K = N = 512
BS_R = 16
BS_C = 1
density = 0.6

# Generate the test data with numpy
X_np = np.random.randn(M, K).astype("float32")
X_np = np.maximum(np.zeros((M, K), dtype="float32"), X_np)  # Relu
W_sp_np = random_bsr_matrix(N, K, BS_R, BS_C, density=density, dtype="float32")
W_np = W_sp_np.todense()
Y_np = X_np @ W_np.T  # Process the matrix multiplication
B_np = np.random.randn(M, N).astype("float32")
Y_np = Y_np + B_np  # Bias add
Y_np = np.maximum(np.zeros((M, N), dtype="float32"), Y_np)  # Relu

######################################################################
# Create the search task
# ^^^^^^^^^^^^^^^^^^^^^^
# We then create a search task with M=N=K=512 and dtype="float32"
# If your machine supports avx instructions, you can
#
#   - replace "llvm" below with "llvm -mcpu=core-avx2" to enable AVX2
#   - replace "llvm" below with "llvm -mcpu=skylake-avx512" to enable AVX-512

target = tvm.target.Target("llvm")

task = tvm.auto_scheduler.SearchTask(
    func=sparse_dense,
    args=(
        M, N, K,
        W_sp_np.data.shape,
        W_sp_np.indices.shape,
        W_sp_np.indptr.shape,
        "float32"
    ),
    target=target
)

# Register the sparse data to special buffer
prefix = "sparse_dense_bsr_%d_%d_%d_%d_%d_%.2f_" % (M, N, K, BS_R, BS_C, density)
task.add_task_input(prefix + "W_data", runtime.ndarray.array(W_sp_np.data))
task.add_task_input(prefix + "W_indices", runtime.ndarray.array(W_sp_np.indices))
task.add_task_input(prefix + "W_indptr", runtime.ndarray.array(W_sp_np.indptr))

# Inspect the computational graph
print("Computational DAG:")
print(task.compute_dag)

print(task.task_inputs)

######################################################################
# Write the custom sketch for sparse dense op
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Before tuning, we will need to define the CustomSketchRule for the sparse dense op.
#
# CustomSketchRule consists of two parts: the condition function and the apply function.
#
#   - condition function: describe when to apply this sketch rule. For example, we can only apply
#     the rule to the sparse ops by matching their name and tag.
#   - apply function: describe how to generate the initial sketch. You can implement it using
#     auto-scheduler provided loop state APIs.

def meet_condition_func(search_policy, state, stage_id):
    state = auto_scheduler.loop_state.State(state, search_policy.search_task.compute_dag)
    if state.stages[stage_id].op.tag in [
        "sparse_dense_sp_rhs_bsrmm", "sparse_dense_sp_rhs_bsrmm_block"
    ]:
        return auto_scheduler.PreloadCustomSketchRule.APPLY_AND_SKIP_REST
    else:
        return auto_scheduler.PreloadCustomSketchRule.PASS

def apply_func(search_policy, state, stage_id):
    ret = []
    s0 = auto_scheduler.loop_state.State(state, search_policy.search_task.compute_dag)
    if s0.stages[stage_id].op.tag == "sparse_dense_sp_rhs_bsrmm_block":
        return [s0.state_object, stage_id - 1]

    sparse_dense = s0.stages[stage_id].op
    sparse_dense_block = s0.stages[stage_id - 1].op
    assert sparse_dense.tag == "sparse_dense_sp_rhs_bsrmm"
    assert sparse_dense_block.tag == "sparse_dense_sp_rhs_bsrmm_block"

    # Set the default consumer of compute block
    consumer = sparse_dense

    # If sparse dense has a single elementwise consumer
    # We can compute inline the sparse_dense output stage
    consumers = _ffi_api.SearchPolicyUtilsGetConsumers(
        search_policy.search_task, s0.state_object, stage_id
    )
    if len(consumers) == 1:
        consumer_id = int(consumers.items()[0][0])
        if _ffi_api.SearchPolicyUtilsIsElementwiseMatch(
            search_policy.search_task, s0.state_object, stage_id, consumer_id
        ):
            consumer = s0.stages[consumer_id].op
            s0.compute_inline(sparse_dense)

    i, nb_j, j, row_offset, c = s0[sparse_dense_block].iters
    m, n = s0[consumer].iters
    i0, i1, i2 = s0.split(sparse_dense_block, i, [None, None])
    m0, m1 = s0.follow_split(consumer, m, len(s0.transform_steps) - 1, 1)
    j0, j1 = s0.split(sparse_dense_block, nb_j, [None])
    n0, n1 = s0.follow_split(consumer, n, len(s0.transform_steps) - 1, 1)
    s0.reorder(sparse_dense_block, [i0, j0, i1, j1, row_offset, i2, j, c])
    s0.reorder(consumer, [m0, n0, m1, n1])
    s0.compute_at(sparse_dense_block, consumer, n0)

    ret.append([s0.state_object, stage_id - 2])

    return ret

######################################################################
# Next, we set parameters for the auto-scheduler with the custom sketch plugged in.
#
# * :code:`num_measure_trials` is the number of measurement trials we can use during the search.
#   We only make 10 trials in this tutorial for a fast demonstration. In practice, 1000 is a
#   good value for the search to converge. You can do more trials according to your time budget.
# * In addition, we use :code:`RecordToFile` to dump measurement records into a file `matmul.json`.
#   The measurement records can be used to query the history best, resume the search,
#   and do more analyses later.
# * see :any:`auto_scheduler.TuningOptions` for more parameters
# * Here, we need to create a :code:`auto_scheduler.SketchPolicy` object, and add the custom sketch
#   rule as a `init_search_callbacks`.

log_file = "sparse_dense.json"
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=2,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)

search_policy = auto_scheduler.SketchPolicy(
    task,
    program_cost_model=auto_scheduler.XGBModel(),
    init_search_callbacks=[
        auto_scheduler.PreloadCustomSketchRule(meet_condition_func, apply_func, "SparseDense")
    ]
)

######################################################################
# Run the search
# ^^^^^^^^^^^^^^
# Now we get all inputs ready.
# We can kick off the search and let the auto-scheduler do its magic.
# After some measurement trials, we can load the best schedule from the log
# file and apply it.

# Run auto-tuning (search)
task.tune(tune_option, search_policy)
# Apply the best schedule
sch, args = task.apply_best(log_file)

######################################################################
# We can lower the schedule to see the IR after auto-scheduling.
# The auto-scheduler correctly performs optimizations including multi-level tiling,
# layout transformation, parallelization, vectorization, unrolling, and operator fusion.

print("Lowered TIR:")
print(tvm.lower(sch, args, simple_mode=True))

######################################################################
# Check correctness and evaluate performance
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We build the binary and check its correctness and performance.

func = tvm.build(sch, args, target)

ctx = tvm.cpu()

X_tvm = tvm.nd.array(X_np, ctx=ctx)
W_data_tvm = tvm.nd.array(W_sp_np.data, ctx=ctx)
W_indices_tvm = tvm.nd.array(W_sp_np.indices, ctx=ctx)
W_indptr_tvm = tvm.nd.array(W_sp_np.indptr, ctx=ctx)
B_tvm = tvm.nd.array(B_np, ctx=ctx)
Y_tvm = tvm.nd.empty(Y_np.shape, ctx=ctx)

func(X_tvm, W_data_tvm, W_indices_tvm, W_indptr_tvm, B_tvm, Y_tvm)

# Check results
tvm.testing.assert_allclose(Y_np, Y_tvm.asnumpy(), atol=1e-4, rtol=1e-4)

# Evaluate execution time.
evaluator = func.time_evaluator(func.entry_name, ctx, min_repeat_ms=500)
print(
    "Execution time of this operator: %.3f ms"
    % (np.median(evaluator(X_tvm, W_data_tvm, W_indices_tvm, W_indptr_tvm, B_tvm, Y_tvm).results) * 1000)
)
