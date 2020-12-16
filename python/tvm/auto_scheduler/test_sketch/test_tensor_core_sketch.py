"""Use auto scheduler to generate tensor core intrinsics"""

import tvm
from tvm import auto_scheduler, te

def intrin_wmma_load_matrix(scope):
    n = 16
    A = te.placeholder((n, n), name='A', dtype='float16')
    BA = tvm.tir.decl_buffer(A.shape, A.dtype, scope='shared', data_alignment=32, offset_factor=256)
    C = te.compute((n, n), lambda i, j: A[i, j], name='C')
    BC = tvm.tir.decl_buffer(C.shape, C.dtype, scope=scope, data_alignment=32, offset_factor=256)

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()

        BA = ins[0]
        BC = outs[0]
        ib.emit(tvm.tir.call_intrin('handle', 'tir.tvm_load_matrix_sync',
                                    BC.data, n, n, n, BC.elem_offset // 256,
                                    BA.access_ptr('r'), n, 'row_major'))
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})

@tvm._ffi.register_func
def intrin_wmma_load_matrix_a():
    # print("?????")
    return intrin_wmma_load_matrix("wmma.matrix_a")

@tvm._ffi.register_func
def intrin_wmma_load_matrix_b():
    return intrin_wmma_load_matrix("wmma.matrix_b")

@tvm._ffi.register_func
def intrin_wmma_gemm():
    n = 16
    A = te.placeholder((n, n), name='A', dtype='float16')
    B = te.placeholder((n, n), name='B', dtype='float16')
    k = te.reduce_axis((0, n), name="k")
    C = te.compute((n, n),
                   lambda ii, jj:
                   te.sum(A[ii, k].astype('float') * B[k, jj].astype('float'), axis=k),
                   name='C')
    BA = tvm.tir.decl_buffer(A.shape, A.dtype, name='BA', scope='wmma.matrix_a', data_alignment=32, offset_factor=256)
    BB = tvm.tir.decl_buffer(B.shape, B.dtype, name='BB', scope='wmma.matrix_b', data_alignment=32, offset_factor=256)
    BC = tvm.tir.decl_buffer(C.shape, C.dtype, name='BC', scope='wmma.accumulator', data_alignment=32, offset_factor=256)

    def intrin_func(ins, outs):
        BA, BB = ins
        BC, = outs

        def init():
            ib = tvm.tir.ir_builder.create()
            ib.emit(tvm.tir.call_intrin('handle', 'tir.tvm_fill_fragment', BC.data, n, n, n, BC.elem_offset // 256, 0.0))
            return ib.get()

        def update():
            ib = tvm.tir.ir_builder.create()
            ib.emit(tvm.tir.call_intrin('handle', 'tir.tvm_mma_sync',
                                        BC.data, BC.elem_offset // 256,
                                        BA.data, BA.elem_offset // 256,
                                        BB.data, BB.elem_offset // 256,
                                        BC.data, BC.elem_offset // 256))
            return ib.get()

        return update(), init(), update()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, B: BB, C: BC})

@tvm._ffi.register_func
def intrin_wmma_store_matrix():
    n = 16
    A = te.placeholder((n, n), name='A', dtype='float32')
    BA = tvm.tir.decl_buffer(A.shape, A.dtype, scope='wmma.accumulator', data_alignment=32, offset_factor=256)
    C = te.compute((n, n), lambda i, j: A[i, j], name='C')
    BC = tvm.tir.decl_buffer(C.shape, C.dtype, scope='global', data_alignment=32, offset_factor=256)

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()
        BA = ins[0]
        BC = outs[0]
        ib.emit(tvm.tir.call_intrin('handle', 'tir.tvm_store_matrix_sync',
                                    BA.data, n, n, n, BA.elem_offset // 256,
                                    BC.access_ptr('w'), n, 'row_major'))
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})

@tvm._ffi.register_func
def tensor_core_apply(search_policy, state, stage_id):
    ret = []
    state = auto_scheduler.loop_state.State(state, search_policy.search_task.compute_dag)

    A, B, C = search_policy.search_task.compute_dag.ops

    C_local = state.cache_write(C, "wmma.accumulator")

    its0 = state.split(C_local, state[C_local].iters[0], [None, None])
    split_step0 = len(state.transform_steps) - 1
    its1 = state.split(C_local, state[C_local].iters[3], [None, None])
    split_step1 = len(state.transform_steps) - 1
    its2 = state.split(C_local, state[C_local].iters[8], [None])

    state.reorder(C_local, [its0[0], its1[0], its0[1], its1[1], its0[2], its1[2],
                            its2[0], its2[1],
                            state[C_local].iters[6],
                            state[C_local].iters[7],
                            state[C_local].iters[10]])
    state.fuse(C_local, [state[C_local].iters[0], state[C_local].iters[1]])
    state.fuse(C_local, [state[C_local].iters[1], state[C_local].iters[2]])
    state.fuse(C_local, [state[C_local].iters[2], state[C_local].iters[3]])

    its0 = state.follow_split(C, state[C].iters[0], split_step0, 2)
    its1 = state.follow_split(C, state[C].iters[3], split_step1, 2)
    state.reorder(C, [its0[0], its1[0], its0[1], its1[1], its0[2], its1[2],
                      state[C].iters[6], state[C].iters[7]])
    state.fuse(C, [state[C].iters[0], state[C].iters[1]])
    state.fuse(C, [state[C].iters[1], state[C].iters[2]])
    local_write_pos = state.fuse(C, [state[C].iters[2], state[C].iters[3]])
    state.compute_at(C_local, C, local_write_pos)
    shared_read_pos = state[C_local].iters[3]
    local_read_pos = state[C_local].iters[4]
    state.bind(C, state[C].iters[0], "blockIdx.x")
    state.bind(C, state[C].iters[1], "vthread")
    state.bind(C, state[C].iters[2], "threadIdx.x")

    B_shared = state.cache_read(B, "shared", [C_local])
    B_local = state.cache_read(B_shared, "wmma.matrix_b", [C_local])
    state.compute_at(B_shared, C_local, shared_read_pos)
    state.compute_at(B_local, C_local, local_read_pos)

    it = state.fuse(B_shared, state[B_shared].iters[:])
    its = state.split(B_shared, it, [4]) # vectorize add a callback check function
    state.vectorize(B_shared, its[1])
    its = state.follow_fused_split(B_shared, its[0], [split_step0, split_step1], 1, True)
    state.bind(B_shared, its[1], "threadIdx.x")

    A_shared = state.cache_read(A, "shared", [C_local])
    A_local = state.cache_read(A_shared, "wmma.matrix_a", [C_local])
    state.compute_at(A_shared, C_local, shared_read_pos)
    state.compute_at(A_local, C_local, local_read_pos)

    it = state.fuse(A_shared, state[A_shared].iters[:])
    its = state.split(A_shared, it, [4]) # vectorize add a callback check function
    state.vectorize(A_shared, its[1])
    its = state.follow_fused_split(A_shared, its[0], [split_step0, split_step1], 1, True)
    state.bind(A_shared, its[1], "threadIdx.x")

    state.tensorize(A_local, state[A_local].iters[-2], "intrin_wmma_load_matrix_a")
    state.tensorize(B_local, state[B_local].iters[-2], "intrin_wmma_load_matrix_b")
    state.tensorize(C_local, state[C_local].iters[-3], "intrin_wmma_gemm")
    state.tensorize(C, state[C].iters[-2], "intrin_wmma_store_matrix")

    ret.append([state.state_object, -1])
    return ret
