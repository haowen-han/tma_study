#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <chrono>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cutlass/numeric_types.h"
#include <cute/arch/cluster_sm90.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"

#include "cutlass/detail/layout.hpp"
#include <cute/util/print.hpp>

#include "smem_helper.hpp"
#include "shared_storage.h"
#include "smem_helper.hpp"


#include <cstdint>
#include <torch/extension.h>
#include <iostream>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>  // Correct include for CUDA Stream
#include <c10/cuda/CUDAGuard.h>
#include "c10/util/Exception.h"

template <typename _TiledCopyS, typename _GmemLayout,
          typename _SmemLayout, typename _SingleSmemLayout>
struct Params {
  using TiledCopyS = _TiledCopyS;
  using GmemLayout = _GmemLayout;
  using SmemLayout = _SmemLayout;
  using SingleSmemLayout = _SingleSmemLayout;

  TiledCopyS const tmaLoad;
  GmemLayout const gmemLayout;
  SmemLayout const smemLayout;
  SingleSmemLayout const singlesmemLayout;

  Params(_TiledCopyS const &tmaLoad,
         _GmemLayout const &gmemLayout, _SmemLayout const &smemLayout, _SingleSmemLayout const &singlesmemLayout)
      : tmaLoad(tmaLoad), gmemLayout(gmemLayout),
        smemLayout(smemLayout), singlesmemLayout(singlesmemLayout) {}
};

template <class Params>
__global__ void double_buffer_add_kernel(
    CUTE_GRID_CONSTANT Params const params,
    float* input_ptr,
    int input_stride_0
){
    using namespace cute;
    using GmemLayout = typename Params::GmemLayout;
    using SmemLayout = typename Params::SmemLayout;
    using SingleSmemLayout = typename Params::SingleSmemLayout;
    auto &tmaLoad = params.tmaLoad;
    auto &gmemLayout = params.gmemLayout;
    auto &smemLayout = params.smemLayout;
    auto &singlesmemLayout = params.singlesmemLayout;

    // 获取当前block和线程的index
    int thread_id = threadIdx.x;

    extern __shared__ char shared_memory[];

    using SharedStorage = SharedStorageTMA<float, SmemLayout>;
    SharedStorage &shared_storage =
        *reinterpret_cast<SharedStorage *>(shared_memory);

    // auto single_smemLayout = make_layout(make_shape(Int<128>{}, Int<128>{}), LayoutRight{});

    // Define smem tensor
    Tensor sS =
        make_tensor(make_smem_ptr(shared_storage.smem.data()), singlesmemLayout); // [0:128, 0:128]
    Tensor sS_2 =
        make_tensor(make_smem_ptr(shared_storage.smem.data() + 128 * 128), singlesmemLayout);
    
    // Print the addresses
    // printf("shared_memory address: %p\n", shared_memory);
    // printf("shared_storage.smem.data() address: %p\n", shared_storage.smem.data());   
    // printf("shared_storage.smem.data() + 128 * 128 address: %p\n", shared_storage.smem.data() + 128 * 128);   

    // Get mbarrier object and its value type
    auto &mbarrier = shared_storage.mbarrier;
    using BarrierType = cutlass::arch::ClusterTransactionBarrier::ValueType;
    static_assert(cute::is_same_v<BarrierType, uint64_t>,
                "Value type of mbarrier is uint64_t.");

    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const bool lane_predicate = cute::elect_one_sync();
    constexpr int kTmaTransactionBytes = sizeof(ArrayEngine<float, size(SingleSmemLayout{})>);
    // printf("kTmaTransactionBytes:%d \n", kTmaTransactionBytes);
    // Prefetch TMA descriptors for load and store
    if (warp_idx == 0 && lane_predicate) {
        prefetch_tma_descriptor(tmaLoad.get_tma_descriptor());
    }

    //=================================================================
    // now we load the first block data for double buffer
    //=================================================================
    Tensor mS = tmaLoad.get_tma_tensor(shape(gmemLayout));
    auto blkCoord = make_coord(blockIdx.x * 32, blockIdx.y);
    Tensor gS = local_tile(mS, make_shape(Int<128>{}, Int<128>{}), blkCoord);

    auto cta_tmaS = tmaLoad.get_slice(Int<0>{});
    if (warp_idx == 0 and lane_predicate) {
        mbarrier.init(1 /* arrive count */);
        mbarrier.arrive_and_expect_tx(kTmaTransactionBytes);
        copy(tmaLoad.with(reinterpret_cast<BarrierType &>(mbarrier)),
            cta_tmaS.partition_S(gS), cta_tmaS.partition_D(sS));
    }
    __syncthreads();
    mbarrier.wait(0 /* phase */);
    cutlass::arch::fence_view_async_shared();

    //==============double buffer start===========
    for(int i = 0; i < 32; i++){
        //===========pre-load the data that will be used in the next loop iteration to sS_2==========
        if(i <= 31){
            blkCoord = make_coord(blockIdx.x * 32 + i + 1, blockIdx.y);
            gS = local_tile(mS, make_shape(Int<128>{}, Int<128>{}), blkCoord);
            if (warp_idx == 0 and lane_predicate) {
                mbarrier.arrive_and_expect_tx(kTmaTransactionBytes);
                copy(tmaLoad.with(reinterpret_cast<BarrierType &>(mbarrier)),
                    cta_tmaS.partition_S(gS), cta_tmaS.partition_D(sS_2));
            }
        }

        // ==========================compute sS==============================
        auto sS_data_ptr = sS.data();
        float* sS_float_ptr = reinterpret_cast<float*>(sS_data_ptr.get());
        for(int row = threadIdx.x / 32; row < 128; row = row + 8){
            int col = (threadIdx.x % 32) * 4;
            float4 temp = reinterpret_cast<float4*>(sS_float_ptr + (row * 128) + col)[0];
            temp.x = rsqrtf(temp.x * temp.x + 0.5) / 0.3; 
            temp.y = rsqrtf(temp.y * temp.y + 0.5) / 0.3;
            temp.z = rsqrtf(temp.z * temp.z + 0.5) / 0.3;
            temp.w = rsqrtf(temp.w * temp.w + 0.5) / 0.3;
            reinterpret_cast<float4*>(input_ptr + (blockIdx.x * 4096 + 128 * i + row) * input_stride_0 + (blockIdx.y * 128 + col))[0] = temp;
        }
        

        Tensor temp = sS;
        sS = sS_2;
        sS_2 = temp;
        if(i <= 31){
            __syncthreads();
            mbarrier.wait((i + 1)%2 /* phase */);
            cutlass::arch::fence_view_async_shared();
        }
    }

}


// every CTA will process [4096, 128] data, and in each for loop, the CTA will process [128, 128] data
void double_buffer_add(torch::Tensor &input){
    using namespace cute;
    TORCH_CHECK(input.dim() == 2, "Tensor must be 2-dimensional");
    TORCH_CHECK(input.is_contiguous(), "Tensor must be contiguous");

    // 检查 shape[0] % 4096 == 0 和 shape[1] % 128 == 0
    TORCH_CHECK(
        input.size(0) % 4096 == 0 && input.size(1) % 128 == 0,
        "Tensor shape must be divisible by [4096, 128], but got [", 
        input.size(0), ", ", input.size(1), "]"
    );

    // now we initialize params for kernel
    auto stream = at::cuda::getCurrentCUDAStream(input.device().index());
    
    // get tma desc
    auto gmemLayoutS = make_layout(make_shape(input.size(0), input.size(1)), make_stride(input.size(1), Int<1>{}));
    Tensor tensor_S = make_tensor(
        make_gmem_ptr((float*)input.data_ptr()), gmemLayoutS);
    auto single_smemLayout = make_layout(make_shape(Int<128>{}, Int<128>{}), LayoutRight{});
    auto smemLayout = make_layout(make_shape(Int<256>{}, Int<128>{}), LayoutRight{});
    
    auto tma_load =
      make_tma_copy(SM90_TMA_LOAD{}, tensor_S, single_smemLayout);

    Params params(tma_load, gmemLayoutS, smemLayout, single_smemLayout);
    dim3 block_dim(256);
    dim3 grid_dim(input.size(0)/4096, input.size(1)/128);
    dim3 cluster_dims(1);

    int smem_size = int(sizeof(SharedStorageTMA<float, decltype(smemLayout)>));
    // printf("smem size: %d.\n", smem_size);

    void const *kernel = (void const *)double_buffer_add_kernel<decltype(params)>;
    cfx::set_smem_size(smem_size, kernel);
    
    cutlass::ClusterLaunchParams launch_params{grid_dim, block_dim, cluster_dims,
                                             smem_size};
    cutlass::Status status = cutlass::launch_kernel_on_cluster(launch_params, kernel, params, (float*)input.data_ptr(), input.size(1));




}