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
#define BLOCK_NUM 132 * 8

template <typename _TiledCopyS, typename _TiledCopyD, typename _GmemLayout,
          typename _SmemLayout, typename _SingleSmemLayout>
struct Params {
  using TiledCopyS = _TiledCopyS;
  using TiledCopyD = _TiledCopyD;
  using GmemLayout = _GmemLayout;
  using SmemLayout = _SmemLayout;
  using SingleSmemLayout = _SingleSmemLayout;

  TiledCopyS const tmaLoad;
  TiledCopyD const tmaStore;
  GmemLayout const gmemLayout;
  SmemLayout const smemLayout;
  SingleSmemLayout const singlesmemLayout;

  Params(_TiledCopyS const &tmaLoad, _TiledCopyD const &tmaStore,
         _GmemLayout const &gmemLayout, _SmemLayout const &smemLayout, _SingleSmemLayout const &singlesmemLayout)
      : tmaLoad(tmaLoad),tmaStore(tmaStore), gmemLayout(gmemLayout),
        smemLayout(smemLayout), singlesmemLayout(singlesmemLayout) {}
};

template <class Params>
__global__ void __launch_bounds__(256, 1) double_buffer_add_kernel(
    CUTE_GRID_CONSTANT Params const params,
    CUTE_GRID_CONSTANT float*const input_ptr,
    CUTE_GRID_CONSTANT const int input_stride_0,
    CUTE_GRID_CONSTANT const int data_block_num_x, 
    CUTE_GRID_CONSTANT const int data_block_num_y,
    CUTE_GRID_CONSTANT const int data_block_num

){
    using namespace cute;
    using GmemLayout = typename Params::GmemLayout;
    using SmemLayout = typename Params::SmemLayout;
    using SingleSmemLayout = typename Params::SingleSmemLayout;
    auto &tmaLoad = params.tmaLoad;
    auto &tmaStore = params.tmaStore;
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
        prefetch_tma_descriptor(tmaStore.get_tma_descriptor());
    }

    for(int linear_block_idx = blockIdx.x; linear_block_idx < data_block_num; linear_block_idx = linear_block_idx + BLOCK_NUM){
        int block_idx_x = linear_block_idx  / data_block_num_y;
        int block_idx_y = linear_block_idx  % data_block_num_y;
        
        //=================================================================
        // now we load the first block data for double buffer to shm0
        //=================================================================
        Tensor mS = tmaLoad.get_tma_tensor(shape(gmemLayout));  
        auto blkCoord = make_coord(block_idx_x * 16, block_idx_y);
        Tensor gS = local_tile(mS, make_shape(Int<128>{}, Int<128>{}), blkCoord);
        auto cta_tmaS = tmaLoad.get_slice(Int<0>{});

        Tensor mD = tmaStore.get_tma_tensor(shape(gmemLayout));
        Tensor gD = local_tile(mD, make_shape(Int<128>{}, Int<128>{}), blkCoord);
        auto cta_tmaD = tmaStore.get_slice(Int<0>{});

        if (warp_idx == 0 and lane_predicate) {
            mbarrier.init(1 /* arrive count */);
            mbarrier.arrive_and_expect_tx(kTmaTransactionBytes);
            copy(tmaLoad.with(reinterpret_cast<BarrierType &>(mbarrier)),
                cta_tmaS.partition_S(gS), cta_tmaS.partition_D(sS));
        }

        __syncthreads();
        mbarrier.wait(0 /* phase */);
        cutlass::arch::fence_view_async_shared();

        int phase = 1;
        //==============double buffer start===========
#pragma unroll
        for(int i = 0; i < 8; i++){


            //===========launch the data loading for shm1==========
            blkCoord = make_coord(block_idx_x * 16 + 2 * i + 1, block_idx_y);
            gS = local_tile(mS, make_shape(Int<128>{}, Int<128>{}), blkCoord);
            if (warp_idx == 0 and lane_predicate) {
                mbarrier.arrive_and_expect_tx(kTmaTransactionBytes);
                copy(tmaLoad.with(reinterpret_cast<BarrierType &>(mbarrier)),
                    cta_tmaS.partition_S(gS), cta_tmaS.partition_D(sS_2));
            }
            
            // ==========================compute shm0==============================
            auto sS_data_ptr = sS.data();
            float* sS_float_ptr = reinterpret_cast<float*>(sS_data_ptr.get());
            for(int row = threadIdx.x / 32; row < 128; row = row + 8){
                int col = (threadIdx.x % 32) * 4;
                float4 temp = reinterpret_cast<float4*>(sS_float_ptr + (row * 128) + col)[0];
                temp.x = rsqrt((temp.x * temp.x + 0.5) / 0.3); 
                temp.y = rsqrt((temp.y * temp.y + 0.5) / 0.3);
                temp.z = rsqrt((temp.z * temp.z + 0.5) / 0.3);
                temp.w = rsqrt((temp.w * temp.w + 0.5) / 0.3);
                reinterpret_cast<float4*>(sS_float_ptr + (row * 128) + col)[0] = temp;
            }

            //==========================launch the data storing for shm0==============================
            if (warp_idx == 0 and lane_predicate) {
                blkCoord = make_coord(block_idx_x * 16 + 2 * i, block_idx_y);
                gD = local_tile(mD,  make_shape(Int<128>{}, Int<128>{}), blkCoord);
                cute::copy(tmaStore, cta_tmaD.partition_S(sS), cta_tmaD.partition_D(gD));
                cute::tma_store_arrive();
            }

            // ==========================waiting for the data loading to shm1 to complete==============================
            mbarrier.wait(phase /* phase */);   
            phase = (phase + 1) % 2;
            cutlass::arch::fence_view_async_shared();

            // ==========================compute shm1==============================
            auto sS_2_data_ptr = sS_2.data();
            float* sS_2_float_ptr = reinterpret_cast<float*>(sS_2_data_ptr.get());
            for(int row = threadIdx.x / 32; row < 128; row = row + 8){
                int col = (threadIdx.x % 32) * 4;
                float4 temp = reinterpret_cast<float4*>(sS_2_float_ptr + (row * 128) + col)[0];
                temp.x = rsqrt((temp.x * temp.x + 0.5) / 0.3); 
                temp.y = rsqrt((temp.y * temp.y + 0.5) / 0.3);
                temp.z = rsqrt((temp.z * temp.z + 0.5) / 0.3);
                temp.w = rsqrt((temp.w * temp.w + 0.5) / 0.3);
                reinterpret_cast<float4*>(sS_2_float_ptr + (row * 128) + col)[0] = temp;
            }

            // ==========================waiting for the data storing from shm0==============================
            cute::tma_store_wait<0>();

            // =========================launch the data loading for shm0===============================
            if(i < 7){
                blkCoord = make_coord(block_idx_x * 16 + 2 * i + 2, block_idx_y);
                gS = local_tile(mS, make_shape(Int<128>{}, Int<128>{}), blkCoord);
                if (warp_idx == 0 and lane_predicate) {
                    mbarrier.arrive_and_expect_tx(kTmaTransactionBytes);
                    copy(tmaLoad.with(reinterpret_cast<BarrierType &>(mbarrier)),
                        cta_tmaS.partition_S(gS), cta_tmaS.partition_D(sS));
                }
            }


            // =========================launch the data storing for shm1===============================
            if (warp_idx == 0 and lane_predicate) {
                blkCoord = make_coord(block_idx_x * 16 + 2 * i + 1, block_idx_y);
                gD = local_tile(mD,  make_shape(Int<128>{}, Int<128>{}), blkCoord);
                cute::copy(tmaStore, cta_tmaD.partition_S(sS_2), cta_tmaD.partition_D(gD));
                cute::tma_store_arrive();
            }            
            

            // ==========================waiting for the data loading to shm0==============================
            if(i < 7){
                mbarrier.wait(phase /* phase */);       
                phase = (phase + 1) % 2;
                cutlass::arch::fence_view_async_shared();
            }

            // ==========================waiting for the data storing from shm1==============================
            cute::tma_store_wait<0>();

         }
    }
}


// every CTA will process [2048, 128] data, and in each for loop, the CTA will process [128, 128] data
void double_buffer_add(torch::Tensor &input){
    using namespace cute;
    TORCH_CHECK(input.dim() == 2, "Tensor must be 2-dimensional");
    TORCH_CHECK(input.is_contiguous(), "Tensor must be contiguous");

    // 检查 shape[0] % 2048 == 0 和 shape[1] % 128 == 0
    TORCH_CHECK(
        input.size(0) % 2048 == 0 && input.size(1) % 128 == 0,
        "Tensor shape must be divisible by [2048, 128], but got [", 
        input.size(0), ", ", input.size(1), "]"
    );

    TORCH_CHECK(
        ((input.size(0) / 2048) * (input.size(1) / 128)) % 132 == 0,
        "for persistent block, so the number of block must be the multiple of 132",
        input.size(0), ", ", input.size(1), "]"
    );

    // now we initialize params for kernel
    auto stream = at::cuda::getCurrentCUDAStream(input.device().index());
    
    // get tma desc
    auto gmemLayoutS = make_layout(make_shape(input.size(0), input.size(1)), make_stride(input.size(1), Int<1>{}));
    Tensor tensor_S = make_tensor(
        make_gmem_ptr((float*)input.data_ptr()), gmemLayoutS);
    Tensor tensor_D = tensor_S;  // we change the input tensor inplace
    auto single_smemLayout = make_layout(make_shape(Int<128>{}, Int<128>{}), LayoutRight{});
    auto smemLayout = make_layout(make_shape(Int<256>{}, Int<128>{}), LayoutRight{});
    
    auto tma_load =
      make_tma_copy(SM90_TMA_LOAD{}, tensor_S, single_smemLayout);
    auto tma_store =
      make_tma_copy(SM90_TMA_STORE{}, tensor_D, single_smemLayout);

    Params params(tma_load, tma_store, gmemLayoutS, smemLayout, single_smemLayout);
    dim3 block_dim(256);
    dim3 grid_dim(BLOCK_NUM); //sm counts
    dim3 cluster_dims(1);

    int smem_size = int(sizeof(SharedStorageTMA<float, decltype(smemLayout)>));
    // printf("smem size: %d.\n", smem_size);

    void const *kernel = (void const *)double_buffer_add_kernel<decltype(params)>;
    cfx::set_smem_size(smem_size, kernel);
    
    cutlass::ClusterLaunchParams launch_params{grid_dim, block_dim, cluster_dims,
                                             smem_size};
    
    int data_block_num_x = input.size(0) / 2048;
    int data_block_num_y = input.size(1) / 128;
    int data_block_num = data_block_num_x * data_block_num_y;
    cutlass::Status status = cutlass::launch_kernel_on_cluster(
        launch_params, kernel, params, (float*)input.data_ptr(), input.size(1), data_block_num_x, data_block_num_y, data_block_num);
}



__global__ void __launch_bounds__(256, 1) origin_add_kernel(
    float* input_ptr,
    int input_stride_0
){
    int block_offset_x = blockIdx.x * 2048;
    int block_offset_y = blockIdx.y * 128;
#pragma unroll
    for(int i = 0; i< 256;i++){
        int offset_x = block_offset_x + 8 * i + threadIdx.x / 32;
        int offset_y = block_offset_y + (threadIdx.x % 32) * 4;
        float4* data_ptr = reinterpret_cast<float4*>(input_ptr + offset_x * input_stride_0 + offset_y);
        float4 temp = data_ptr[0];
        temp.x = rsqrt((temp.x * temp.x + 0.5) / 0.3); 
        temp.y = rsqrt((temp.y * temp.y + 0.5) / 0.3);
        temp.z = rsqrt((temp.z * temp.z + 0.5) / 0.3);
        temp.w = rsqrt((temp.w * temp.w + 0.5) / 0.3);        
        data_ptr[0] = temp;
    }
}



// every CTA will process [2048, 128] data, and in each for loop, the CTA will process [128, 128] data
void origin_add(torch::Tensor &input){
    TORCH_CHECK(input.dim() == 2, "Tensor must be 2-dimensional");
    TORCH_CHECK(input.is_contiguous(), "Tensor must be contiguous");

    // 检查 shape[0] % 2048 == 0 和 shape[1] % 128 == 0
    TORCH_CHECK(
        input.size(0) % 2048 == 0 && input.size(1) % 128 == 0,
        "Tensor shape must be divisible by [2048, 128], but got [", 
        input.size(0), ", ", input.size(1), "]"
    );

    auto stream = at::cuda::getCurrentCUDAStream(input.device().index());
    dim3 block_dim(256);
    dim3 grid_dim(input.size(0)/2048, input.size(1)/128);
    
    origin_add_kernel<<<grid_dim, block_dim, 0, stream>>>((float*)input.data_ptr(), input.size(1));
}