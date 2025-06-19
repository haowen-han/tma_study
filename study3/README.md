using double buffer and tma load to get (0.5 + element * element) / 0.3


```bash
python3 setup.py install
python3 test.py
```

The double_buffer_add function launches 132 blocks (the number of SMs in H100), with each block containing 256 threads. In each outermost for loop iteration, each block processes 4096 * 128 elements, while the inner for loop processes 128 * 128 elements out of these 4096 * 128 elements each time.

test machine: H800
here is the result:
```bash
[Benchmark Results]
1. double_buffer_add: 0.276 ms
2. origin_add: 0.839 ms
3. torch_compile_fn: 0.228 ms
torch_compile_fn / double_buffer_add: 0.83x
[Benchmark Results]
1. double_buffer_add: 0.275 ms
2. origin_add: 0.841 ms
3. torch_compile_fn: 0.219 ms
torch_compile_fn / double_buffer_add: 0.80x
[Benchmark Results]
1. double_buffer_add: 0.276 ms
2. origin_add: 0.837 ms
3. torch_compile_fn: 0.217 ms
torch_compile_fn / double_buffer_add: 0.79x
[Benchmark Results]
1. double_buffer_add: 0.276 ms
2. origin_add: 0.832 ms
3. torch_compile_fn: 0.216 ms
torch_compile_fn / double_buffer_add: 0.78x
[Benchmark Results]
1. double_buffer_add: 0.275 ms
2. origin_add: 0.836 ms
3. torch_compile_fn: 0.217 ms
torch_compile_fn / double_buffer_add: 0.79x
[Benchmark Results]
1. double_buffer_add: 0.275 ms
2. origin_add: 0.834 ms
3. torch_compile_fn: 0.214 ms
torch_compile_fn / double_buffer_add: 0.78x
[Benchmark Results]
1. double_buffer_add: 0.276 ms
2. origin_add: 0.835 ms
3. torch_compile_fn: 0.213 ms
torch_compile_fn / double_buffer_add: 0.77x
[Benchmark Results]
1. double_buffer_add: 0.275 ms
2. origin_add: 0.835 ms
3. torch_compile_fn: 0.212 ms
torch_compile_fn / double_buffer_add: 0.77x
[Benchmark Results]
1. double_buffer_add: 0.275 ms
2. origin_add: 0.836 ms
3. torch_compile_fn: 0.217 ms
torch_compile_fn / double_buffer_add: 0.79x
[Benchmark Results]
1. double_buffer_add: 0.274 ms
2. origin_add: 0.835 ms
3. torch_compile_fn: 0.213 ms
torch_compile_fn / double_buffer_add: 0.78x
[Benchmark Results]
1. double_buffer_add: 0.546 ms
2. origin_add: 0.967 ms
3. torch_compile_fn: 0.398 ms
torch_compile_fn / double_buffer_add: 0.73x
[Benchmark Results]
1. double_buffer_add: 0.546 ms
2. origin_add: 0.969 ms
3. torch_compile_fn: 0.398 ms
torch_compile_fn / double_buffer_add: 0.73x
[Benchmark Results]
1. double_buffer_add: 0.546 ms
2. origin_add: 0.965 ms
3. torch_compile_fn: 0.398 ms
torch_compile_fn / double_buffer_add: 0.73x
[Benchmark Results]
1. double_buffer_add: 0.546 ms
2. origin_add: 0.971 ms
3. torch_compile_fn: 0.399 ms
torch_compile_fn / double_buffer_add: 0.73x
[Benchmark Results]
1. double_buffer_add: 0.546 ms
2. origin_add: 0.969 ms
3. torch_compile_fn: 0.397 ms
torch_compile_fn / double_buffer_add: 0.73x
[Benchmark Results]
1. double_buffer_add: 0.546 ms
2. origin_add: 0.963 ms
3. torch_compile_fn: 0.397 ms
torch_compile_fn / double_buffer_add: 0.73x
[Benchmark Results]
1. double_buffer_add: 0.546 ms
2. origin_add: 0.970 ms
3. torch_compile_fn: 0.396 ms
torch_compile_fn / double_buffer_add: 0.73x
[Benchmark Results]
1. double_buffer_add: 0.546 ms
2. origin_add: 0.969 ms
3. torch_compile_fn: 0.397 ms
torch_compile_fn / double_buffer_add: 0.73x
[Benchmark Results]
1. double_buffer_add: 0.546 ms
2. origin_add: 0.972 ms
3. torch_compile_fn: 0.398 ms
torch_compile_fn / double_buffer_add: 0.73x
[Benchmark Results]
1. double_buffer_add: 0.546 ms
2. origin_add: 0.968 ms
3. torch_compile_fn: 0.401 ms
torch_compile_fn / double_buffer_add: 0.73x
[Benchmark Results]
1. double_buffer_add: 0.276 ms
2. origin_add: 0.838 ms
3. torch_compile_fn: 0.249 ms
torch_compile_fn / double_buffer_add: 0.90x
[Benchmark Results]
1. double_buffer_add: 0.277 ms
2. origin_add: 0.835 ms
3. torch_compile_fn: 0.247 ms
torch_compile_fn / double_buffer_add: 0.89x
[Benchmark Results]
1. double_buffer_add: 0.277 ms
2. origin_add: 0.839 ms
3. torch_compile_fn: 0.249 ms
torch_compile_fn / double_buffer_add: 0.90x
[Benchmark Results]
1. double_buffer_add: 0.277 ms
2. origin_add: 0.841 ms
3. torch_compile_fn: 0.251 ms
torch_compile_fn / double_buffer_add: 0.91x
[Benchmark Results]
1. double_buffer_add: 0.276 ms
2. origin_add: 0.838 ms
3. torch_compile_fn: 0.247 ms
torch_compile_fn / double_buffer_add: 0.89x
[Benchmark Results]
1. double_buffer_add: 0.276 ms
2. origin_add: 0.838 ms
3. torch_compile_fn: 0.246 ms
torch_compile_fn / double_buffer_add: 0.89x
[Benchmark Results]
1. double_buffer_add: 0.276 ms
2. origin_add: 0.836 ms
3. torch_compile_fn: 0.250 ms
torch_compile_fn / double_buffer_add: 0.90x
[Benchmark Results]
1. double_buffer_add: 0.277 ms
2. origin_add: 0.836 ms
3. torch_compile_fn: 0.246 ms
torch_compile_fn / double_buffer_add: 0.89x
[Benchmark Results]
1. double_buffer_add: 0.276 ms
2. origin_add: 0.839 ms
3. torch_compile_fn: 0.246 ms
torch_compile_fn / double_buffer_add: 0.89x
[Benchmark Results]
1. double_buffer_add: 0.276 ms
2. origin_add: 0.835 ms
3. torch_compile_fn: 0.248 ms
torch_compile_fn / double_buffer_add: 0.90x
```

It can be observed that the operator using TMA (Tensor Memory Accelerator) is significantly faster than the equivalent CUDA operator without TMA. However, since TMA store was not incorporated into the pipeline, its performance still falls short of the results generated by torch.compile (according to profiling data, torch.compile only accesses global memory, and due to its larger gridDim, it can achieve computation-memory access overlap by switching blocks/warps within the SM)