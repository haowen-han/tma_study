using double buffer and tma load to get rsqrt(0.5 + element * element) / 0.3, torch.compile is good enough for this kind of elementwise, so we compare our custom kernel with it.


```bash
python3 setup.py install
python3 test.py
```

# my own test result
device: H800
result:
```bash
All correctness tests passed!
Size: 8192x8192 | Custom: 0.236ms | PyTorch: 0.248ms | Speedup: 1.05x
Size: 8192x8192 | Custom: 0.235ms | PyTorch: 0.246ms | Speedup: 1.05x
Size: 8192x8192 | Custom: 0.236ms | PyTorch: 0.247ms | Speedup: 1.05x
Size: 8192x8192 | Custom: 0.236ms | PyTorch: 0.244ms | Speedup: 1.04x
Size: 8192x8192 | Custom: 0.236ms | PyTorch: 0.246ms | Speedup: 1.04x
Size: 8192x8192 | Custom: 0.236ms | PyTorch: 0.244ms | Speedup: 1.03x
Size: 8192x8192 | Custom: 0.236ms | PyTorch: 0.245ms | Speedup: 1.04x
Size: 8192x8192 | Custom: 0.236ms | PyTorch: 0.248ms | Speedup: 1.05x
Size: 8192x8192 | Custom: 0.236ms | PyTorch: 0.243ms | Speedup: 1.03x
Size: 8192x8192 | Custom: 0.237ms | PyTorch: 0.242ms | Speedup: 1.02x
Size: 8192x8192 | Custom: 0.235ms | PyTorch: 0.245ms | Speedup: 1.04x
Size: 8192x8192 | Custom: 0.236ms | PyTorch: 0.243ms | Speedup: 1.03x
Size: 8192x8192 | Custom: 0.236ms | PyTorch: 0.242ms | Speedup: 1.03x
Size: 8192x8192 | Custom: 0.235ms | PyTorch: 0.244ms | Speedup: 1.04x
Size: 8192x8192 | Custom: 0.237ms | PyTorch: 0.244ms | Speedup: 1.03x
Size: 8192x8192 | Custom: 0.235ms | PyTorch: 0.243ms | Speedup: 1.03x
Size: 8192x8192 | Custom: 0.235ms | PyTorch: 0.242ms | Speedup: 1.03x
Size: 8192x8192 | Custom: 0.236ms | PyTorch: 0.242ms | Speedup: 1.03x
Size: 8192x8192 | Custom: 0.236ms | PyTorch: 0.243ms | Speedup: 1.03x
Size: 8192x8192 | Custom: 0.236ms | PyTorch: 0.243ms | Speedup: 1.03x
Size: 16384x8192 | Custom: 0.461ms | PyTorch: 0.446ms | Speedup: 0.97x
Size: 16384x8192 | Custom: 0.461ms | PyTorch: 0.446ms | Speedup: 0.97x
Size: 16384x8192 | Custom: 0.459ms | PyTorch: 0.446ms | Speedup: 0.97x
Size: 16384x8192 | Custom: 0.462ms | PyTorch: 0.446ms | Speedup: 0.97x
Size: 16384x8192 | Custom: 0.463ms | PyTorch: 0.447ms | Speedup: 0.96x
Size: 16384x8192 | Custom: 0.461ms | PyTorch: 0.444ms | Speedup: 0.96x
Size: 16384x8192 | Custom: 0.460ms | PyTorch: 0.444ms | Speedup: 0.97x
Size: 16384x8192 | Custom: 0.461ms | PyTorch: 0.445ms | Speedup: 0.97x
Size: 16384x8192 | Custom: 0.461ms | PyTorch: 0.446ms | Speedup: 0.97x
Size: 16384x8192 | Custom: 0.461ms | PyTorch: 0.447ms | Speedup: 0.97x
Size: 16384x16384 | Custom: 0.909ms | PyTorch: 0.854ms | Speedup: 0.94x
Size: 16384x16384 | Custom: 0.909ms | PyTorch: 0.854ms | Speedup: 0.94x
Size: 16384x16384 | Custom: 0.909ms | PyTorch: 0.855ms | Speedup: 0.94x
Size: 16384x16384 | Custom: 0.908ms | PyTorch: 0.853ms | Speedup: 0.94x
Size: 16384x16384 | Custom: 0.909ms | PyTorch: 0.852ms | Speedup: 0.94x
Size: 16384x16384 | Custom: 0.910ms | PyTorch: 0.855ms | Speedup: 0.94x
Size: 16384x16384 | Custom: 0.909ms | PyTorch: 0.853ms | Speedup: 0.94x
Size: 16384x16384 | Custom: 0.911ms | PyTorch: 0.870ms | Speedup: 0.95x
Size: 16384x16384 | Custom: 0.910ms | PyTorch: 1.099ms | Speedup: 1.21x
Size: 16384x16384 | Custom: 0.911ms | PyTorch: 0.863ms | Speedup: 0.95x
Size: 24576x16384 | Custom: 1.358ms | PyTorch: 1.267ms | Speedup: 0.93x
Size: 24576x16384 | Custom: 1.357ms | PyTorch: 1.272ms | Speedup: 0.94x
Size: 24576x16384 | Custom: 1.357ms | PyTorch: 1.266ms | Speedup: 0.93x
Size: 24576x16384 | Custom: 1.356ms | PyTorch: 1.263ms | Speedup: 0.93x
Size: 24576x16384 | Custom: 1.358ms | PyTorch: 1.262ms | Speedup: 0.93x
Size: 24576x16384 | Custom: 1.356ms | PyTorch: 1.273ms | Speedup: 0.94x
Size: 24576x16384 | Custom: 1.358ms | PyTorch: 1.261ms | Speedup: 0.93x
Size: 24576x16384 | Custom: 1.356ms | PyTorch: 1.266ms | Speedup: 0.93x
Size: 24576x16384 | Custom: 1.357ms | PyTorch: 1.263ms | Speedup: 0.93x
Size: 24576x16384 | Custom: 1.357ms | PyTorch: 1.266ms | Speedup: 0.93x
Size: 24576x16384 | Custom: 1.358ms | PyTorch: 1.264ms | Speedup: 0.93x
Size: 24576x16384 | Custom: 1.357ms | PyTorch: 1.266ms | Speedup: 0.93x
Size: 24576x16384 | Custom: 1.356ms | PyTorch: 1.263ms | Speedup: 0.93x
Size: 24576x16384 | Custom: 1.357ms | PyTorch: 1.269ms | Speedup: 0.94x
Size: 24576x16384 | Custom: 1.356ms | PyTorch: 1.265ms | Speedup: 0.93x
Size: 24576x16384 | Custom: 1.356ms | PyTorch: 1.264ms | Speedup: 0.93x
Size: 24576x16384 | Custom: 1.356ms | PyTorch: 1.264ms | Speedup: 0.93x
Size: 24576x16384 | Custom: 1.356ms | PyTorch: 1.261ms | Speedup: 0.93x
Size: 24576x16384 | Custom: 1.357ms | PyTorch: 1.263ms | Speedup: 0.93x
Size: 24576x16384 | Custom: 1.365ms | PyTorch: 1.283ms | Speedup: 0.94x
Size: 32768x24576 | Custom: 2.702ms | PyTorch: 2.494ms | Speedup: 0.92x
Size: 32768x24576 | Custom: 2.698ms | PyTorch: 2.480ms | Speedup: 0.92x
Size: 32768x24576 | Custom: 2.696ms | PyTorch: 2.474ms | Speedup: 0.92x
Size: 32768x24576 | Custom: 2.697ms | PyTorch: 2.482ms | Speedup: 0.92x
Size: 32768x24576 | Custom: 2.703ms | PyTorch: 2.515ms | Speedup: 0.93x
Size: 32768x24576 | Custom: 2.699ms | PyTorch: 2.484ms | Speedup: 0.92x
Size: 32768x24576 | Custom: 2.698ms | PyTorch: 2.488ms | Speedup: 0.92x
Size: 32768x24576 | Custom: 2.697ms | PyTorch: 2.479ms | Speedup: 0.92x
Size: 32768x24576 | Custom: 2.697ms | PyTorch: 2.475ms | Speedup: 0.92x
Size: 32768x24576 | Custom: 2.697ms | PyTorch: 2.475ms | Speedup: 0.92x
Size: 32768x24576 | Custom: 2.697ms | PyTorch: 2.477ms | Speedup: 0.92x
Size: 32768x24576 | Custom: 2.696ms | PyTorch: 2.472ms | Speedup: 0.92x
Size: 32768x24576 | Custom: 2.697ms | PyTorch: 2.477ms | Speedup: 0.92x
Size: 32768x24576 | Custom: 2.696ms | PyTorch: 2.478ms | Speedup: 0.92x
Size: 32768x24576 | Custom: 2.697ms | PyTorch: 2.472ms | Speedup: 0.92x
Size: 32768x24576 | Custom: 2.695ms | PyTorch: 2.471ms | Speedup: 0.92x
Size: 32768x24576 | Custom: 2.697ms | PyTorch: 2.471ms | Speedup: 0.92x
Size: 32768x24576 | Custom: 2.695ms | PyTorch: 2.473ms | Speedup: 0.92x
Size: 32768x24576 | Custom: 2.696ms | PyTorch: 2.473ms | Speedup: 0.92x
Size: 32768x24576 | Custom: 2.698ms | PyTorch: 2.473ms | Speedup: 0.92x
Size: 40960x24576 | Custom: 3.369ms | PyTorch: 3.093ms | Speedup: 0.92x
Size: 40960x24576 | Custom: 3.368ms | PyTorch: 3.095ms | Speedup: 0.92x
Size: 40960x24576 | Custom: 3.368ms | PyTorch: 3.094ms | Speedup: 0.92x
Size: 40960x24576 | Custom: 3.368ms | PyTorch: 3.095ms | Speedup: 0.92x
Size: 40960x24576 | Custom: 3.367ms | PyTorch: 3.094ms | Speedup: 0.92x
Size: 40960x24576 | Custom: 3.366ms | PyTorch: 3.092ms | Speedup: 0.92x
Size: 40960x24576 | Custom: 3.369ms | PyTorch: 3.109ms | Speedup: 0.92x
Size: 40960x24576 | Custom: 3.367ms | PyTorch: 3.096ms | Speedup: 0.92x
Size: 40960x24576 | Custom: 3.384ms | PyTorch: 3.413ms | Speedup: 1.01x
Size: 40960x24576 | Custom: 3.512ms | PyTorch: 3.127ms | Speedup: 0.89x
Size: 40960x32768 | Custom: 4.587ms | PyTorch: 4.147ms | Speedup: 0.90x
Size: 40960x32768 | Custom: 4.480ms | PyTorch: 4.093ms | Speedup: 0.91x
Size: 40960x32768 | Custom: 4.482ms | PyTorch: 4.084ms | Speedup: 0.91x
Size: 40960x32768 | Custom: 4.486ms | PyTorch: 4.144ms | Speedup: 0.92x
Size: 40960x32768 | Custom: 4.484ms | PyTorch: 4.122ms | Speedup: 0.92x
Size: 40960x32768 | Custom: 4.559ms | PyTorch: 4.121ms | Speedup: 0.90x
Size: 40960x32768 | Custom: 4.482ms | PyTorch: 4.101ms | Speedup: 0.92x
Size: 40960x32768 | Custom: 4.484ms | PyTorch: 4.102ms | Speedup: 0.91x
Size: 40960x32768 | Custom: 4.485ms | PyTorch: 4.120ms | Speedup: 0.92x
```

We can observe that for the size of 8192x8192, our custom operator is faster than the one generated by torch.compile, while for larger shapes, our operator is slower than torch.compile. From the profiling results of ncu, the function compiled by torch.compile only uses global memory and does not utilize TMA or shared memory. Therefore, it is speculated that when reading data via TMA, the speed is slower than direct thread access (perhaps due to suboptimal handling of memory access coalescing in TMA). Additionally, since the computation part is too simple, the benefits of computation-memory overlap are insufficient to offset the negative impact of slower TMA memory access. In Study4, I will attempt to make the compute portion take more time to see if the hypothesis holds true.


