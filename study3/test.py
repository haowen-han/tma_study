import torch
import my_torch_ext
import math
torch.manual_seed(1)
import random
random.seed(1)
import numpy as np
np.random.seed(1)

def correctness_verification(height: int, width: int):
    temp = torch.randn(height, width, dtype=torch.float32, device="cuda")
    process_input = temp.clone()
    my_torch_ext.ops.double_buffer_add(process_input)
    temp = mul_add_half(temp)
    assert torch.allclose(temp, process_input), f"{height} * {width} is not correct!"

@torch.compile
def mul_add_half(tensor):
    return torch.rsqrt(0.5 + tensor * tensor) / 0.3

def measure_performance(height: int, width: int):
    # Create input tensor
    temp = torch.randn(height, width, dtype=torch.float32, device="cuda")
    
    # Warm-up
    for _ in range(3):
        my_torch_ext.ops.double_buffer_add(temp.clone())
        out = mul_add_half(temp.clone())
    
    # Create CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Measure custom kernel
    torch.cuda.synchronize()
    start_event.record()
    my_torch_ext.ops.double_buffer_add(temp)
    end_event.record()
    torch.cuda.synchronize()
    custom_time = start_event.elapsed_time(end_event)
    
    # Measure PyTorch's add_
    torch.cuda.synchronize()
    start_event.record()
    temp = mul_add_half(temp)
    end_event.record()
    torch.cuda.synchronize()
    pytorch_time = start_event.elapsed_time(end_event)
    
    print(f"Size: {height}x{width} | Custom: {custom_time:.3f}ms | PyTorch: {pytorch_time:.3f}ms | "
          f"Speedup: {pytorch_time/custom_time:.2f}x")


if __name__ == "__main__":
    for i in range(1, 100):
        width = 128 * 32 * math.ceil(i/30)
        height = 4096 * math.ceil(i/20)
        correctness_verification(height, width)
    print("All correctness tests passed!")

    for i in range(1, 100):
        width = 128 * 32 * math.ceil(i/30) * 2
        height = 4096 * math.ceil(i/20)* 2
        measure_performance(height, width)


    

