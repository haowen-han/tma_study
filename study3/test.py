import torch
import my_torch_ext
import math
torch.manual_seed(1)
import random
random.seed(1)
import numpy as np
np.random.seed(1)
import nvtx
from torch.cuda import Event
start_event = Event(enable_timing=True)
end_event = Event(enable_timing=True)

def measure_operator(operator, input_tensor, num_warmup=2, num_repeats=5):
    """测量算子耗时，预热后取后三次的平均值"""
    times = []
    
    # 预热（不记录时间）
    for _ in range(num_warmup):
        _ = operator(input_tensor)
    
    # 正式测量
    for _ in range(num_repeats):
        start_event.record()
        _ = operator(input_tensor)
        end_event.record()
        end_event.synchronize()  # 等待事件完成
        elapsed_time = start_event.elapsed_time(end_event)  # 毫秒
        times.append(elapsed_time)
    
    # 取后三次的平均值
    avg_time = sum(times[-3:]) / 3
    return avg_time


@torch.compile
def torch_compile_fn(input):
    return torch.rsqrt(input * input + 0.5) / 0.3

def correctness_verification(height: int, width: int):
    temp = torch.randn(height, width, dtype=torch.float32, device="cuda")
    process_input = temp.clone()
    process_input_2 = temp.clone()
    process_input_3 = temp.clone()
    for _ in range(5):
        __ = torch_compile_fn(temp.clone())
    with nvtx.annotate("test", color="blue"):
        my_torch_ext.ops.double_buffer_add(process_input)
        my_torch_ext.ops.origin_add(process_input_2)
        process_input_3 = torch_compile_fn(process_input_3)
    assert torch.allclose(process_input_2, process_input), f"{height} * {width} is not correct!"
    assert torch.allclose(process_input_2, process_input_3), f"{height} * {width} is not correct!"

    time_double_buffer = measure_operator(my_torch_ext.ops.double_buffer_add, process_input)
    time_origin = measure_operator(my_torch_ext.ops.origin_add, process_input_2)
    time_compile = measure_operator(torch_compile_fn, process_input_3)

    print(f"[Benchmark Results]")
    print(f"1. double_buffer_add: {time_double_buffer:.3f} ms")
    print(f"2. origin_add: {time_origin:.3f} ms")
    print(f"3. torch_compile_fn: {time_compile:.3f} ms")
    print(f"torch_compile_fn / double_buffer_add: {time_compile/time_double_buffer:.2f}x")

if __name__ == "__main__":
    for height, width in ((4096 * 4, 128 * 33), (4096 * 8, 128 * 33), (4096 * 2, 128 * 33 * 2)):
        for _ in range(10):
            correctness_verification(height, width)


    

