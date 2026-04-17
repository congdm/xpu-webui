import torch
import types
import time

def _device_type(device: str | torch.device | None) -> str:
    if device is None:
        return "cpu"
    if isinstance(device, str):
        return device.split(":")[0]
    return device.type

def torch_gpu(device: str | torch.device | None):
    device_type = _device_type(device)
    if device_type == "cuda":
        return torch.cuda
    elif device_type == "xpu":
        return torch.xpu
    else:
        return None

Profiler = types.SimpleNamespace()

def init_profiler(device, num_preallocated_events: int = 10) -> None:
    _torch_gpu = torch_gpu(device)
    if _torch_gpu is not None:
        # Pre-allocate events to avoid overhead during profiling
        Profiler.preallocated_events = []
        for i in range(num_preallocated_events):
            Profiler.preallocated_events.append(_torch_gpu.Event(enable_timing=True))
        
    Profiler.start_events = {}
    Profiler.end_events = {}
    Profiler.start_times = {}
    Profiler.end_times = {}

def create_event(device) -> torch.cuda.Event | torch.xpu.Event | None:
    _torch_gpu = torch_gpu(device)
    if _torch_gpu is not None and hasattr(Profiler, "preallocated_events") and Profiler.preallocated_events:
        if len(Profiler.preallocated_events) > 0:
            event = Profiler.preallocated_events.pop()
        else:
            event = _torch_gpu.Event(enable_timing=True)
            print("Warning: No preallocated events available, creating a new event which may introduce overhead.")
    elif _torch_gpu is not None:
        event = _torch_gpu.Event(enable_timing=True)
        print("Warning: Preallocated events not initialized, creating a new event which may introduce overhead.")
    return event

def record_start(device, metric: str, stream=None) -> None:
    _torch_gpu = torch_gpu(device)
    if _torch_gpu is not None:
        event = create_event(device)
        if stream is not None:
            event.record(stream)
        else:
            event.record()
        Profiler.start_events[metric] = event
    else: # fallback to high-resolution timer for CPU
        Profiler.start_times[metric] = time.perf_counter()

def record_end(device, metric: str, stream=None) -> None:
    _torch_gpu = torch_gpu(device)
    if _torch_gpu is not None:
        event = create_event(device)
        if stream is not None:
            event.record(stream)
        else:
            event.record()
        Profiler.end_events[metric] = event
    else: # fallback to high-resolution timer for CPU
        Profiler.end_times[metric] = time.perf_counter()

def summary(device) -> dict[str, float]:
    _torch_gpu = torch_gpu(device)
    # if _torch_gpu is not None:        # Ensure all events are recorded
    #     for event in list(Profiler.start_events.values()) + list(Profiler.end_events.values()):
    #         event.synchronize()

    summary = {}
    for metric in Profiler.start_events:
        if metric in Profiler.end_events:
            start_event = Profiler.start_events[metric]
            end_event = Profiler.end_events[metric]
            end_event.synchronize()
            summary[metric] = start_event.elapsed_time(end_event)
    for metric in Profiler.start_times:
        if metric in Profiler.end_times:
            start_time = Profiler.start_times[metric]
            end_time = Profiler.end_times[metric]
            summary[metric] = (end_time - start_time) * 1000  # Convert to milliseconds
    return summary
