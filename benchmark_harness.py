import time
import datetime
import numpy as np
import mlx.core as mx

def run_benchmark(experiment_name, variant_name, dtype, shape, func, bytes_fn=None, flops_fn=None, warmup=5, timed=25):
    # Format shape cleanly as "4096x4096" to avoid comma-separated CSV column shifts
    shape_str = "x".join(map(str, shape)) if isinstance(shape, (list, tuple)) else str(shape)
    dtype_str = str(dtype).split(".")[-1]
    
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = "benchmark_results.csv"

    print(f"--- Running: {experiment_name} | {variant_name} | {dtype_str} | {shape_str} ---")

    # Warmup runs
    for _ in range(warmup):
        res = func()
        mx.eval(res)

    # Timed runs
    times_ms = []
    for run_idx in range(timed):
        start_time = time.perf_counter_ns()
        
        res = func()
        mx.eval(res)  # Force execution before stopping clock
        
        end_time = time.perf_counter_ns()
        elapsed_ms = (end_time - start_time) / 1_000_000.0
        times_ms.append(elapsed_ms)

    arr = np.array(times_ms)
    median_ms = float(np.median(arr))
    min_ms = float(np.min(arr))
    p10 = float(np.percentile(arr, 10))
    p90 = float(np.percentile(arr, 90))

    metric_name = "gbps"
    metric_val = 0.0
    if flops_fn is not None:
        metric_name = "tflops"
        total_flops = flops_fn(shape)
        metric_val = (total_flops / (median_ms / 1000.0)) / 1e12
    elif bytes_fn is not None:
        metric_name = "gbps"
        bytes_counted = bytes_fn(shape, dtype)
        metric_val = (bytes_counted / (median_ms / 1000.0)) / 1e9

    print(f"Stats -> Median: {median_ms:.3f} ms | Min: {min_ms:.3f} ms | P10-P90: [{p10:.3f}, {p90:.3f}] ms | {metric_name.upper()}: {metric_val:.2f}")

    file_exists = False
    try:
        with open(csv_filename, "r"):
            file_exists = True
    except FileNotFoundError:
        pass

    with open(csv_filename, "a") as f:
        if not file_exists:
            f.write("run_id,experiment,variant,dtype,shape,run_idx,time_ms,metric_name,metric_val\n")
        for run_idx, t in enumerate(times_ms):
            if flops_fn is not None:
                row_metric = (flops_fn(shape) / (t / 1000.0)) / 1e12
            elif bytes_fn is not None:
                row_metric = (bytes_fn(shape, dtype) / (t / 1000.0)) / 1e9
            else:
                row_metric = 0.0
            
            f.write(f"{run_id},{experiment_name},{variant_name},{dtype_str},{shape_str},{run_idx},{t:.4f},{metric_name},{row_metric:.4f}\n")

    return median_ms, min_ms, p10, p90
