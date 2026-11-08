import time
import numpy as np
import mlx.core as mx

def run_iterations(block_name):
    shape = (4096, 4096)
    a = mx.random.normal(shape=shape)
    b = mx.random.normal(shape=shape)
    mx.eval(a, b)  # Pre-generate inputs

    total_runs = 200
    times = []

    print(f"\n=== ITEM 2: {block_name} ({total_runs} iterations) ===")
    for idx in range(total_runs):
        t0 = time.perf_counter_ns()
        res = a @ b
        mx.eval(res)
        t1 = time.perf_counter_ns()
        
        t_ms = (t1 - t0) / 1_000_000.0
        times.append(t_ms)

    arr = np.array(times)
    print(f"[{block_name}] First Run: {arr[0]:.4f} ms | Median: {np.median(arr):.4f} ms | Min: {np.min(arr):.4f} ms")
    return times

if __name__ == "__main__":
    # Block 1: Captures initial setup + potential clock ramp
    times_block1 = run_iterations("Block_1_I
print("\Sleeping for 5 seconds to let GPU clocks step down...")
    time.sleep(5)
    
    # Block 2: If only the first run here is slow, it's setup. If early runs are slow again, it's clock ramp.
    times_block2 = run_iterations("Block_2_PostSleep")

    # Write raw iteration times to CSV
    csv_filename = "single_vs_median_raw.csv"
    with open(csv_filename, "w") as f:
        f.write("block,iteration,time_ms\n")
        for i, t in enumerate(times_block1):
            f.write(f"Block_1_Initial,{i},{t:.4f}\n")
        for i, t in enumerate(times_block2):
            f.write(f"Block_2_PostSleep,{i},{t:.4f}\n")
    print(f"\nRaw per-iteration list successfully written to {csv_filename}")
