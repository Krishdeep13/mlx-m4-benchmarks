import time
import numpy as np
import mlx.core as mx

N = 2**27
BYTES = 2 * N * 4   # read x once + write y once, float32. Same for both variants.

def f(x):
    y = x * 2.0
    y = y + 1.0
    y = mx.sin(y)
    y = y * y
    y = y - 0.5
    return y

fc = mx.compile(f)

x = mx.random.normal((N,))
mx.eval(x)
print("dtype:", x.dtype, "| elements:", N, "| bytes counted:", BYTES)

def timed(fn, warmup=5, runs=25):
    for _ in range(warmup):
        mx.eval(fn(x))
    out = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        mx.eval(fn(x))
        out.append((time.perf_counter_ns() - t0) / 1e6)
    return out

# first compiled call (trace + compile + run), measured separately
t0 = time.perf_counter_ns()
mx.eval(fc(x))
first_call_ms = (time.perf_counter_ns() - t0) / 1e6
print(f"first compiled call (trace+compile+run): {first_call_ms:.2f} ms")

# correctness
print("max abs diff:", mx.abs(f(x) - fc(x)).max().item())

with open("fusion_raw.csv", "w") as fh:
    fh.write("variant,run_idx,time_ms,bytes_counted,gbps\n")
    for name, fn in [("eager", f), ("compiled", fc)]:
        t = timed(fn)
        arr = np.array(t)
        med = np.median(arr)
        gbps = BYTES / (med / 1000) / 1e9
        print(f"{name}: median {med:.3f} ms | min {arr.min():.3f} | "
              f"p10-p90 [{np.percentile(arr,10):.3f}, {np.percentile(arr,90):.3f}] | {gbps:.1f} GB/s")
        for i, x_ in enumerate(t):
            fh.write(f"{name},{i},{x_:.4f},{BYTES},{BYTES/(x_/1000)/1e9:.3f}\n")
