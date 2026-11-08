import time
import numpy as np
import mlx.core as mx

N = 2**27
BYTES = 2 * N * 4

def make(K):
    def f(x):
        y = x
        for i in range(K):
            y = y * 1.0001 + 0.0001   # 2 ops per step
        return y
    return f

x = mx.random.normal((N,))
mx.eval(x)

def timed(fn, warmup=3, runs=15):
    for _ in range(warmup):
        mx.eval(fn(x))
    out = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        mx.eval(fn(x))
        out.append((time.perf_counter_ns() - t0) / 1e6)
    return out

with open("ksweep_raw.csv", "w") as fh:
    fh.write("K,variant,run_idx,time_ms,gbps\n")
    for K in [1, 2, 4, 8, 16, 32, 64, 128]:
        f = make(K)
        fc = mx.compile(f)
        t0 = time.perf_counter_ns()
        mx.eval(fc(x))
        first = (time.perf_counter_ns() - t0) / 1e6
        diff = mx.abs(f(x) - fc(x)).max().item()
        te, tc = timed(f), timed(fc)
        me, mc = np.median(te), np.median(tc)
        print(f"K={K:3d} | eager {me:8.3f} ms | compiled {mc:8.3f} ms | "
              f"speedup {me/mc:5.2f}x | first compiled {first:8.1f} ms | maxdiff {diff}")
        for name, t in [("eager", te), ("compiled", tc)]:
            for i, v in enumerate(t):
                fh.write(f"{K},{name},{i},{v:.4f},{BYTES/(v/1000)/1e9:.3f}\n")
