import time
import numpy as np
import mlx.core as mx

def stats(arr, unit):
    a = np.array(arr)
    return (f"median {np.median(a):.3f} {unit} | min {np.min(a):.3f} | "
            f"p10-p90 [{np.percentile(a,10):.3f}, {np.percentile(a,90):.3f}]")

def timed(fn, warmup=5, runs=25):
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        fn()
        out.append((time.perf_counter_ns() - t0) / 1e6)  # ms
    return out

a = mx.random.normal((4096, 4096))
b = mx.random.normal((4096, 4096))
mx.eval(a, b)
print("dtype:", a.dtype)

def A():            # no eval: builds graph only
    r1 = a @ b
    r2 = r1 + a

def B():            # one eval, same two ops
    r1 = a @ b
    r2 = r1 + a
    mx.eval(r2)

def C():            # eval after each op, same two ops
    r1 = a @ b
    mx.eval(r1)
    r2 = r1 + a
    mx.eval(r2)

ta = [t * 1000 for t in timed(A)]   # convert ms to µs
tb = timed(B)
tc = timed(C)
print("A (no eval, graph build only):", stats(ta, "µs"))
print("B (one eval):                 ", stats(tb, "ms"))
print("C (eval per op):              ", stats(tc, "ms"))
print(f"sync cost C-B (median): {np.median(tc) - np.median(tb):.3f} ms")
