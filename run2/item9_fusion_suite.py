import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn

shape = (4096, 4096)
def elem(x):
    y = x * 2.0 + 1.0
    y = mx.sin(y)
    y = y * y
    return y - 0.5
def where_(x): return mx.where(x > 0, x * 2.0 + 1.0, x * 0.5 - 1.0)
def layernorm(x):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) * mx.rsqrt(v + 1e-5)
def softmax(x): return mx.softmax(x, axis=-1)
def mm_gelu(x, w): return nn.gelu(x @ w)

cases = [("elementwise", elem, 1), ("where", where_, 1), ("layernorm", layernorm, 1),
         ("softmax", softmax, 1), ("matmul+gelu", mm_gelu, 2)]

def med(fn, args, warm=5, runs=15):
    for _ in range(warm): mx.eval(fn(*args))
    out = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        mx.eval(fn(*args))
        out.append((time.perf_counter_ns() - t0) / 1e6)
    return np.array(out)

with open("fusion_suite_raw.csv", "w") as fh:
    fh.write("dtype,case,variant,run_idx,time_ms\n")
    for dt in [mx.float32, mx.float16]:
        dname = str(dt).split(".")[-1]
        x = mx.random.normal(shape).astype(dt)
        w = mx.random.normal(shape).astype(dt)
        mx.eval(x, w)
        for name, f, nargs in cases:
            args = (x,) if nargs == 1 else (x, w)
            fc = mx.compile(f)
            diff = mx.abs(f(*args).astype(mx.float32) - fc(*args).astype(mx.float32)).max().item()
            te, tc = med(f, args), med(fc, args)
            for v, t in [("eager", te), ("compiled", tc)]:
                for i, val in enumerate(t):
                    fh.write(f"{dname},{name},{v},{i},{val:.4f}\n")
            print(f"{dname:8s} {name:12s} eager {np.median(te):8.3f} | compiled {np.median(tc):8.3f} | "
                  f"speedup {np.median(te)/np.median(tc):5.2f}x | maxdiff {diff}")
