import time
import numpy as np
import mlx.core as mx

N = 2**27
x = mx.random.normal((N,))
mx.eval(x)

def make(n):
    def f(x):
        y = x
        for i in range(n):
            y = y * 1.0001 if i % 2 == 0 else y + 0.0001
        return y
    return f

base = None
with open("staircase_raw.csv", "w") as fh:
    fh.write("n_ops,run_idx,time_ms\n")
    for n in range(1, 41):
        fc = mx.compile(make(n))
        for _ in range(3):
            mx.eval(fc(x))
        t = []
        for _ in range(10):
            t0 = time.perf_counter_ns()
            mx.eval(fc(x))
            t.append((time.perf_counter_ns() - t0) / 1e6)
        for i, v in enumerate(t):
            fh.write(f"{n},{i},{v:.4f}\n")
        med = float(np.median(t))
        base = base or med
        print(f"ops={n:2d} | median {med:7.3f} ms | x first-op time: {med/base:4.2f}")
