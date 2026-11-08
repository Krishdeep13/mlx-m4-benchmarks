import sys, time
import numpy as np
import mlx.core as mx

n_ops = int(sys.argv[1]); mode = sys.argv[2]
x = mx.random.normal((2**24,)); mx.eval(x)    # smaller array: capture files get big

def f(x):
    y = x
    for i in range(n_ops):
        y = y * 1.0001 if i % 2 == 0 else y + 0.0001
    return y

fn = mx.compile(f) if mode == "compiled" else f
for _ in range(5): mx.eval(fn(x))
t = []
for _ in range(10):
    t0 = time.perf_counter_ns(); mx.eval(fn(x)); t.append((time.perf_counter_ns() - t0) / 1e6)
print(f"{mode} {n_ops} ops @2^24: median {np.median(t):.3f} ms")

path = f"capture_{mode}_{n_ops}.gputrace"
mx.metal.start_capture(path)     # VERIFY this name in the MLX docs (Metal debugger page)
mx.eval(fn(x))
mx.metal.stop_capture()
print("saved", path)
