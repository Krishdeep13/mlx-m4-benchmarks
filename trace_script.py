import sys, time
import mlx.core as mx

n_ops = int(sys.argv[1])    # TOTAL elementwise ops
mode = sys.argv[2]          # "eager" or "compiled"
x = mx.random.normal((2**27,)); mx.eval(x)

def f(x):
    y = x
    for i in range(n_ops):
        y = y * 1.0001 if i % 2 == 0 else y + 0.0001
    return y

fn = mx.compile(f) if mode == "compiled" else f
for _ in range(3):          # warmup: compile and shader build happen here
    mx.eval(fn(x))
time.sleep(3)               # gap 1: the lanes go quiet
for _ in range(3):          # the 3 calls to inspect
    mx.eval(fn(x))
    time.sleep(0.5)         # gap between calls so each is separable
time.sleep(3)
print("done", n_ops, mode)
