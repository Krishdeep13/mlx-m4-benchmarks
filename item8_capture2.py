import sys, os, shutil
import mlx.core as mx

n_ops = int(sys.argv[1]); mode = sys.argv[2]
x = mx.random.normal((2**24,)); mx.eval(x)

def f(x):
    y = x
    for i in range(n_ops):
        y = y * 1.0001 if i % 2 == 0 else y + 0.0001
    return y

fn = mx.compile(f) if mode == "compiled" else f
for _ in range(5):
    mx.eval(fn(x))

path = f"cap2_{mode}_{n_ops}.gputrace"
if os.path.exists(path):
    shutil.rmtree(path)
print("start_capture ->", path)
mx.metal.start_capture(path)
out = fn(x)
mx.eval(out)
mx.metal.stop_capture()
total = sum(os.path.getsize(os.path.join(d, n)) for d, _, fs in os.walk(path) for n in fs)
print("capture size (bytes):", total)
