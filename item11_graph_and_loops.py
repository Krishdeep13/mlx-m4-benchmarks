import time
import mlx.core as mx

print("--- A. lazy graph to dot file ---")
x = mx.random.normal((1024,)); mx.eval(x)
def chain(x):
    y = x
    for i in range(6):
        y = y * 1.0001 if i % 2 == 0 else y + 0.0001
    return y
try:
    mx.export_to_dot("graph_eager.dot", chain(x))
    print("wrote graph_eager.dot")
except Exception as e:
    print("export_to_dot failed:", type(e).__name__, e)

print("--- B. compile once vs compile every iteration ---")
big = mx.random.normal((2**20,)); mx.eval(big)
count = [0]
def body(x):
    count[0] += 1
    y = x
    for i in range(10):
        y = y * 1.0001 if i % 2 == 0 else y + 0.0001
    return y

fc = mx.compile(body)
mx.eval(fc(big)); count[0] = 0
t0 = time.perf_counter()
for _ in range(20): mx.eval(fc(big))
once = (time.perf_counter() - t0) * 1000 / 20
print(f"compile once:   {once:.3f} ms/iter | traces in loop: {count[0]}")

count[0] = 0
t0 = time.perf_counter()
for _ in range(20):
    mx.eval(mx.compile(lambda a: body(a))(big))
each = (time.perf_counter() - t0) * 1000 / 20
print(f"compile each:   {each:.3f} ms/iter | traces in loop: {count[0]}")

print("--- C. fixed shape vs new shape every call ---")
fc2 = mx.compile(body)
mx.eval(fc2(big)); count[0] = 0
t0 = time.perf_counter()
for _ in range(20): mx.eval(fc2(big))
fixed = (time.perf_counter() - t0) * 1000 / 20
print(f"fixed shape:    {fixed:.3f} ms/iter | traces: {count[0]}")

inputs = [mx.random.normal((2**20 + n + 1,)) for n in range(20)]
mx.eval(inputs)
count[0] = 0
t0 = time.perf_counter()
for a in inputs: mx.eval(fc2(a))
varied = (time.perf_counter() - t0) * 1000 / 20
print(f"varying shape:  {varied:.3f} ms/iter | traces: {count[0]}")
