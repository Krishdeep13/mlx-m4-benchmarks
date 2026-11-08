import mlx.core as mx

print("--- A. scalar as mx.array: does it retrace? ---")
trace = [0]
def p(x, k):
    trace[0] += 1
    return x * k
pc = mx.compile(p)
x = mx.ones((4,))
for k in [2.0, 3.0, 4.0]:
    mx.eval(pc(x, mx.array(k)))
print("body ran", trace[0], "times for mx.array k = 2, 3, 4")

print("--- B. correctness: 5 fns x 3 dtypes x 6 shapes ---")
fns = {
    "elem":    lambda x: mx.sin(x * 2.0 + 1.0) * x - 0.5,
    "where":   lambda x: mx.where(x > 0, x * 2.0 + 1.0, x * 0.5 - 1.0),
    "ln":      lambda x: (x - x.mean(-1, keepdims=True)) * mx.rsqrt(x.var(-1, keepdims=True) + 1e-5),
    "softmax": lambda x: mx.softmax(x, axis=-1),
    "tanh":    lambda x: mx.tanh(x) + mx.exp(-mx.abs(x)),
}
shapes = [(8, 8), (64, 64), (127, 129), (512, 512), (1024, 1024), (4096, 4096)]
fails, total, maxd = 0, 0, 0.0
for name, f in fns.items():
    fc = mx.compile(f)
    for dt in [mx.float32, mx.float16, mx.bfloat16]:
        for s in shapes:
            x = mx.random.normal(s).astype(dt)
            a, b = f(x), fc(x)
            mx.eval(a, b)
            d = mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item()
            maxd = max(maxd, d); total += 1
            if d != 0.0:
                fails += 1; print("DIFF", name, dt, s, d)
print(f"{total} cases, {fails} with nonzero diff, max diff {maxd}")

print("--- C. edge values ---")
e = mx.array([0.0, -0.0, 1e30, -1e30, float("inf"), float("-inf"), float("nan")])
for name, f in fns.items():
    if name in ("ln", "softmax"):
        continue
    a, b = f(e), mx.compile(f)(e)
    try:
        print(name, "equal incl. NaN:", mx.array_equal(a, b, equal_nan=True).item())
    except Exception as ex:
        print(name, "array_equal error:", ex)

print("--- D. gradients ---")
for name in ["elem", "tanh"]:
    f = fns[name]
    g = lambda x, f=f: f(x).sum()
    x = mx.random.normal((1024, 1024))
    ga, gb = mx.grad(g)(x), mx.compile(mx.grad(g))(x)
    print(name, "grad max diff:", mx.abs(ga - gb).max().item())
