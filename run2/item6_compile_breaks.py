import time
import mlx.core as mx

x = mx.ones((4,))

print("--- 1a. python float global changed after first call ---")
SCALE = 2.0
def g(x): return x * SCALE
gc = mx.compile(g)
print("compiled before:", gc(x).tolist())
SCALE = 3.0
print("eager after:    ", g(x).tolist())
print("compiled after: ", gc(x).tolist())

print("--- 1b. mx.array global changed after first call ---")
W = mx.array(2.0)
def g2(x): return x * W
g2c = mx.compile(g2)
print("compiled before:", g2c(x).tolist())
W = mx.array(3.0)
print("eager after:    ", g2(x).tolist())
print("compiled after: ", g2c(x).tolist())

print("--- 2. side effect in body ---")
calls = []
def s(x):
    calls.append(1)
    return x + 1
sc = mx.compile(s)
for _ in range(5):
    mx.eval(sc(x))
print("python body ran", len(calls), "times in 5 compiled calls")

print("--- 3. RNG inside compiled fn ---")
def r(x): return x + mx.random.normal(x.shape)
rc = mx.compile(r)
a, b = rc(x), rc(x)
print("compiled, no state: identical outputs?", mx.array_equal(a, b).item())
print("eager: identical outputs?", mx.array_equal(r(x), r(x)).item())
state = [mx.random.state]
rs = mx.compile(r, inputs=state, outputs=state)   # verify this API in the MLX docs
a, b = rs(x), rs(x)
print("compiled, with state: identical outputs?", mx.array_equal(a, b).item())

print("--- 4. value-dependent control flow (.item) ---")
def it(x):
    if x.sum().item() > 0:
        return x * 2
    return x
try:
    print(mx.compile(it)(x))
except Exception as e:
    print(type(e).__name__, ":", str(e)[:300])

print("--- 5. retrace on shape / dtype change ---")
trace = [0]
def h(x):
    trace[0] += 1
    return mx.sin(x) * 2 + 1
hc = mx.compile(h)
A = mx.random.normal((4096, 4096)); Bm = mx.random.normal((2048, 2048))
A16 = A.astype(mx.float16)
mx.eval(A, Bm, A16)
def t(label, arr):
    t0 = time.perf_counter_ns()
    mx.eval(hc(arr))
    ms = (time.perf_counter_ns() - t0) / 1e6
    print(f"{label:22s} {ms:8.2f} ms | python body ran so far: {trace[0]}")
t("A 1st", A); t("A 2nd", A); t("A 3rd", A)
t("B (new shape) 1st", Bm); t("B 2nd", Bm)
t("A again", A)
t("A fp16 (new dtype)", A16)

print("--- 6. shapeless=True ---")
trace2 = [0]
def h2(x):
    trace2[0] += 1
    return mx.sin(x) * 2 + 1
hs = mx.compile(h2, shapeless=True)
mx.eval(hs(A)); mx.eval(hs(Bm))
print("python body ran", trace2[0], "times for 2 different shapes")

print("--- 7. python scalar argument changed ---")
trace3 = [0]
def p(x, k):
    trace3[0] += 1
    return x * k
pc = mx.compile(p)
mx.eval(pc(x, 2.0)); mx.eval(pc(x, 2.0)); mx.eval(pc(x, 3.0))
print("python body ran", trace3[0], "times for k=2, 2, 3")
