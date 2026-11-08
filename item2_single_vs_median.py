import time
import numpy as np
import mlx.core as mx

a = mx.random.normal((4096, 4096))
b = mx.random.normal((4096, 4096))
mx.eval(a, b)
print("dtype:", a.dtype)

def run(n):
    out = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        mx.eval(a @ b)
        out.append((time.perf_counter_ns() - t0) / 1e6)
    return out

rows = []  # (block, iteration, time_ms)

# Initial 200-run baseline
t = run(200)
rows += [("initial", i, x) for i, x in enumerate(t)]
print(f"[initial] first {t[0]:.3f} ms | median {np.median(t):.3f} ms | min {np.min(t):.3f} ms")
print("[initial] first 15:", [round(x, 3) for x in t[:15]])

# Sleep sweep, repeated 5 times per sleep value
REPEATS = 5
for s in [0.1, 1, 5, 10]:
    firsts, seconds = [], []
    for r in range(REPEATS):
        time.sleep(s)
        t = run(20)
        rows += [(f"sleep_{s}s_rep{r}", i, x) for i, x in enumerate(t)]
        firsts.append(t[0])
        seconds.append(t[1])
    print(f"[sleep {s}s] first-run ms: {[round(x, 1) for x in firsts]} | median {np.median(firsts):.1f}")
    print(f"[sleep {s}s] second-run ms: {[round(x, 2) for x in seconds]}")

with open("single_vs_median_raw.csv", "w") as f:
    f.write("block,iteration,time_ms\n")
    for blk, i, x in rows:
        f.write(f"{blk},{i},{x:.4f}\n")
print("wrote single_vs_median_raw.csv")
