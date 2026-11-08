import time
import numpy as np
import mlx.core as mx

def med(fn, warm=5, runs=25):
    for _ in range(warm): fn()
    out = []
    for _ in range(runs):
        t0 = time.perf_counter_ns(); fn(); out.append((time.perf_counter_ns() - t0) / 1e6)
    return np.array(out)

print("--- matmul sweep ---")
with open("matmul_sweep_raw.csv", "w") as fh:
    fh.write("dtype,n,run_idx,time_ms,tflops\n")
    for dt in [mx.float32, mx.float16, mx.bfloat16]:
        dname = str(dt).split(".")[-1]
        for n in [64, 256, 1024, 2048, 4096, 8192]:
            a = mx.random.normal((n, n)).astype(dt); b = mx.random.normal((n, n)).astype(dt)
            mx.eval(a, b)
            t = med(lambda: mx.eval(a @ b), runs=15 if n == 8192 else 25)
            for i, v in enumerate(t):
                fh.write(f"{dname},{n},{i},{v:.4f},{2*n**3/(v/1000)/1e12:.3f}\n")
            m = np.median(t)
            print(f"{dname:9s} n={n:5d} median {m:9.4f} ms | {2*n**3/(m/1000)/1e12:6.2f} TFLOPS")

print("--- tiny-op sync cost: 20 elementwise ops on 64x64 ---")
x = mx.random.normal((64, 64)); mx.eval(x)
def one_eval():
    y = x
    for _ in range(20): y = y * 1.0001
    mx.eval(y)
def eval_each():
    y = x
    for _ in range(20):
        y = y * 1.0001
        mx.eval(y)
b1, b2 = med(one_eval, runs=100), med(eval_each, runs=100)
print(f"one eval:  median {np.median(b1):.4f} ms | p10-p90 [{np.percentile(b1,10):.4f}, {np.percentile(b1,90):.4f}]")
print(f"eval each: median {np.median(b2):.4f} ms | p10-p90 [{np.percentile(b2,10):.4f}, {np.percentile(b2,90):.4f}]")
print(f"extra cost per eval: {(np.median(b2)-np.median(b1))/20*1000:.1f} µs")
