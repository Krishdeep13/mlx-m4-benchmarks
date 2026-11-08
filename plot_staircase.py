import matplotlib.pyplot as plt
import re

ops = []
times = []

with open('run2/item5b.log', 'r') as f:
    for line in f:
        match = re.search(r'ops=\s*(\d+)\s*\|\s*median\s+([\d.]+)\s*ms', line)
        if match:
            ops.append(int(match.group(1)))
            times.append(float(match.group(2)))

plt.figure(figsize=(10, 6.5))
plt.step(ops, times, where='post', color='black', linewidth=2)
plt.plot(ops, times, 'ro', alpha=0.6, markersize=5)
plt.axvline(x=11, color='red', linestyle='--', alpha=0.5, label='multiples of max_compile_depth = 11 (11, 22, 33)')
plt.axvline(x=22, color='red', linestyle='--', alpha=0.5)
plt.axvline(x=33, color='red', linestyle='--', alpha=0.5)

plt.title('Compiled time vs chain length (MLX 0.32.4, M4 Pro)')
plt.xlabel('Number of Operations in Chain')
plt.ylabel('Median time per call (ms), 2^27 float32 elements')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()

# Adding the caption at the bottom
plt.figtext(0.5, 0.02, "Each dot is the median of 10 timed runs after 3 warmups.", ha="center", fontsize=10)
plt.subplots_adjust(bottom=0.12)

plt.savefig('staircase_plot.png', dpi=300)
print("Saved staircase_plot.png successfully.")
