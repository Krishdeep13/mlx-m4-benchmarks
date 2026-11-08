import platform
import subprocess
import mlx.core as mx
import sys
def get_output(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True).strip()
    except Exception as e:
        return f"Unavailable ({e})"

if __name__ == "__main__":
    print("=== ENVIRONMENT DUMP ===")
    print(f"Python Version: {platform.python_version()}")
    print(f"MLX Version: {mx.__version__}")
    print(f"MLX Device Info: {mx.device_info()}")
    print(f"CPU Brand (sysctl): {get_output('sysctl -n machdep.cpu.brand_string')}")
    print(f"macOS Version (sw_vers): {get_output('sw_vers')}")
    print(f"Xcode Select Path: {get_output('xcode-select -p')}")
    print(f"Xcode Version: {get_output('xcodebuild -version')}")
    print("\n--- PYTHON PACKAGES ---")
    print(get_output(f"{sys.executable} -m pip list 2>/dev/null | grep -i -E 'mlx|numpy|jax'"))
    print("\n--- POWER STATE (Ensure Plugged In) ---")
    print(get_output("pmset -g batt"))
    print(f"Python executable: {sys.executable}")   # shows if you are inside the venv
    print(get_output("pmset -g | grep -i -E 'lowpower|powermode' || echo 'no lowpowermode line'"))
