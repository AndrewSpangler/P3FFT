import numpy as np
import time
from panda3d.core import load_prc_file_data
from direct.showbase.ShowBase import ShowBase

PROCESS_SIZE = 64

from gpu_math import GPUMath

class ChainingDemo(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.hmath = GPUMath(self, headless=True)

    def run_test(self, N=10_000_000):
        a_cpu = np.random.randint(1, 100, N, dtype=np.int32)
        b_cpu = np.random.randint(1, 100, N, dtype=np.int32)

        c_cpu = a_cpu
        d_cpu = b_cpu.astype(np.float32)

        print(f"--- Chain: ((a + b) * a) - b ---")

        t0 = time.perf_counter()
        
        ga = self.hmath.push(a_cpu) # Upload A once
        gb = self.hmath.push(b_cpu) # Upload B once
        
        # Logic below happens entirely in VRAM
        g_sum  = self.hmath.add(ga, gb)
        g_prod = self.hmath.mult(g_sum, ga)
        g_res  = self.hmath.sub(g_prod, gb)
        
        final_gpu = self.hmath.fetch(g_res) # Download once
        t_gpu = time.perf_counter() - t0


        t0 = time.perf_counter()
        
        gc = self.hmath.push(c_cpu) # Upload A once
        gd = self.hmath.push(d_cpu) # Upload B once
        
        # Logic below happens entirely in VRAM
        g_sum  = self.hmath.add(gc, gd)
        g_prod = self.hmath.mult(g_sum, gc)
        g_res  = self.hmath.sub(g_prod, gd)
        
        final_gpu2 = self.hmath.fetch(g_res) # Download once
        t_gpu2 = time.perf_counter() - t0


        # --- CPU NUMPY EXECUTION ---
        t1 = time.perf_counter()
        final_cpu = ((a_cpu + b_cpu) * a_cpu) - b_cpu
        t_cpu = time.perf_counter() - t1

        print(f"NumPy Time:  {t_cpu:.5f}s")
        print(f"GPU Chained: {t_gpu:.5f}s")
        print(f"Speedup:     {t_cpu/t_gpu:.2f}x")
        print(f"Valid:       {np.array_equal(final_gpu, final_cpu)}")
        print(f"Valid2:       {np.array_equal(final_gpu, final_gpu2)}")


if __name__ == "__main__":
    # Headless configuration
    for k, v in {
        "window-type": "none",
        "audio-library-name":
        "null", "gl-debug": "#f"
    }.items():
        load_prc_file_data("", f"{k} {v}")

    app = ChainingDemo()
    app.run_test(N=2**25)