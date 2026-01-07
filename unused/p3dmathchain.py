import numpy as np
import time
from panda3d.core import (
    NodePath, Shader, ShaderAttrib, ShaderBuffer, 
    GeomEnums, ComputeNode, load_prc_file_data,
    GraphicsPipeSelection, FrameBufferProperties,
    WindowProperties, GraphicsPipe
)
from direct.showbase.ShowBase import ShowBase
from lib import CastBuffer

PROCESS_SIZE = 64

from lib.templating.basic import create_one_input_shader, create_two_input_shader
from lib.templating.basic_functions import ONE_INPUT, TWO_INPUT
from lib.type_utils import get_glsl_type, GLSL_TO_DTYPE

class HyperMath:
    # Map NumPy dtypes to GLSL type strings
    TYPE_MAP = {
        np.int32: "int",
        np.float32: "float"
    }
    # Reverse map to restore dtypes to CastBuffers
    REV_MAP = {v: k for k, v in TYPE_MAP.items()}

    def __init__(self, app, one_cfg, two_cfg):
        self.app = app
        self._setup_context() # Logic from p3dmathchain.py
        self.op_registry = {}

        # Compile 1-Input Overloads
        for name, (expr, overloads) in one_cfg.items():
            self.op_registry[name] = {}
            for (args, res) in overloads:
                glsl_code = create_one_input_shader(expr, args[0], res)
                self.op_registry[name][(args[0],)] = self._compile(glsl_code, res)

        # Compile 2-Input Overloads
        for name, (expr, overloads) in two_cfg.items():
            if name not in self.op_registry: self.op_registry[name] = {}
            for (args, res) in overloads:
                glsl_code = create_two_input_shader(expr, args[0], args[1], res)
                self.op_registry[name][(args[0], args[1])] = self._compile(glsl_code, res)
            
            # Attach the dynamic method to allow .name(a, b) access
            setattr(self, name, lambda a, b, n=name: self._dispatch(n, a, b))

    def _compile(self, code, res_type):
        """Creates a ComputeNode and stores the result type metadata."""
        shader = Shader.make_compute(Shader.SL_GLSL, code)
        node = NodePath(ComputeNode("math_node"))
        node.set_shader(shader)
        node.set_python_tag("dtype", self.REV_MAP[res_type])
        return node

    def _dispatch(self, op_name, a, b):
        """Selects the correct shader based on input buffer types."""
        buf_a = a if isinstance(a, CastBuffer) else self.push(a)
        buf_b = b if isinstance(b, CastBuffer) else self.push(b)

        # Build lookup signature (e.g., ("int", "int"))
        sig = (self.TYPE_MAP[buf_a.cast], self.TYPE_MAP[buf_b.cast])
        
        if sig not in self.op_registry[op_name]:
            raise TypeError(f"No shader variant for {op_name} with types {sig}")

        node = self.op_registry[op_name][sig]
        res_dtype = node.get_python_tag("dtype")
        n_items = buf_a.n_items

        # Execution logic
        out_sbuf = ShaderBuffer("DR", n_items * res_dtype().itemsize, GeomEnums.UH_stream)
        node.set_shader_input("DA", buf_a.buffer)
        node.set_shader_input("DB", buf_b.buffer)
        node.set_shader_input("DR", out_sbuf)
        node.set_shader_input("nItems", int(n_items))

        self.app.graphics_engine.dispatch_compute(
            ((n_items + 63) // 64, 1, 1), 
            node.get_attrib(ShaderAttrib), 
            self.app.win.get_gsg()
        )
        return CastBuffer(out_sbuf, n_items, cast=res_dtype)

    def _setup_context(self):
        pipe = GraphicsPipeSelection.get_global_ptr().make_default_pipe()
        fb_prop = FrameBufferProperties()
        win_prop = WindowProperties.size(1, 1)
        self.app.win = self.app.graphics_engine.make_output(
            pipe, "math_headless", 0, fb_prop, win_prop, GraphicsPipe.BF_refuse_window
        )

    def push(self, data):
        """Manually upload a NumPy array to the GPU."""
        data = np.ascontiguousarray(data, dtype=np.int32)
        sbuf = ShaderBuffer("Data", data.tobytes(), GeomEnums.UH_stream)
        return CastBuffer(sbuf, len(data))

    def fetch(self, gpu_handle):
        """Download a GPU buffer back to a NumPy array."""
        gsg = self.app.win.get_gsg()
        raw = self.app.graphics_engine.extract_shader_buffer_data(gpu_handle.buffer, gsg)
        return np.frombuffer(raw, dtype=gpu_handle.cast)



class ChainingDemo(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.hmath = HyperMath(self, ONE_INPUT, TWO_INPUT)

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
    app.run_test(N=2**24)