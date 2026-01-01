import numpy as np
import math
from panda3d.core import (
    NodePath, Shader, ShaderBuffer, 
    GeomEnums, ComputeNode, ShaderAttrib
)

class ShaderStage:
    def __init__(self, base, stage_idx, p_value, size, is_p1=False):
        self.base = base
        self.stage_idx = stage_idx
        self.p_value = p_value
        self.size = size
        self.is_p1 = is_p1
        self.np = NodePath(f"ShaderStage_{stage_idx}")
        
        ubo_data = np.array([p_value, size, 0, 0, 0, 0, 0, 0], dtype=np.uint32)
        self.ubo_buffer = ShaderBuffer(f"UBO_{stage_idx}", ubo_data.tobytes(), GeomEnums.UH_static)
        
        self._prepare_shader()

    def _prepare_shader(self):
        with open("fft_template.glsl", "r") as f:
            shader_src = f.read()

        definitions = """
#version 430
layout (local_size_x = 64) in;
#define FFT_RADIX 4
#define FFT_VEC2      
#define FFT_HORIZ     
#define FFT_SHARED_BANKED 1
#define uP constant_data.p_stride_padding.x
#define uStride constant_data.p_stride_padding.y
#define FFT_P1        // First pass (simplest case)
        """.strip() + "\n"
        
        if self.is_p1:
            definitions += "#define FFT_P1\n"

        main_entry = """
        void main() {
            FFT4();
        }
        """
        
        shader_src = shader_src.replace("{{ DEFINITIONS }}", definitions)
        shader_src = shader_src.replace("{{ MAIN }}", main_entry)

        self.shader = Shader.make_compute(Shader.SL_GLSL, shader_src)
        self.np.set_shader(self.shader)
        self.np.set_shader_input("UBO", self.ubo_buffer)

class Radix4FFT:
    def __init__(self, base, size=1024):
        self.base = base
        self.size = size
        self.num_stages = int(math.log(size, 4))
        
        # Pre-create stages
        self.stages = []
        for i in range(self.num_stages):
            p = 4**i
            self.stages.append(ShaderStage(base, i, p, size, is_p1=(i == 0)))

    def process(self, signal):
        interleaved = np.zeros(self.size * 2, dtype=np.float32)
        interleaved[0::2] = signal.real
        interleaved[1::2] = signal.imag
        
        buf_a = ShaderBuffer("BufferA", interleaved.tobytes(), GeomEnums.UH_dynamic)
        buf_b = ShaderBuffer("BufferB", interleaved.tobytes(), GeomEnums.UH_dynamic)
        
        current_in, current_out = buf_a, buf_b
        
        num_groups = max(1, (self.size // 4) // 64)

        for stage in self.stages:
            stage.np.set_shader_input("Block", current_in)
            stage.np.set_shader_input("BlockOut", current_out)
            sattr = stage.np.get_attrib(ShaderAttrib)
            gsg = self.base.win.get_gsg()
            self.base.graphicsEngine.dispatch_compute(
                (num_groups, 1, 1), 
                sattr, 
                gsg
            )
            
            current_in, current_out = current_out, current_in

        raw_output = self.base.graphicsEngine.extract_shader_buffer_data(current_in, gsg)
        raw_output = np.frombuffer(raw_output, dtype=np.float32)
        complex_gpu = raw_output[0::2] + 1j * raw_output[1::2]
        
        return complex_gpu

    def verify(self, signal):
        print(f"--- Starting {self.size}-point Radix-4 Verification ---")
        gpu_res = self.process(signal)
        cpu_res = np.fft.fft(signal)
        
        if np.allclose(gpu_res, cpu_res, atol=1e-1):
            print("GPU output matches NumPy!")
        else:
            print("Discrepancy detected.")
            print(gpu_res[:3])
            print(cpu_res[:3])

if __name__ == "__main__":
    from direct.showbase.ShowBase import ShowBase
    base = ShowBase()
    
    size = 1024
    test_sig = 5 * np.sin(np.arange(size) * 3.1415 * 0.1234).astype(np.complex64)
    fft_engine = Radix4FFT(base, size=size)
    fft_engine.verify(test_sig)
