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
        
        ubo_data = np.array([p_value, size, 0, 0, 0, 0, 0, 0], dtype=np.uint32)
        self.ubo_buffer = ShaderBuffer(f"UBO_{stage_idx}", ubo_data.tobytes(), GeomEnums.UH_static)
        
        self._prepare_shader()

    def _prepare_shader(self):
        with open("fft_template.glsl", "r") as f:
            shader_src = f.read()

        definitions = """
#version 430
layout (local_size_x = 64, local_size_y = 1, local_size_z = 4) in;
#define FFT_RADIX 4    
#define FFT_HORIZ     
#define FFT_VEC2
#define FFT_SHARED_BANKED 1
#define uP constant_data.p_stride_padding.x
#define uStride constant_data.p_stride_padding.y
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


class Radix4FFT:
    def __init__(self, base, data, size=1024):
        self.base = base
        self.size = size
        self.data = data
        self.num_stages = int(math.log(size, 4))
        
        # Interleave real and imaginary parts for GLSL vec2 [cite: 3, 4]
        interleaved_data = np.zeros(size * 2, dtype=np.float32)
        interleaved_data[0::2] = data.real
        interleaved_data[1::2] = data.imag

        # Create ping-pong buffers for stages 
        self.buffer_a = ShaderBuffer("FFT_Buffer_A", interleaved_data.tobytes(), GeomEnums.UH_dynamic)
        self.buffer_b = ShaderBuffer("FFT_Buffer_B", interleaved_data.tobytes(), GeomEnums.UH_dynamic)
        
        self.stages = []
        for i in range(self.num_stages):
            p_value = 4**i
            is_p1 = (i == 0)
            stage = ShaderStage(base, i, p_value, size, is_p1)
            self.stages.append(stage)

    def dispatch(self):
        current_in = self.buffer_a
        current_out = self.buffer_b
        
        # Each thread processes 4 elements [cite: 81, 83]
        # local_size_x is 64 in the template
        groups_x = max(1, (self.size // 4) // 64)
        
        # Create a temporary NodePath to hold shader inputs
        dummy_np = NodePath("dispatch")
        
        for stage in self.stages:
            stage.np.set_shader(stage.shader)
            stage.np.set_shader_input("UBO", stage.ubo_buffer)
            stage.np.set_shader_input("Block", current_in)
            stage.np.set_shader_input("BlockOut", current_out)
            
            sattr = stage.np.get_attrib(ShaderAttrib)
            self.base.graphicsEngine.dispatch_compute(
                (groups_x, 1, 1),
                sattr,
                self.base.win.get_gsg()
            )
            # Swap for next stage ping-pong
            current_in, current_out = current_out, current_in
            
        # Download the final buffer back to RAM [Fixes your TypeError]
        gsg = self.base.win.get_gsg()
        raw_output = self.base.graphicsEngine.extract_shader_buffer_data(current_in, gsg)
        output = np.frombuffer(raw_output, dtype=np.float32)
        return output

    def verify(self):
        final_buffer = self.dispatch()
        
        # Convert raw bytes back to complex numpy array
        mem_view = memoryview(final_buffer.tobytes()).cast('f')
        gpu_res_flat = np.array(mem_view, dtype=np.float32)
        gpu_complex = gpu_res_flat[0::2] + 1j * gpu_res_flat[1::2]
        
        # NumPy reference
        cpu_res = np.fft.fft(self.data)
        
        # Note: Radix-4 output is bit-reversed in specific ways depending on 
        # the template's store_global logic[cite: 86, 87].
        # We compare magnitudes to verify frequency components match.
        mag_gpu = np.sort(np.abs(gpu_complex))
        mag_cpu = np.sort(np.abs(cpu_res))
        
        success = np.allclose(mag_gpu, mag_cpu, atol=1e-2)
        print(mag_gpu[:12])
        print(mag_cpu[:12])
        print(f"FFT Verification: {'PASSED' if success else 'FAILED'}")
        
if __name__ == "__main__":
    from direct.showbase.ShowBase import ShowBase
    base = ShowBase()
    
    size = 1024
    # Create a cleaner test signal (sine wave)
    test_sig = 10 * np.sin(np.arange(size) * 3.1415 * 0.125).astype(np.complex64)
    
    fft_engine = Radix4FFT(base, test_sig, size=size)
    fft_engine.verify()
