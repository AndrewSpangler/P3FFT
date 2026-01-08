import numpy as np
from panda3d.core import (
    NodePath, Shader, ShaderAttrib, ShaderBuffer,
    GeomEnums, ComputeNode, GraphicsPipeSelection,
    FrameBufferProperties, WindowProperties, GraphicsPipe
)
from .compilation import create_shader
from .declaration import CONFIG

class CastBuffer:
    def __init__(self, buff, n_items, cast=np.float32):
        self.buffer = buff
        self.n_items = n_items
        self.cast = cast
    def __len__(self):
        return self.n_items

class GPUMath:
    TYPE_MAP = {
        bool: "bool", np.bool_: "bool",
        np.int32: "int", np.uint32: "uint",
        np.float32: "float", np.float64: "double"
    }
    
    # Map GLSL types back to NumPy dtypes
    REV_MAP = {
        "bool": np.bool_, "int": np.int32, "uint": np.uint32, 
        "float": np.float32, "double": np.float64
    }

    def __init__(self, base, headless=False):
        self.base = base
        self.op_registry = {} # name -> arity -> sig -> node

        for name, arity_map in CONFIG.items():
            self.op_registry[name] = {}
            for arity, (expr, overloads) in arity_map.items():
                self.op_registry[name][arity] = {}
                for arg_types, res_type in overloads:
                    code = create_shader(expr, arg_types, res_type)
                    node = self._compile(code, res_type)
                    self.op_registry[name][arity][tuple(arg_types)] = node
            
            setattr(self, name, lambda *args, n=name: self._dispatch(n, *args))
        
        if headless:
            self._setup_headless()

    def _compile(self, code, res_type):
        shader = Shader.make_compute(Shader.SL_GLSL, code)
        node = NodePath(ComputeNode("math_node"))
        node.set_shader(shader)
        node.set_python_tag("dtype", self.REV_MAP[res_type])
        return node

    def _dispatch(self, op_name, *args):
        # Determine size for broadcasting
        n_items = 1
        for a in args:
            if isinstance(a, (CastBuffer, np.ndarray)):
                n_items = len(a)
                break

        # Normalize arguments
        buffers = []
        for a in args:
            if isinstance(a, (int, float, bool, np.generic)):
                buffers.append(self.push(np.full(n_items, a)))
            elif isinstance(a, np.ndarray):
                buffers.append(self.push(a))
            else:
                buffers.append(a)

        # Match Arity and Signature
        arity = len(buffers)
        sig = tuple(self.TYPE_MAP.get(b.cast, "float") for b in buffers)

        if arity not in self.op_registry[op_name]:
            raise TypeError(f"Operator '{op_name}' does not support {arity} arguments.")
        
        if sig not in self.op_registry[op_name][arity]:
            # Fallback: try to find a float version
            sig = tuple("float" for _ in range(arity))
            if sig not in self.op_registry[op_name][arity]:
                raise TypeError(f"No variant for '{op_name}' matching {sig}")

        node = self.op_registry[op_name][arity][sig]
        res_dtype = node.get_python_tag("dtype")

        # Dispatch
        res_size = n_items * np.dtype(res_dtype).itemsize
        result_buffer = ShaderBuffer("DR", res_size, GeomEnums.UH_stream)

        for i, b in enumerate(buffers):
            node.set_shader_input(f"D{i}", b.buffer)
        
        node.set_shader_input("DR", result_buffer)
        node.set_shader_input("nItems", int(n_items))

        self.base.graphics_engine.dispatch_compute(
            ((n_items + 63) // 64, 1, 1), 
            node.get_attrib(ShaderAttrib), 
            self.base.win.get_gsg()
        )
        return CastBuffer(result_buffer, n_items, cast=res_dtype)

    def push(self, data):
        if not isinstance(data, np.ndarray):
            data = np.array([data])
        sbuf = ShaderBuffer("Data", data.tobytes(), GeomEnums.UH_stream)
        return CastBuffer(sbuf, len(data), cast=data.dtype.type)

    def fetch(self, handle):
        gsg = self.base.win.get_gsg()
        raw = self.base.graphics_engine.extract_shader_buffer_data(handle.buffer, gsg)
        # Handle GLSL 4-byte bool alignment vs NumPy 1-byte bool
        if handle.cast == np.bool_:
            return np.frombuffer(raw, dtype=np.int32).astype(np.bool_)
        return np.frombuffer(raw, dtype=handle.cast)

    def _setup_headless(self):
        pipe = GraphicsPipeSelection.get_global_ptr().make_default_pipe()
        fb_prop = FrameBufferProperties()
        win_prop = WindowProperties.size(1, 1)
        self.base.win = self.base.graphics_engine.make_output(
            pipe, "math_headless", 0, fb_prop, win_prop, GraphicsPipe.BF_refuse_window
        )