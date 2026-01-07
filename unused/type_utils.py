# type_utils.py
import numpy as np

DTYPE_TO_GLSL = {
    np.int32: "int",
    np.float32: "float",
    np.uint32: "uint",
}

GLSL_TO_DTYPE = {v: k for k, v in DTYPE_TO_GLSL.items()}

def get_glsl_type(dtype):
    return DTYPE_TO_GLSL.get(dtype, "int") # Default to int