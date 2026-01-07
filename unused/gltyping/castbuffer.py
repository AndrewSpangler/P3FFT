from panda3d.core import ShaderBuffer
from direct.showbase.ShowBase import ShowBase
import numpy as np

class CastBuffer:
    """A handle to data residing on the GPU."""
    def __init__(
        self,
        buff    : ShaderBuffer,
        n_items : int,
        base    : ShowBase = None,
        cast    : "dtype"  = np.int32
    ):
        self.buffer  = buff
        self.n_items = n_items
        self.base    = base
        self.cast    = cast