import numpy as np

from panda3d.core import (
    Texture, CardMaker, TextureStage, GeomNode,
    Shader, ComputeNode, NodePath, Vec2, Vec3
)
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from collections import deque
import threading
import queue
import struct
import colorsys

class WaterfallTexture:    
    def __init__(self, width:int, height:int):
        """
        width: Number of bins (texture width)
        height: Number of samples (texture height)
        """
        self.width = width
        self.height = height
        
        # Circular buffer state
        self.write_row = 0
        self.wraparound_count = 0
        
        self.data_texture = Texture("waterfall_data")
        self.data_texture.setup2dTexture(
            width, height,
            Texture.T_float,
            Texture.F_r32
        )
        self.data_texture.setWrapU(Texture.WMClamp)
        self.data_texture.setWrapV(Texture.WMRepeat)  # Repeat for circular display
        self.data_texture.setMinfilter(Texture.FTNearest)
        self.data_texture.setMagfilter(Texture.FTNearest)
        
        self.data_buffer = np.zeros((height, width), dtype=np.float32)
        self.data_texture.setRamImage(self.data_buffer.tobytes())
                
    def add_row(self, magnitudes):
        if len(magnitudes) != self.width:
            # Resample if needed
            magnitudes = np.interp(
                np.linspace(0, len(magnitudes)-1, self.width),
                np.arange(len(magnitudes)),
                magnitudes
            )
        
        # Update data buffer (circular buffer)
        self.data_buffer[self.write_row] = magnitudes.astype(np.float32)
        
        # Write to texture memory
        ram_image = self.data_texture.modifyRamImage()
        if ram_image:
            # Create numpy view of texture memory
            ram_view = np.frombuffer(ram_image, dtype=np.float32).reshape(
                self.height, self.width
            )
            
            # Write updated row
            ram_view[self.write_row] = magnitudes
        
        # Advance circular buffer
        self.write_row = (self.write_row + 1) % self.height
        if self.write_row == 0:
            self.wraparound_count += 1
    
    def clear(self):
        """Clear the waterfall display"""
        self.data_buffer.fill(self.min_db)
        self.data_texture.setRamImage(self.data_buffer.tobytes())
        self.write_row = 0
        self.wraparound_count = 0
    
    def set_db_range(self, min_db, max_db):
        """Set the dB range for visualization"""
        self.min_db = min_db
        self.max_db = max_db
  
    def get_write_row_normalized(self):
        """Get normalized write position (0-1)"""
        return (self.write_row + 1) / self.height
    
    
