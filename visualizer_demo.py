import numpy as np
import colorsys
from panda3d.core import Vec2, Vec3, Vec4
from scipy.ndimage import gaussian_filter1d

# Local
from graphing import Graph
from base import TestBase
from fft import Radix16FFT
from audio_capture import AudioCapture

class TestApp(TestBase):
    def __init__(
        self,
        smoothing_width:    int   =  1,   # Width of smoothing
        smoothing_sigma:    float =  1.0, # Smoothing intensity, higher for smoother
        amplification:      float =  50,  # Post-Process amplification
        history_frames:     int   =  100, # Number of itterations
        direction:          bool  =  True,# Placement order
        base_rotation:      float =  0.0, # Starting rotation
        rotation_speed:     float =  8.0, # Degrees / s
        rotation_per_layer: float =  360 / 100, # Rotation per graph layer
        wave_speed:         float =  1.0, # Wave movement speed
        wave_amplitude:     float =  0.3, # How far the layers move
        wave_frequency:     float =  3.0, # How many waves per cycle
        starting_color_hue: float =  0.0, # Starting color
        base_color_speed:   float =  0.001, # How quickly colors cycle
        base_radius:        float =  2.75, # Base max radius
        base_thickness:     float =  2.75,# Base graph thickness
    ):
        TestBase.__init__(
            self,
            Radix16FFT,
            AudioCapture,
            update_callback=self._update_graph,
        )
        self.smoothing_width = smoothing_width
        self.smoothing_sigma = smoothing_sigma
        self.amplification = amplification
        self.num_history_frames = history_frames
        # 0 = outside-in (newest on outside)
        # 1 = inside-out (newest on inside)
        self.direction = direction
        self.base_rotation = base_rotation
        self.rotation_speed = rotation_speed
        self.rotation_per_layer = rotation_per_layer
        self.wave_speed = wave_speed
        self.wave_amplitude = wave_amplitude
        self.wave_frequency = wave_frequency
        self.color_hue = starting_color_hue
        self.base_color_speed = base_color_speed
        self.base_radius = base_radius
        self.base_thickness = base_thickness
        self.num_freq_bins = self.fft_size // 4

        self.wave_time = 0

        self.history_buffer = []
        self.graphs = []

        for i in range(self.num_history_frames, -1, -1):
            # Calculate age factor: 0 = newest, 1 = oldest
            age_factor = i / self.num_history_frames
            # Color fades with age
            color_shift = 1.0 - age_factor * 0.1

            if self.direction:
                thickness = base_thickness + (1 - age_factor)
                radius = base_radius * (1.0 - age_factor)
            else:
                thickness = base_thickness - age_factor
                radius = base_radius * (1.0 - age_factor)

            alpha = 1.0
            show_bg = (i == 0)

            graph = Graph(
                parent=self.aspect2d,
                min_val=0.0,
                max_val=self.fft_size / 4,
                max_values=self.num_freq_bins + 1,
                line_color=Vec4(0.2 * color_shift, 0.8 * color_shift, 0.2 * color_shift, alpha),
                bg_color=Vec4(0.1, 0.1, 0.2, 0.9) if show_bg else Vec4(0, 0, 0, 0),
                position=Vec2(0, 0),
                size=Vec2(2.0, 2.0),
                thickness=thickness,
                circular=True,
                base_radius=radius
            )
            self.graphs.append(graph)

        self.graphs.reverse()

        self.setup_controls()
        self.audio.start()

    def hsv_to_rgb(self, h, s, v):
        """Convert HSV to RGB, returns Vec4 with alpha=1.0"""
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return Vec4(r, g, b, 1.0)

    def _update_graph(self, complex_result, dt:float):
        # "High-pass" filter
        magnitudes = np.abs(complex_result[30:(self.fft_size // 4) + 30])
        
        # Calculate total volume for color speed
        total_volume = np.sum(magnitudes)
        volume_normalized = np.clip(total_volume / 100000.0, 0.0, 5.0)  # Normalize and clamp
        self.volume_multiplier = volume_normalized
        
        # Janky scaling but looks decent
        num_bins = len(magnitudes)
        freq_scale = np.linspace(0.75, 1.75, num_bins)
        freq_scale = freq_scale ** 1.5
        magnitudes = magnitudes * freq_scale
        freq_scale = np.exp(np.linspace(np.log(1.0), np.log(2.0), num_bins))
        magnitudes = magnitudes * freq_scale

        # Scale for visualization
        magnitudes = magnitudes * 100.0
        
        # Circular smoothing
        magnitudes_circular = np.concatenate([
            magnitudes[-self.smoothing_width:],
            magnitudes,
            magnitudes[:self.smoothing_width]
        ])
        
        smoothed_circular = gaussian_filter1d(magnitudes_circular, sigma=self.smoothing_sigma)
        magnitudes = smoothed_circular[self.smoothing_width:-self.smoothing_width]
        
        # Close the loop
        values_circular = np.concatenate([magnitudes, [magnitudes[0]]])
        
        # Calculate volume-based radius scale (1.0 = base, scales up with volume)
        radius_scale = 1.15 * (0.5 + self.volume_multiplier * 80)
        
        # Update history buffer with both values and radius scale
        self.history_buffer.insert(0, (values_circular, radius_scale))
        
        if len(self.history_buffer) > self.num_history_frames + 1:
            self.history_buffer = self.history_buffer[:self.num_history_frames + 1]
        
        # Update base rotation (all layers rotate together)
        rotation_delta = self.rotation_speed * dt
        self.base_rotation = (self.base_rotation + rotation_delta) % 360.0
        
        # Update wave time for moving tunnel effect
        self.wave_time += self.wave_speed * dt
        
        # Update color hue (faster with higher volume)
        color_speed = self.base_color_speed * (1.0 + self.volume_multiplier * 2.0)
        self.color_hue = (self.color_hue + color_speed) % 1.0
        
        # Update all graph layers based on direction
        for i, graph in enumerate(self.graphs):
            # Determine which history buffer index to use based on direction
            if self.direction == 0:
                # Outside-in: index 0 (newest) goes to outer graph (index 0)
                history_index = i
            else:
                # Inside-out: index 0 (newest) goes to inner graph (last index)
                history_index = len(self.graphs) - 1 - i
            
            # Get values and radius scale from history
            if history_index < len(self.history_buffer):
                values, radius_scale = self.history_buffer[history_index]
                graph.set_values(values)
                # Apply the radius scale from this specific history frame
                graph.set_base_radius(graph.initial_base_radius * radius_scale)
            else:
                graph.set_values(np.zeros(self.num_freq_bins + 1, dtype=np.float32))
                graph.set_base_radius(graph.initial_base_radius)
            
            # Set rotation with offset for each layer
            layer_rotation = self.base_rotation + (i * self.rotation_per_layer)
            graph.set_rotation(layer_rotation)
            
            # Apply wave effect for moving tunnel
            # Each layer gets offset based on its depth and time
            layer_depth = i / len(self.graphs)
            wave_phase = self.wave_time + layer_depth * self.wave_frequency * np.pi
            
            # Create circular wave motion
            x_offset = self.wave_amplitude * np.sin(wave_phase)
            z_offset = self.wave_amplitude * np.cos(wave_phase * 1.3)  # Different frequency for interesting pattern
            
            graph.set_position_offset(x_offset, z_offset)
            
            # Set color based on hue rotation and layer
            # For color, we want the visual effect to match the physical position
            # So we use i (physical position) not history_index (data age)
            age_factor = i / len(self.graphs)
            layer_hue = (self.color_hue + age_factor * 0.8) % 1.0
            
            # Create color with high saturation and varying value
            saturation = 0.8 - age_factor * 0.3
            value = 1.0 - age_factor * 0.8
            color = self.hsv_to_rgb(layer_hue, saturation, value)
            graph.set_color(color * 0.8 + Vec3(0.2, 0.2, 0.2))
        
    def setup_controls(self):
        """Setup keyboard controls"""
        self.accept('arrow_up', self.increase_rotation_speed)
        self.accept('arrow_down', self.decrease_rotation_speed)        
        print("\nControls:")
        print("  UP ARROW - Increase rotation speed")
        print("  DOWN ARROW - Decrease rotation speed")

    def increase_rotation_speed(self):
        self.rotation_speed += 5.0
        print(f"Rotation speed: {self.rotation_speed:.1f}°/s")
    
    def decrease_rotation_speed(self):
        self.rotation_speed = max(0.0, self.rotation_speed - 5.0)
        print(f"Rotation speed: {self.rotation_speed:.1f}°/s")

if __name__ == "__main__":
    # Moving tunnel
    app = TestApp(
        smoothing_width     =  1,
        smoothing_sigma     =  1.0,
        amplification       =  50,  # Post-Process amplification
        history_frames      =  100, # Number of itterations
        direction           =  True,# Wave movement direction
        base_rotation       =  0.0, # Starting rotation
        rotation_speed      =  120.0, # Degrees / s
        rotation_per_layer  =  360 / 100, # Rotation per graph layer
        wave_speed          =  1.0, # Wave movement speed
        wave_amplitude      =  0.3, # How far the layers move
        wave_frequency      =  1.0, # How many waves per cycle
        starting_color_hue  =  0.0, # Starting color
        base_color_speed    =  0.001, # How quickly colors cycle
        base_radius         =  2.75, # Base max radius
        base_thickness      =  2.75, # Base graph thickness
    )

    # Flat
    # app = GraphTest(
    #     smoothing_width     =  1,
    #     smoothing_sigma     =  1.0,
    #     amplification       =  50,  # Post-Process amplification
    #     history_frames      =  100, # Number of itterations
    #     direction           =  True,# Wave movement direction
    #     base_rotation       =  0.0, # Starting rotation
    #     rotation_speed      =  8.0, # Degrees / s
    #     rotation_per_layer  =  360 / 100, # Rotation per graph layer
    #     wave_speed          =  0, # Wave movement speed
    #     wave_amplitude      =  0, # How far the layers move
    #     wave_frequency      =  0, # How many waves per cycle
    #     starting_color_hue  =  0.0, # Starting color
    #     base_color_speed    =  0.001, # How quickly colors cycle
    #     base_radius         =  2.75, # Base max radius
    #     base_thickness      =  2.75, # Base graph thickness
    # )

    app.run()