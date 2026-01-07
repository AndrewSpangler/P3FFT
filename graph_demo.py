import numpy as np
from panda3d.core import Vec2, Vec4
from scipy.ndimage import gaussian_filter1d

# Local
from graphing import Graph
from base import TestBase
from fft import Radix16FFT
from audio_capture import AudioCapture

class TestApp(TestBase):
    def __init__(
        self,
        smoothing_width:int=4,
        smoothing_sigma:float=4.0,
        amplification:float=50,
        circular=False
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
        self.circular = circular
        self.num_freq_bins = self.fft_size // 2
        self.graph = Graph(
            parent=self.aspect2d,
            min_val=0.0,
            max_val=self.fft_size // 2,
            max_values=self.num_freq_bins,
            line_color=Vec4(0.2, 0.8, 0.2, 1.0),
            bg_color=Vec4(0.1, 0.1, 0.2, 0.9),
            position=Vec2(0, 0),
            size=Vec2(2.0, 2.0),
            thickness=6,
            circular=False,
        )
        self.audio.start()

    def _update_graph(self, complex_result, dt:float):
        magnitudes = np.abs(complex_result[:self.fft_size//2])
        total_volume = np.sum(magnitudes)
        if self.smoothing_width:
            if self.circular:
                magnitudes_circular = np.concatenate([
                    magnitudes[-self.smoothing_window:],
                    magnitudes,
                    magnitudes[:self.smoothing_window]
                ])
                smoothed_circular = gaussian_filter1d(magnitudes_circular, sigma=self.smoothing_sigma)
                magnitudes = smoothed_circular[smoothing_window:-smoothing_window]
                values_circular = np.concatenate([magnitudes, [magnitudes[0]]])
            else:
                magnitudes = gaussian_filter1d(magnitudes, sigma=self.smoothing_sigma)
        
        magnitudes *= self.amplification
        self.graph.set_values(magnitudes)

if __name__ == "__main__":
    app = TestApp()
    app.run()