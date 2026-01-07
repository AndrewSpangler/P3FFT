import numpy as np
from panda3d.core import Vec2, Vec4
from scipy.ndimage import gaussian_filter1d

# Local
from waterfall import WaterfallDisplay
from base import TestBase
from fft import Radix16FFT
from audio_capture import AudioCapture

class TestApp(TestBase):
    def __init__(self):
        TestBase.__init__(
            self,
            Radix16FFT,
            AudioCapture,
            update_callback=self._update_graph,
        )
        self.waterfall = WaterfallDisplay(self, height=512)
        self.num_freq_bins = (self.fft_size // 2) - 1
        self.audio.start()

    def _update_graph(self, complex_result, dt:float):
        magnitudes = np.abs(complex_result[:self.fft_size//2])

        # Convert to dB scale
        magnitudes = 20 * np.log10(magnitudes + 1e-10)

        # Normalize to reasonable
        magnitudes = np.clip(magnitudes, -80, 0)

        # Resample if needed
        width = self.waterfall.waterfall.width
        if len(magnitudes) != width:
            magnitudes = np.interp(
                np.linspace(0, len(magnitudes)-1, width),
                np.arange(len(magnitudes)),
                magnitudes
            )
        self.waterfall.update(magnitudes)

if __name__ == "__main__":
    app = TestApp()
    app.run()