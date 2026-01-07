import sys
import queue
import threading
import numpy as np
from direct.showbase.ShowBase import ShowBase

class TestBase(ShowBase):
    def __init__(
        self,
        fft_engine,
        audio_capture,
        fft_size:int=4096,
        update_callback:callable=None
    ):
        ShowBase.__init__(self)

        self.fft_size = fft_size
        self.update_callback = update_callback

        # Setup window
        self.setBackgroundColor(0.1, 0.1, 0.15, 1.0) # Dark blue

        # Setup camera
        self.cam.setPos(0, -5, 0)
        self.cam.lookAt(0, 10, 0)

        # Performance monitoring
        self.last_fps_time = 0
        self.fps_frames = 0
        self.last_time = 0
        self.frame_count = 0

        self.disableMouse()

        # Set up audio manager
        self.audio = audio_capture(fft_size=fft_size)
        self.audio_buffer = np.zeros(fft_size * 2, dtype=np.float32)
        self.buffer_lock = threading.Lock()
        self.fft_engine = fft_engine(base)
        self.fft_window = np.hanning(self.fft_size)

        self.taskMgr.add(self._update_task, "update_task")
        
    def _update_task(self, task):
        """Update audio buffer"""
        self.frame_count += 1

        current_time = task.time
        if self.last_time == 0.0:
            self.last_time = current_time
        dt = current_time - self.last_time
        self.last_time = current_time

        try:
            # Collect all available audio chunks
            audio_chunks = []
            while not self.audio.audio_queue.empty():
                chunk = self.audio.audio_queue.get_nowait()
                audio_chunks.append(chunk)
            
            if audio_chunks:
                # Concatenate and update buffer
                new_audio = np.concatenate(audio_chunks)
                with self.buffer_lock:
                    # Shift buffer and add new data
                    shift_amount = len(new_audio)
                    self.audio_buffer[:-shift_amount] = self.audio_buffer[shift_amount:]
                    self.audio_buffer[-shift_amount:] = new_audio
        except queue.Empty:
            pass
        except ValueError:
            pass

        with self.buffer_lock:
            signal = self.audio_buffer[-self.fft_size:].copy()

        # Apply window function
        signal = signal * self.fft_window
        
        # GPU FFT Processing
        gpu_handle = self.fft_engine.fft(signal.astype(np.complex64))
        complex_result = self.fft_engine.fetch(gpu_handle)

        if self.update_callback:
            self.update_callback(complex_result, dt)

        return task.cont

    def cleanup(self):
        sys.exit(0)

    def start(self, device):
        self.audio.start(device=device)