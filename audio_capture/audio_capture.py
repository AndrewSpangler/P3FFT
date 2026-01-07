import numpy as np
import queue
from collections import deque
import threading

# Imports the patched version of pyaudio if available
from .audio_helpers import (
    pyaudio,
    AudioClass,
    check_power_16
)

class AudioCapture:
    def __init__(self, fft_size=4096, frames_per_buffer=1024):
        self.fft_size = fft_size
        self.frames_per_buffer = frames_per_buffer

        self.sample_rate = None
        self.channels = None
        
        # Verify FFT size is power of 16
        log16 = np.log(fft_size) / np.log(16)
        if not check_power_16(fft_size):
            print(f"Warning: FFT size {fft_size} is not a power of 16")
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=fft_size * 4)
        self.lock = threading.Lock()
        
        # FFT window
        self.window = np.hanning(self.fft_size)
        
        # Audio queue for real-time processing
        self.audio_queue = queue.Queue()
        
        # PyAudio instance and stream
        self.p = AudioClass() # Custom PyAudio with QoL additions
        self.stream = None
        self.is_running = False
        
        # List available devices
        self.p.enumerate_devices()
            
    def _handle_stream(self, in_data, frame_count, time_info, status):
        """Callback for PyAudio stream"""
        if status: print(f"Audio status: {status}")
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        if self.channels == 2: # Convert to mono if stereo
            audio_data = audio_data.reshape(-1, 2).mean(axis=1)
        self.audio_queue.put(audio_data.copy()) # Add to queue
        return (in_data, pyaudio.paContinue)
    
    def start(self, device=None):
        if self.is_running:
            raise RuntimeError("Capture already running")
        
        try:
            # If no device specified, find loopback device
            if device is None:
                print("\nSearching for loopback devices...")
                if (loopback_device := self.p.find_loopback_device()):
                    device = loopback_device['index']
                    self.sample_rate = int(loopback_device['defaultSampleRate'])
                    self.channels = loopback_device['maxInputChannels']
                    print(f"Found loopback device: {loopback_device['name']}")
                    print(f"  Sample Rate: {self.sample_rate} Hz")
                    print(f"  Channels: {self.channels}")
                else:
                    print("No loopback devices found")
                    print("\nTo capture desktop audio, you need to:")
                    print("  • Windows: Install patched pyaudio (pip install pyaudiowpatch)")
                    print("  • macOS: Install BlackHole (brew install blackhole-2ch)")
                    print("  • Linux: Use PulseAudio monitor")
                    
                    # Fall back to default input device
                    default_device = self.p.get_default_input_device_info()
                    if default_device:
                        device = default_device['index']
                        print(f"\nFalling back to default input device: {default_device['name']}")
            
            # Open audio stream
            device_info = self.p.get_device_info_by_index(device) if isinstance(device, int) else device
            
            self.stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_info['index'] if isinstance(device_info, dict) else device,
                frames_per_buffer=self.frames_per_buffer,
                stream_callback=self._handle_stream
            )
            
            self.stream.start_stream()
            self.is_running = True

            print("Audio capture started.")
            
        except Exception as e:
            print(f"\nError starting audio capture: {e}")
            print("\nPlease check:")
            print("  1. The device supports the requested format")
            print("  2. You have permission to access audio devices")
            print("  3. No other application is using the device")
            raise
    
    def stop(self):
        """Stop audio capture"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.is_running = False
        if self.p:
            self.p.terminate()
        print("\nAudio capture stopped.")