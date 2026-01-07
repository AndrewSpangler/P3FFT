import numpy as np

IMPORT_ERROR_MESSAGE = """
Install pyaudio (cross-platform, input only)
or pyaudiowpatch (Windows only, automatic loopback)
""".strip()
try: # Windows patched Pyaudio (automatic loopback)
    import pyaudiowpatch as pyaudio
    PATCHED_AUDIO = True
except: # Other OS fallback (inputs only)
    PATCHED_AUDIO = False
    try:
        import pyaudio as pyaudio
    except:
        raise ImportError(IMPORT_ERROR_MESSAGE)

LOOPBACK_KEYWORDS = [
    'stereo mix',   'wave out mix',    'what u hear',   # Windows
    'blackhole',    'soundflower',                      # MacOS
    'monitor',      'loopback',                         # Linux
]

def get_available_devices(p:pyaudio.PyAudio) -> list[dict[str:any]]:
    """Get available pyaudio devices"""
    devices = {}
    try:
        get_dev_info = p.get_device_info_by_host_api_device_index
        info = p.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        devices = {i : get_dev_info(0, i) for i in range(num_devices)}
    except Exception as e:
        raise(f"Error getting audio devices: {e}")
    return devices

def get_default_loopback(p:pyaudio.PyAudio) -> dict[str:any]:
    """Get pyaudiowpatch default loopback device"""
    if not PATCHED_AUDIO:
        return
    try:
        return p.get_default_wasapi_loopback()
    except (OSError, AttributeError): # Redundant, idc :shrug:
        pass

def find_loopback_device(p:pyaudio.PyAudio) -> dict[str:any]:
    """Attempt to find a loopback device"""
    if (device_info := get_default_loopback(p)):
        return device_info
    candidates = []
    for idx, device_info in get_available_devices(p).items():
        device_name = device_info.get('name', '').lower()
        if device_info.get('maxInputChannels', 0) > 0:
            for keyword in LOOPBACK_KEYWORDS:
                if keyword in device_name:
                    candidates.append(device_info)
                    break
    return candidates[0] if candidates else None

def enumerate_devices(p:pyaudio.PyAudio):
    """Print audio devices for debug / user interaction"""
    for idx, device_info in get_available_devices(p).items():
        name = device_info.get('name', 'Unknown')
        max_input = device_info.get('maxInputChannels', 0)
        max_output = device_info.get('maxOutputChannels', 0)
        device_type = []
        if max_input > 0: device_type.append(f"IN:{max_input}")
        if max_output > 0: device_type.append(f"OUT:{max_output}")
        channels_info = " [" + ", ".join(device_type) + "]"
        print(f"  {idx}: {name}{channels_info}")

def check_power_16(fft_size:int) -> bool:
    """Checks if a size is an exponent of 16"""
    return (np.log(fft_size) / np.log(16)).is_integer()

class AudioClass(pyaudio.PyAudio):
    """Class to wrap around both patched and unpatched pyaudio"""
    def __init__(self, *args, **kw):
        pyaudio.PyAudio.__init__(self, *args, **kw)
        self.get_available_devices  = lambda: get_available_devices(self)
        self.get_default_loopback   = lambda: get_default_loopback(self)
        self.find_loopback_device   = lambda: find_loopback_device(self)
        self.enumerate_devices      = lambda: enumerate_devices(self)