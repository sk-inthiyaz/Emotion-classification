import os
import sys

try:
    import imageio_ffmpeg
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"imageio_ffmpeg path: {ffmpeg_path}")
    print(f"Exists: {os.path.exists(ffmpeg_path)}")
except ImportError:
    print("imageio_ffmpeg not installed")

try:
    from pydub import AudioSegment
    print(f"Default AudioSegment.converter: {AudioSegment.converter}")
    
    if 'imageio_ffmpeg' in sys.modules:
        AudioSegment.converter = ffmpeg_path
        print(f"Set AudioSegment.converter to: {AudioSegment.converter}")

    # Create a dummy silent audio segment to test export (which uses ffmpeg)
    # Note: creating silence doesn't use ffmpeg, but exporting might if we convert format
    # But better to try to read something or just check if pydub can call it.
    
    from pydub.utils import which
    print(f"pydub.utils.which('ffmpeg'): {which('ffmpeg')}")
    
except ImportError:
    print("pydub not installed")
