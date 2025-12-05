import imageio_ffmpeg
from pydub import AudioSegment
import os

print(f"FFmpeg path from imageio: {imageio_ffmpeg.get_ffmpeg_exe()}")

# Set converter path if needed
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

try:
    # Create a silent segment
    seg = AudioSegment.silent(duration=1000)
    # Try to export to mp3 (requires ffmpeg)
    seg.export("test.mp3", format="mp3")
    print("Export successful!")
    os.remove("test.mp3")
except Exception as e:
    print(f"Export failed: {e}")
