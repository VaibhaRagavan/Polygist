import tempfile
import os
from ffprobe import FFProbe
from langsmith import traceable

@traceable
def get_duration(file):
    ext = file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    try:
        metadata = FFProbe(tmp_path)
        for stream in metadata.streams:
            if stream.is_video() or stream.is_audio():
                print(stream.duration)
                return float(stream.duration)
    finally:
        os.remove(tmp_path)
        file.seek(0)