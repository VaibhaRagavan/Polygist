from ffprobe import FFProbe
from langsmith import traceable

@traceable
def get_duration(file):
    metadata=FFProbe(file)
    duration=float(metadata.streams[0].duration)
    return duration