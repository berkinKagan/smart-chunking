from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np

@dataclass
class GOPInfo:
    gop_id: int
    start_time: float
    end_time: float
    avg_packet_size: float
    packet_size_variance: float
    motion_magnitude: float
    iframe_pts: float
    iframe_size: int

@dataclass
class BoundaryCandidate:
    time: float
    confidence: str  # 'high', 'medium', 'low'
    packet_size_jump: float
    reason: str

@dataclass
class SamplingWindow:
    start_time: float
    end_time: float
    fps: float
    priority: str  # 'high', 'medium', 'low'

@dataclass
class Sample:
    time: float
    visual_embedding: Optional[np.ndarray] = None
    audio_embedding: Optional[np.ndarray] = None
    text_embedding: Optional[np.ndarray] = None
    transcript: str = ""
    codec_confidence: str = "low"
    nearest_iframe_time: float = 0.0
    
    @property
    def multimodal_embedding(self) -> np.ndarray:
        # Concatenate 512 (CLIP) + 512 (CLAP) + 768 (SBERT/Whisper) = 1792
        v = self.visual_embedding if self.visual_embedding is not None else np.zeros(512)
        a = self.audio_embedding if self.audio_embedding is not None else np.zeros(512)
        t = self.text_embedding if self.text_embedding is not None else np.zeros(768)
        return np.concatenate([v, a, t])

@dataclass
class Chunk:
    chunk_id: str
    level: str  # 'fine', 'medium', 'coarse'
    start_time: float
    end_time: float
    sample_indices: List[int] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    caption: str = ""
    transcript: str = ""
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = field(default_factory=list)
    codec_aligned: bool = False
    confidence_score: float = 0.0

    @property
    def duration(self) -> float:
        """Calculate duration from end_time - start_time"""
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "level": self.level,
            "start": self.start_time,
            "end": self.end_time,
            "caption": self.caption,
            "transcript": self.transcript,
            "parent_id": self.parent_chunk_id,
            "codec_aligned": self.codec_aligned,
            "confidence_score": self.confidence_score
        }

@dataclass
class Shot:
    shot_id: int
    start_time: float
    end_time: float
    keyframe_path: Optional[str] = None
    transcript: Optional[str] = None
    fused_embedding: Optional[np.ndarray] = None

def extract_keyframes(
    video_path: str,
    start_time: float,
    end_time: float,
    num_frames: int
) -> List[np.ndarray]:
    import cv2
    duration = end_time - start_time
    if duration <= 0 or num_frames <= 0:
        return []
    
    timestamps = np.linspace(start_time, end_time, num_frames, endpoint=False)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    
    for ts in timestamps:
        frame_num = int(ts * video_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    cap.release()
    return frames

def extract_iframes_for_segment(
    video_path: str,
    start_time: float,
    end_time: float,
    max_frames: int = 3
) -> List[np.ndarray]:
    import subprocess
    import tempfile
    import os
    from PIL import Image
    duration = end_time - start_time
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_pattern = os.path.join(tmpdir, "frame_%04d.jpg")
        
        cmd = [
            "ffmpeg",
            "-ss", str(start_time),
            "-i", video_path,
            "-t", str(duration),
            "-vf", f"select='eq(pict_type,I)',scale=224:224",
            "-vsync", "vfr",
            "-frames:v", str(max_frames),
            "-q:v", "2",
            output_pattern,
            "-y"
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=30)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return []
        
        frames = []
        for f in sorted(os.listdir(tmpdir)):
            if f.endswith(".jpg"):
                img = Image.open(os.path.join(tmpdir, f))
                frames.append(np.array(img))
        
        return frames
