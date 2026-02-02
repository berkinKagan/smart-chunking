import os
import cv2
import pickle
import logging
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Any, Dict
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json


@dataclass
class Shot:
    shot_id: int
    start_time: float
    end_time: float
    visual_embedding: Optional[List[float]] = None
    audio_embedding: Optional[List[float]] = None
    text_embedding: Optional[List[float]] = None
    transcript: str = ""
    fused_embedding: Optional[List[float]] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> Dict:
        return {
            "shot_id": self.shot_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "visual_embedding": self.visual_embedding,
            "audio_embedding": self.audio_embedding,
            "text_embedding": self.text_embedding,
            "transcript": self.transcript
        }


@dataclass
class Chunk:
    chunk_id: str
    level: str
    start_time: float
    end_time: float
    shot_ids: List[int]
    caption: str = ""
    transcript: str = ""
    embedding: Optional[List[float]] = None
    parent_chunk_id: Optional[str] = None
    reasoning: str = ""

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def num_shots(self) -> int:
        return len(self.shot_ids)

    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "level": self.level,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "num_shots": self.num_shots,
            "shot_ids": self.shot_ids,
            "caption": self.caption,
            "transcript": self.transcript,
            "embedding": self.embedding,
            "parent_chunk_id": self.parent_chunk_id,
            "reasoning": self.reasoning,
            "child_shot_ids": self.shot_ids
        }


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("video_chunker")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_video_metadata(video_path: str) -> Dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    return {
        "video_path": video_path,
        "duration": duration,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "processed_date": datetime.now().isoformat()
    }


def extract_frames_from_video(
    video_path: str,
    start_time: float,
    end_time: float,
    fps: float = 1.0
) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * video_fps)
    end_frame = int(end_time * video_fps)
    frame_interval = int(video_fps / fps) if fps > 0 else 1
    
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    current_frame = start_frame
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        if (current_frame - start_frame) % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        current_frame += 1
    
    cap.release()
    return frames


def extract_keyframes(
    video_path: str,
    start_time: float,
    end_time: float,
    num_frames: int
) -> List[np.ndarray]:
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


def extract_audio_segment(
    video_path: str,
    start_time: float,
    end_time: float,
    output_path: Optional[str] = None,
    sample_rate: int = 48000
) -> str:
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
    
    duration = end_time - start_time
    
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(duration),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        output_path
    ]
    
    try:
        subprocess.run(
            cmd,
            capture_output=True,
            check=True
        )
    except subprocess.CalledProcessError:
        with open(output_path, "wb") as f:
            pass
    
    return output_path


def save_checkpoint(
    data: Any,
    checkpoint_path: str
) -> None:
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "wb") as f:
        pickle.dump(data, f)


def load_checkpoint(checkpoint_path: str) -> Optional[Any]:
    if not os.path.exists(checkpoint_path):
        return None
    with open(checkpoint_path, "rb") as f:
        return pickle.load(f)


def save_embeddings_cache(
    embeddings: Dict[int, Dict[str, List[float]]],
    cache_path: str
) -> None:
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(embeddings, f)


def load_embeddings_cache(cache_path: str) -> Optional[Dict[int, Dict[str, List[float]]]]:
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embedding)
    if norm > 0:
        return embedding / norm
    return embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = normalize_embedding(a)
    b_norm = normalize_embedding(b)
    return float(np.dot(a_norm, b_norm))


def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
    return f"{minutes:02d}:{secs:05.2f}"


def validate_output_schema(output: Dict) -> List[str]:
    errors = []
    
    required_fields = ["video_metadata", "shots", "chunks", "hierarchy"]
    for field in required_fields:
        if field not in output:
            errors.append(f"Missing required field: {field}")
    
    if "video_metadata" in output:
        meta_fields = ["video_path", "duration", "total_shots", "processed_date"]
        for field in meta_fields:
            if field not in output["video_metadata"]:
                errors.append(f"Missing metadata field: {field}")
    
    if "chunks" in output:
        for level in ["fine", "medium", "coarse"]:
            if level not in output["chunks"]:
                errors.append(f"Missing chunk level: {level}")
    
    if "hierarchy" in output:
        if "fine_to_medium" not in output["hierarchy"]:
            errors.append("Missing hierarchy mapping: fine_to_medium")
        if "medium_to_coarse" not in output["hierarchy"]:
            errors.append("Missing hierarchy mapping: medium_to_coarse")
    
    return errors


def compute_quality_metrics(
    shots: List[Shot],
    chunks: Dict[str, List[Chunk]]
) -> Dict:
    metrics = {}
    
    for level, level_chunks in chunks.items():
        if not level_chunks:
            continue
        
        durations = [c.duration for c in level_chunks]
        metrics[f"{level}_duration_min"] = min(durations)
        metrics[f"{level}_duration_max"] = max(durations)
        metrics[f"{level}_duration_mean"] = np.mean(durations)
        metrics[f"{level}_duration_std"] = np.std(durations)
        metrics[f"{level}_num_chunks"] = len(level_chunks)
        
        shot_counts = [c.num_shots for c in level_chunks]
        metrics[f"{level}_shots_per_chunk_mean"] = np.mean(shot_counts)
    
    shot_embeddings = {}
    for shot in shots:
        if shot.fused_embedding is not None:
            shot_embeddings[shot.shot_id] = np.array(shot.fused_embedding)
    
    for level, level_chunks in chunks.items():
        intra_sims = []
        for chunk in level_chunks:
            chunk_embs = [
                shot_embeddings[sid] 
                for sid in chunk.shot_ids 
                if sid in shot_embeddings
            ]
            if len(chunk_embs) >= 2:
                for i in range(len(chunk_embs)):
                    for j in range(i + 1, len(chunk_embs)):
                        sim = cosine_similarity(chunk_embs[i], chunk_embs[j])
                        intra_sims.append(sim)
        
        if intra_sims:
            metrics[f"{level}_intra_coherence"] = np.mean(intra_sims)
        
        inter_sims = []
        for i in range(len(level_chunks) - 1):
            if level_chunks[i].embedding and level_chunks[i + 1].embedding:
                sim = cosine_similarity(
                    np.array(level_chunks[i].embedding),
                    np.array(level_chunks[i + 1].embedding)
                )
                inter_sims.append(sim)
        
        if inter_sims:
            metrics[f"{level}_inter_diversity"] = 1 - np.mean(inter_sims)
    
    return metrics


def check_hierarchy_consistency(
    chunks: Dict[str, List[Chunk]],
    hierarchy: Dict[str, Dict[str, str]]
) -> List[str]:
    errors = []
    
    fine_chunks = {c.chunk_id for c in chunks.get("fine", [])}
    medium_chunks = {c.chunk_id for c in chunks.get("medium", [])}
    coarse_chunks = {c.chunk_id for c in chunks.get("coarse", [])}
    
    fine_to_medium = hierarchy.get("fine_to_medium", {})
    for fine_id, medium_id in fine_to_medium.items():
        if fine_id not in fine_chunks:
            errors.append(f"Fine chunk {fine_id} in hierarchy not found in chunks")
        if medium_id not in medium_chunks:
            errors.append(f"Medium chunk {medium_id} in hierarchy not found in chunks")
    
    for fine_id in fine_chunks:
        if fine_id not in fine_to_medium:
            errors.append(f"Fine chunk {fine_id} has no parent medium chunk")
    
    medium_to_coarse = hierarchy.get("medium_to_coarse", {})
    for medium_id, coarse_id in medium_to_coarse.items():
        if medium_id not in medium_chunks:
            errors.append(f"Medium chunk {medium_id} in hierarchy not found in chunks")
        if coarse_id not in coarse_chunks:
            errors.append(f"Coarse chunk {coarse_id} in hierarchy not found in chunks")
    
    for medium_id in medium_chunks:
        if medium_id not in medium_to_coarse:
            errors.append(f"Medium chunk {medium_id} has no parent coarse chunk")
    
    return errors
