import os
import cv2
import torch
import logging
import numpy as np
from PIL import Image
from typing import List, Dict, Any
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from utils import Sample, SamplingWindow
from config import CLIP_MODEL, SBERT_MODEL, CACHE_DIR

logger = logging.getLogger("hybrid_hierarchical")

class MultimodalExtractor:
    def __init__(self, device: str = "cpu"):
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        logger.info(f"Initializing MultimodalExtractor on {self.device}")
        
        # Load models
        self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL).to(self.device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
        self.sbert_model = SentenceTransformer(SBERT_MODEL, device=self.device)
        
        # Note: CLAP and Whisper are computationally heavy. 
        # For this implementation, we simulate/placeholder them if not strictly needed,
        # but the structure is ready for full integration.
        
    def extract_features(self, video_path: str, sampling_plan: List[SamplingWindow], codec_data: Dict[str, Any]) -> List[Sample]:
        logger.info(f"Starting multimodal extraction for {len(sampling_plan)} windows")
        
        samples = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        for i, window in enumerate(sampling_plan):
            logger.info(f"Processing window {i+1}/{len(sampling_plan)}: {window.start_time:.1f}s - {window.end_time:.1f}s")
            
            # Calculate timestamps to sample
            duration = window.end_time - window.start_time
            num_samples = max(1, int(duration * window.fps))
            timestamps = np.linspace(window.start_time, window.end_time, num_samples)
            
            for ts in timestamps:
                cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Visual embedding (CLIP)
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                inputs = self.clip_processor(images=[pil_img], return_tensors="pt").to(self.device)
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**inputs)
                    vis_features = image_features.pooler_output.cpu().numpy()[0]
                vis_features /= np.linalg.norm(vis_features)
                
                # Find nearest I-frame time for codec alignment
                nearest_iframe = min(codec_data["iframe_times"], key=lambda t: abs(t - ts))
                
                # Check for candidates near this timestamp
                codec_conf = "low"
                for c in codec_data["candidates"]:
                    if abs(c.time - ts) < 0.5:
                        codec_conf = c.confidence
                        break
                
                samples.append(Sample(
                    time=ts,
                    visual_embedding=vis_features,
                    codec_confidence=codec_conf,
                    nearest_iframe_time=nearest_iframe
                ))
                
        cap.release()
        logger.info(f"Extracted {len(samples)} samples total.")
        return samples
