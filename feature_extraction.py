import os
import logging
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from transformers import (
    CLIPProcessor,
    CLIPModel,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    ClapProcessor,
    ClapModel
)
from sentence_transformers import SentenceTransformer

from utils import (
    Shot,
    extract_frames_from_video,
    extract_audio_segment,
    normalize_embedding,
    save_checkpoint,
    load_checkpoint,
    get_video_metadata
)


logger = logging.getLogger("video_chunker")


class ShotDetector:
    def __init__(
        self,
        threshold: float = 27.0,
        min_scene_len: int = 15,
        min_shot_duration: float = 0.5,
        max_shot_duration: float = 30.0
    ):
        self.threshold = threshold
        self.min_scene_len = min_scene_len
        self.min_shot_duration = min_shot_duration
        self.max_shot_duration = max_shot_duration
    
    def detect_shots(self, video_path: str) -> List[Shot]:
        logger.info(f"Detecting shots in {video_path}")
        
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(
                threshold=self.threshold,
                min_scene_len=self.min_scene_len
            )
        )
        
        scene_manager.detect_scenes(video, show_progress=True)
        scene_list = scene_manager.get_scene_list()
        
        if not scene_list:
            metadata = get_video_metadata(video_path)
            return [Shot(shot_id=0, start_time=0.0, end_time=metadata["duration"])]
        
        shots = []
        shot_id = 0
        
        for start_frame, end_frame in scene_list:
            start_time = start_frame.get_seconds()
            end_time = end_frame.get_seconds()
            duration = end_time - start_time
            
            if duration < self.min_shot_duration:
                logger.debug(f"Skipping short shot: {duration:.2f}s")
                continue
            
            if duration > self.max_shot_duration:
                num_splits = int(np.ceil(duration / self.max_shot_duration))
                split_duration = duration / num_splits
                
                for i in range(num_splits):
                    split_start = start_time + i * split_duration
                    split_end = start_time + (i + 1) * split_duration
                    shots.append(Shot(
                        shot_id=shot_id,
                        start_time=split_start,
                        end_time=split_end
                    ))
                    shot_id += 1
            else:
                shots.append(Shot(
                    shot_id=shot_id,
                    start_time=start_time,
                    end_time=end_time
                ))
                shot_id += 1
        
        logger.info(f"Detected {len(shots)} shots")
        return shots


class FeatureExtractor:
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        clip_model_name: str = "openai/clip-vit-large-patch14",
        clap_model_name: str = "laion/clap-htsat-unfused",
        whisper_model_name: str = "openai/whisper-large-v3",
        sentence_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        batch_size: int = 32
    ):
        self.device = device
        self.batch_size = batch_size
        
        logger.info(f"Loading models on device: {device}")
        
        logger.info(f"Loading CLIP model: {clip_model_name}")
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.clip_model.eval()
        
        logger.info(f"Loading CLAP model: {clap_model_name}")
        self.clap_processor = ClapProcessor.from_pretrained(clap_model_name)
        self.clap_model = ClapModel.from_pretrained(clap_model_name).to(device)
        self.clap_model.eval()
        
        logger.info(f"Loading Whisper model: {whisper_model_name}")
        self.whisper_processor = WhisperProcessor.from_pretrained(whisper_model_name)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
            whisper_model_name
        ).to(device)
        self.whisper_model.eval()
        
        logger.info(f"Loading sentence transformer: {sentence_model_name}")
        self.sentence_model = SentenceTransformer(sentence_model_name, device=device)
        
        self.visual_dim = 768
        self.audio_dim = 512
        self.text_dim = 768
    
    def extract_visual_features(
        self,
        video_path: str,
        shots: List[Shot],
        fps: float = 1.0
    ) -> Dict[int, np.ndarray]:
        logger.info("Extracting visual features with CLIP")
        visual_features = {}
        
        for shot in tqdm(shots, desc="Visual features"):
            frames = extract_frames_from_video(
                video_path,
                shot.start_time,
                shot.end_time,
                fps=fps
            )
            
            if not frames:
                visual_features[shot.shot_id] = np.zeros(self.visual_dim)
                continue
            
            all_embeddings = []
            
            for i in range(0, len(frames), self.batch_size):
                batch_frames = frames[i:i + self.batch_size]
                pil_images = [Image.fromarray(f) for f in batch_frames]
                
                inputs = self.clip_processor(
                    images=pil_images,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.clip_model.get_image_features(**inputs)
                    if hasattr(outputs, "last_hidden_state") or hasattr(outputs, "pooler_output"):
                        # This shouldn't happen with get_image_features usually, but handle just in case
                        image_features = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs[0]
                    else:
                        image_features = outputs
                
                all_embeddings.append(image_features.cpu().numpy())
            
            if all_embeddings:
                combined = np.vstack(all_embeddings)
                avg_embedding = np.mean(combined, axis=0)
                visual_features[shot.shot_id] = normalize_embedding(avg_embedding)
            else:
                visual_features[shot.shot_id] = np.zeros(self.visual_dim)
        
        return visual_features
    
    def extract_audio_features(
        self,
        video_path: str,
        shots: List[Shot]
    ) -> Dict[int, np.ndarray]:
        logger.info("Extracting audio features with CLAP")
        audio_features = {}
        
        for shot in tqdm(shots, desc="Audio features"):
            try:
                audio_path = extract_audio_segment(
                    video_path,
                    shot.start_time,
                    shot.end_time
                )
                
                if os.path.getsize(audio_path) == 0:
                    audio_features[shot.shot_id] = np.zeros(self.audio_dim)
                    os.remove(audio_path)
                    continue
                
                import librosa
                waveform, sr = librosa.load(audio_path, sr=48000)
                
                if len(waveform) == 0 or np.max(np.abs(waveform)) < 1e-6:
                    audio_features[shot.shot_id] = np.zeros(self.audio_dim)
                    os.remove(audio_path)
                    continue
                
                inputs = self.clap_processor(
                    audios=waveform,
                    sampling_rate=sr,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.clap_model.get_audio_features(**inputs)
                    if hasattr(outputs, "pooler_output"):
                        audio_emb = outputs.pooler_output
                    elif hasattr(outputs, "last_hidden_state"):
                        audio_emb = outputs.last_hidden_state
                    else:
                        audio_emb = outputs
                
                audio_features[shot.shot_id] = normalize_embedding(
                    audio_emb.cpu().numpy().squeeze()
                )
                
                os.remove(audio_path)
                
            except Exception as e:
                logger.warning(f"Audio extraction failed for shot {shot.shot_id}: {e}")
                audio_features[shot.shot_id] = np.zeros(self.audio_dim)
        
        return audio_features
    
    def extract_text_features(
        self,
        video_path: str,
        shots: List[Shot]
    ) -> Tuple[Dict[int, str], Dict[int, np.ndarray]]:
        logger.info("Extracting text features with Whisper + Sentence Transformer")
        transcripts = {}
        text_embeddings = {}
        
        for shot in tqdm(shots, desc="Text features"):
            try:
                audio_path = extract_audio_segment(
                    video_path,
                    shot.start_time,
                    shot.end_time
                )
                
                if os.path.getsize(audio_path) == 0:
                    transcripts[shot.shot_id] = ""
                    text_embeddings[shot.shot_id] = np.zeros(self.text_dim)
                    os.remove(audio_path)
                    continue
                
                import librosa
                waveform, sr = librosa.load(audio_path, sr=16000)
                
                if len(waveform) == 0:
                    transcripts[shot.shot_id] = ""
                    text_embeddings[shot.shot_id] = np.zeros(self.text_dim)
                    os.remove(audio_path)
                    continue
                
                input_features = self.whisper_processor(
                    waveform,
                    sampling_rate=16000,
                    return_tensors="pt"
                ).input_features.to(self.device)
                
                with torch.no_grad():
                    generated_ids = self.whisper_model.generate(input_features)
                
                transcript = self.whisper_processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0].strip()
                
                transcripts[shot.shot_id] = transcript
                
                if transcript:
                    text_emb = self.sentence_model.encode(
                        transcript,
                        convert_to_numpy=True
                    )
                    text_embeddings[shot.shot_id] = normalize_embedding(text_emb)
                else:
                    text_embeddings[shot.shot_id] = np.zeros(self.text_dim)
                
                os.remove(audio_path)
                
            except Exception as e:
                logger.warning(f"Text extraction failed for shot {shot.shot_id}: {e}")
                transcripts[shot.shot_id] = ""
                text_embeddings[shot.shot_id] = np.zeros(self.text_dim)
        
        return transcripts, text_embeddings
    
    def fuse_embeddings(
        self,
        visual: np.ndarray,
        audio: np.ndarray,
        text: np.ndarray
    ) -> np.ndarray:
        visual_proj = visual[:512] if len(visual) > 512 else np.pad(
            visual, (0, 512 - len(visual))
        )
        
        audio_proj = audio[:512] if len(audio) >= 512 else np.pad(
            audio, (0, 512 - len(audio))
        )
        
        fused = np.concatenate([visual_proj, audio_proj, text])
        return normalize_embedding(fused)
    
    def extract_all_features(
        self,
        video_path: str,
        shots: List[Shot],
        cache_dir: Optional[str] = None,
        checkpoint_interval: int = 100,
        skip_audio: bool = False
    ) -> List[Shot]:
        logger.info(f"Extracting features for {len(shots)} shots")
        
        cache_path = None
        if cache_dir:
            cache_path = os.path.join(cache_dir, "embeddings_cache.pkl")
            cached = load_checkpoint(cache_path)
            if cached:
                logger.info("Found cached embeddings, skipping extraction")
                for shot in shots:
                    if shot.shot_id in cached.get("visual", {}):
                        shot.visual_embedding = cached["visual"][shot.shot_id].tolist()
                        shot.audio_embedding = cached["audio"][shot.shot_id].tolist()
                        shot.text_embedding = cached["text"][shot.shot_id].tolist()
                        shot.transcript = cached["transcripts"].get(shot.shot_id, "")
                        
                        fused = self.fuse_embeddings(
                            cached["visual"][shot.shot_id],
                            cached["audio"][shot.shot_id],
                            cached["text"][shot.shot_id]
                        )
                        shot.fused_embedding = fused.tolist()
                return shots
        
        visual_features = self.extract_visual_features(video_path, shots)
        
        if skip_audio:
            audio_features = {s.shot_id: np.zeros(self.audio_dim) for s in shots}
            transcripts = {s.shot_id: "" for s in shots}
            text_embeddings = {s.shot_id: np.zeros(self.text_dim) for s in shots}
        else:
            audio_features = self.extract_audio_features(video_path, shots)
            transcripts, text_embeddings = self.extract_text_features(video_path, shots)
        
        for shot in shots:
            shot.visual_embedding = visual_features[shot.shot_id].tolist()
            shot.audio_embedding = audio_features[shot.shot_id].tolist()
            shot.text_embedding = text_embeddings[shot.shot_id].tolist()
            shot.transcript = transcripts.get(shot.shot_id, "")
            
            fused = self.fuse_embeddings(
                visual_features[shot.shot_id],
                audio_features[shot.shot_id],
                text_embeddings[shot.shot_id]
            )
            shot.fused_embedding = fused.tolist()
        
        if cache_path:
            cache_data = {
                "visual": visual_features,
                "audio": audio_features,
                "text": text_embeddings,
                "transcripts": transcripts
            }
            save_checkpoint(cache_data, cache_path)
            logger.info(f"Saved embeddings cache to {cache_path}")
        
        return shots
