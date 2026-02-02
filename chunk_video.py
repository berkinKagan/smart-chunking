#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from utils import (
    setup_logging,
    get_video_metadata,
    save_checkpoint,
    load_checkpoint,
    validate_output_schema,
    compute_quality_metrics,
    check_hierarchy_consistency,
    Shot,
    Chunk
)
from feature_extraction import ShotDetector, FeatureExtractor
from clustering import HierarchicalChunker
from caption_generation import CaptionGenerator, BoundaryRefiner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Context-preserving video chunking pipeline for long-form videos"
    )
    
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to the input video file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output files"
    )
    parser.add_argument(
        "--temporal_weight",
        type=float,
        default=0.3,
        help="Weight for temporal distance in clustering (default: 0.3)"
    )
    parser.add_argument(
        "--fine_duration",
        type=float,
        default=30.0,
        help="Target duration for fine-level chunks in seconds (default: 30)"
    )
    parser.add_argument(
        "--medium_duration",
        type=float,
        default=120.0,
        help="Target duration for medium-level chunks in seconds (default: 120)"
    )
    parser.add_argument(
        "--coarse_duration",
        type=float,
        default=300.0,
        help="Target duration for coarse-level chunks in seconds (default: 300)"
    )
    parser.add_argument(
        "--use_boundary_refinement",
        action="store_true",
        help="Use LLM-based boundary refinement"
    )
    parser.add_argument(
        "--generate_captions",
        action="store_true",
        help="Generate VLM-based captions for chunks"
    )
    parser.add_argument(
        "--skip_audio",
        action="store_true",
        help="Skip audio feature extraction (for videos without audio)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for inference (cuda/cpu, default: cuda)"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint file to resume from"
    )
    parser.add_argument(
        "--caption_provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic", "ollama"],
        help="LLM provider for captions (default: openai)"
    )
    parser.add_argument(
        "--caption_model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use for caption generation (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--shot_threshold",
        type=float,
        default=27.0,
        help="Threshold for shot detection (default: 27.0)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for feature extraction (default: 32)"
    )
    
    return parser.parse_args()


def run_pipeline(
    video_path: str,
    output_dir: str,
    temporal_weight: float = 0.3,
    fine_duration: float = 30.0,
    medium_duration: float = 120.0,
    coarse_duration: float = 300.0,
    use_boundary_refinement: bool = False,
    generate_captions: bool = False,
    skip_audio: bool = False,
    device: str = "cuda",
    resume_from: Optional[str] = None,
    caption_provider: str = "openai",
    caption_model: str = "gpt-4o-mini",
    reasoning_model: Optional[str] = None,
    ignore_checkpoint: bool = False,
    log_level: str = "INFO",
    shot_threshold: float = 27.0,
    batch_size: int = 32
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    log_file = os.path.join(output_dir, "chunking.log")
    logger = setup_logging(log_level, log_file)
    
    logger.info("=" * 60)
    logger.info("Video Chunking Pipeline Started")
    logger.info("=" * 60)
    logger.info(f"Video: {video_path}")
    logger.info(f"Output directory: {output_dir}")
    
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return None
    
    checkpoint_path = os.path.join(output_dir, "checkpoint.pkl")
    checkpoint_data = {}
    
    if ignore_checkpoint:
        logger.info("Ignoring existing checkpoint as requested (fresh run)")
    elif resume_from:
        checkpoint_data = load_checkpoint(resume_from) or {}
        logger.info(f"Resumed from checkpoint: {resume_from}")
    elif os.path.exists(checkpoint_path):
        checkpoint_data = load_checkpoint(checkpoint_path) or {}
        logger.info("Found existing checkpoint, resuming")
    
    logger.info("Step 1: Getting video metadata")
    try:
        video_metadata = get_video_metadata(video_path)
        logger.info(f"Video duration: {video_metadata['duration']:.2f}s")
        logger.info(f"Video resolution: {video_metadata['width']}x{video_metadata['height']}")
    except Exception as e:
        logger.error(f"Failed to get video metadata: {e}")
        return None
    
    logger.info("Step 2: Shot detection")
    if "shots" in checkpoint_data:
        shots = checkpoint_data["shots"]
        logger.info(f"Loaded {len(shots)} shots from checkpoint")
    else:
        shot_detector = ShotDetector(
            threshold=shot_threshold,
            min_shot_duration=0.5,
            max_shot_duration=30.0
        )
        shots = shot_detector.detect_shots(video_path)
        checkpoint_data["shots"] = shots
        save_checkpoint(checkpoint_data, checkpoint_path)
        logger.info(f"Detected {len(shots)} shots, checkpoint saved")
    
    logger.info("Step 3: Feature extraction")
    if "features_extracted" in checkpoint_data and checkpoint_data["features_extracted"]:
        logger.info("Features already extracted, loading from checkpoint")
        shots = checkpoint_data["shots"]
    else:
        import torch
        actual_device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        logger.info(f"Using device: {actual_device}")
        
        feature_extractor = FeatureExtractor(
            device=actual_device,
            batch_size=batch_size
        )
        
        shots = feature_extractor.extract_all_features(
            video_path,
            shots,
            cache_dir=output_dir,
            skip_audio=skip_audio
        )
        
        checkpoint_data["shots"] = shots
        checkpoint_data["features_extracted"] = True
        save_checkpoint(checkpoint_data, checkpoint_path)
        logger.info("Features extracted and saved to checkpoint")
    
    logger.info("Step 4: Hierarchical clustering")
    chunker = HierarchicalChunker(
        temporal_weight=temporal_weight,
        fine_duration=fine_duration,
        medium_duration=medium_duration,
        coarse_duration=coarse_duration
    )
    
    chunks = chunker.chunk_video(shots)
    hierarchy = chunker.build_hierarchy(chunks)
    
    for lvl, level_chunks in chunks.items():
        logger.info(f"{lvl.capitalize()}: {len(level_chunks)} chunks")
    
    if use_boundary_refinement:
        logger.info("Step 5: Boundary refinement")
        try:
            refiner = BoundaryRefiner(
                provider=caption_provider,
                model=reasoning_model or caption_model
            )
            
            for lvl in ["fine", "medium"]:
                if lvl in chunks:
                    chunks[lvl] = refiner.refine_boundaries(
                        shots, chunks[lvl]
                    )
            
            hierarchy = chunker.build_hierarchy(chunks)
            logger.info("Boundary refinement completed")
        except Exception as e:
            logger.warning(f"Boundary refinement failed: {e}")
    
    if generate_captions:
        logger.info("Step 6: Caption generation")
        try:
            caption_gen = CaptionGenerator(
                provider=caption_provider,
                model=caption_model
            )
            
            chunks = caption_gen.generate_captions_for_chunks(
                video_path,
                chunks
            )
            logger.info("Caption generation completed")
        except Exception as e:
            logger.warning(f"Caption generation failed: {e}")
    
    logger.info("Step 7: Building output")
    
    video_metadata["total_shots"] = len(shots)
    
    output = {
        "video_metadata": video_metadata,
        "shots": [shot.to_dict() for shot in shots],
        "chunks": {
            lvl: [chunk.to_dict() for chunk in level_chunks]
            for lvl, level_chunks in chunks.items()
        },
        "hierarchy": hierarchy
    }
    
    logger.info("Step 8: Validation")
    
    schema_errors = validate_output_schema(output)
    if schema_errors:
        logger.warning("Schema validation errors:")
        for error in schema_errors:
            logger.warning(f"  - {error}")
    else:
        logger.info("Schema validation passed")
    
    hierarchy_errors = check_hierarchy_consistency(chunks, hierarchy)
    if hierarchy_errors:
        logger.warning("Hierarchy consistency errors:")
        for error in hierarchy_errors:
            logger.warning(f"  - {error}")
    else:
        logger.info("Hierarchy consistency check passed")
    
    metrics = compute_quality_metrics(shots, chunks)
    logger.info("Quality metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    output["quality_metrics"] = metrics
    
    output_path = os.path.join(output_dir, "chunks.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    logger.info(f"Output saved to: {output_path}")
    
    logger.info("=" * 60)
    logger.info("Video Chunking Pipeline Completed")
    logger.info("=" * 60)
    logger.info(f"Total shots: {len(shots)}")
    logger.info(f"Fine chunks: {len(chunks.get('fine', []))}")
    logger.info(f"Medium chunks: {len(chunks.get('medium', []))}")
    logger.info(f"Coarse chunks: {len(chunks.get('coarse', []))}")
    
    return output


def main():
    args = parse_args()
    run_pipeline(
        video_path=args.video_path,
        output_dir=args.output_dir,
        temporal_weight=args.temporal_weight,
        fine_duration=args.fine_duration,
        medium_duration=args.medium_duration,
        coarse_duration=args.coarse_duration,
        use_boundary_refinement=args.use_boundary_refinement,
        generate_captions=args.generate_captions,
        skip_audio=args.skip_audio,
        device=args.device,
        resume_from=args.resume_from,
        caption_provider=args.caption_provider,
        caption_model=args.caption_model,
        log_level=args.log_level,
        shot_threshold=args.shot_threshold,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
