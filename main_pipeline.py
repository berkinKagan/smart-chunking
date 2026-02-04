import os
import json
import time
import logging
from typing import Dict, Any
from codec_analyzer import CodecAnalyzer
from smart_sampler import SmartSampler
from multimodal_extractor import MultimodalExtractor
from hybrid_clustering import HybridClustering
from boundary_refiner import BoundaryRefiner
from config import DEFAULT_LEVELS

# Setup local logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hybrid_hierarchical")

def hybrid_hierarchical_chunking(video_path: str, output_path: str, device: str = "cpu"):
    start_time = time.time()
    logger.info(f"Starting Hybrid Hierarchical Pipeline for {video_path}")
    
    # Stage 1: Codec Analysis (5-10s)
    analyzer = CodecAnalyzer()
    codec_data = analyzer.analyze(video_path)
    
    # Stage 2: Smart Sampling (1-2s)
    sampler = SmartSampler()
    sampling_plan = sampler.create_plan(codec_data)
    
    # Stage 3: Multimodal Extraction (~2-3 mins for 1h video)
    extractor = MultimodalExtractor(device=device)
    samples = extractor.extract_features(video_path, sampling_plan, codec_data)
    
    # Stage 4: Hybrid Clustering (~1 min)
    clusterer = HybridClustering()
    chunks_dict = clusterer.cluster(samples, codec_data, DEFAULT_LEVELS)
    
    # Stage 5: Refinement & Captioning (~30s)
    refiner = BoundaryRefiner()
    final_chunks = refiner.refine(chunks_dict, video_path)
    
    # Format to Action100M-style output
    execution_time = time.time() - start_time
    
    # Build hierarchy links
    for level in ["fine", "medium"]:
        parent_level = "medium" if level == "fine" else "coarse"
        for chunk in final_chunks.get(level, []):
            mid_time = (chunk.start_time + chunk.end_time) / 2
            # Find parent that contains this midpoint
            for parent in final_chunks.get(parent_level, []):
                if parent.start_time <= mid_time <= parent.end_time:
                    chunk.parent_chunk_id = parent.chunk_id
                    break

    output = {
        "video_path": video_path,
        "duration": codec_data["duration"],
        "processing_time": execution_time,
        "chunks": {
            lvl: [c.to_dict() for c in clist]
            for lvl, clist in final_chunks.items()
        }
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
        
    logger.info(f"Pipeline complete! Processed in {execution_time:.2f}s. Output: {output_path}")
    return output

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--output", default="output/hybrid_chunks.json", help="Output JSON path")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    args = parser.parse_args()
    
    hybrid_hierarchical_chunking(args.video, args.output, args.device)
