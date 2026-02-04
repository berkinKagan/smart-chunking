import logging
import numpy as np
from typing import List, Dict, Any
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from utils import Sample, Chunk
from config import TEMPORAL_WEIGHT, CODEC_ALIGNMENT_WEIGHT, TARGET_DURATIONS

logger = logging.getLogger("hybrid_hierarchical")

class HybridClustering:
    def cluster(self, samples: List[Sample], codec_data: Dict[str, Any], levels: List[str]) -> Dict[str, List[Chunk]]:
        if not samples:
            return {lvl: [] for lvl in levels}
            
        n = len(samples)
        logger.info(f"Clustering {n} samples using hybrid distance matrix")
        
        # 1. Compute Hybrid Distance Matrix
        dist_matrix = np.zeros((n, n))
        max_time = codec_data["duration"]
        
        for i in range(n):
            for j in range(i + 1, n):
                # Semantic distance (Visual only for now)
                sim = np.dot(samples[i].visual_embedding, samples[j].visual_embedding)
                semantic_dist = 1 - sim
                
                # Temporal distance
                temporal_dist = abs(samples[i].time - samples[j].time) / max_time
                
                # Codec penalty: Don't merge across high-confidence boundaries
                codec_penalty = 0
                if samples[i].codec_confidence == "high" or samples[j].codec_confidence == "high":
                    # Only apply penalty if they are on opposite sides of a codec boundary
                    # (Simplified here: apply if they are far apart or both high confidence)
                    codec_penalty = CODEC_ALIGNMENT_WEIGHT
                
                d = semantic_dist + TEMPORAL_WEIGHT * temporal_dist + codec_penalty
                dist_matrix[i, j] = dist_matrix[j, i] = d
                
        # 2. Perform HAC
        condensed_dist = squareform(dist_matrix)
        linkage_matrix = linkage(condensed_dist, method="ward")
        
        results = {}
        # 3. Cut at target levels
        for level in levels:
            target_dur = TARGET_DURATIONS.get(level, 60.0)
            n_clusters = max(1, int(max_time / target_dur))
            
            labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust")
            
            # Convert clusters to contiguous chunks
            level_chunks = self._labels_to_chunks(samples, labels, level)
            results[level] = level_chunks
            logger.info(f"{level.capitalize()} level: {len(level_chunks)} chunks created.")
            
        return results

    def _labels_to_chunks(self, samples: List[Sample], labels: np.ndarray, level: str) -> List[Chunk]:
        chunks = []
        if len(labels) == 0: return chunks
        
        current_label = labels[0]
        start_idx = 0
        
        for i in range(1, len(labels)):
            # If label changes, we have a boundary
            if labels[i] != current_label:
                chunks.append(self._create_chunk(samples, start_idx, i, level, len(chunks)))
                start_idx = i
                current_label = labels[i]
                
        # Final chunk
        chunks.append(self._create_chunk(samples, start_idx, len(labels), level, len(chunks)))
        return chunks

    def _create_chunk(self, samples: List[Sample], start_idx: int, end_idx: int, level: str, count: int) -> Chunk:
        chunk_samples = samples[start_idx:end_idx]
        start_time = chunk_samples[0].time
        end_time = chunk_samples[-1].time
        
        # Simple centroid embedding
        all_embs = [s.multimodal_embedding for s in chunk_samples]
        avg_emb = np.mean(all_embs, axis=0)
        
        # Check if boundary is codec-aligned (using the first sample's metadata)
        codec_aligned = chunk_samples[0].codec_confidence != "low"
        
        return Chunk(
            chunk_id=f"{level}_{count}",
            level=level,
            start_time=start_time,
            end_time=end_time,
            sample_indices=list(range(start_idx, end_idx)),
            embedding=avg_emb,
            codec_aligned=codec_aligned,
            confidence_score=0.8 if codec_aligned else 0.5
        )
