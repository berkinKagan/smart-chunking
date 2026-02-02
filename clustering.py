import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform

from utils import Shot, Chunk, normalize_embedding, cosine_similarity


logger = logging.getLogger("video_chunker")


class HierarchicalChunker:
    def __init__(
        self,
        temporal_weight: float = 0.3,
        fine_duration: float = 30.0,
        medium_duration: float = 120.0,
        coarse_duration: float = 300.0,
        min_shots_per_chunk: int = 2,
        linkage_method: str = "ward"
    ):
        self.temporal_weight = temporal_weight
        self.fine_duration = fine_duration
        self.medium_duration = medium_duration
        self.coarse_duration = coarse_duration
        self.min_shots_per_chunk = min_shots_per_chunk
        self.linkage_method = linkage_method
    
    def compute_distance_matrix(
        self,
        shots: List[Shot]
    ) -> np.ndarray:
        n = len(shots)
        logger.info(f"Computing distance matrix for {n} shots")
        
        embeddings = []
        timestamps = []
        
        for shot in shots:
            if shot.fused_embedding is not None:
                embeddings.append(np.array(shot.fused_embedding))
            else:
                embeddings.append(np.zeros(1792))
            timestamps.append((shot.start_time + shot.end_time) / 2)
        
        embeddings = np.array(embeddings)
        timestamps = np.array(timestamps)
        
        semantic_dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                dist = 1 - sim
                semantic_dist[i, j] = dist
                semantic_dist[j, i] = dist
        
        max_time_gap = timestamps[-1] - timestamps[0] if n > 1 else 1.0
        temporal_dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                time_gap = abs(timestamps[i] - timestamps[j])
                normalized_gap = time_gap / max_time_gap if max_time_gap > 0 else 0
                temporal_dist[i, j] = normalized_gap
                temporal_dist[j, i] = normalized_gap
        
        combined_dist = semantic_dist + self.temporal_weight * temporal_dist
        
        return combined_dist
    
    def perform_clustering(
        self,
        distance_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        logger.info("Performing hierarchical agglomerative clustering")
        
        condensed_dist = squareform(distance_matrix, checks=False)
        
        linkage_matrix = linkage(condensed_dist, method=self.linkage_method)
        
        return linkage_matrix, condensed_dist
    
    def get_cluster_assignments(
        self,
        linkage_matrix: np.ndarray,
        n_clusters: int
    ) -> np.ndarray:
        n_clusters = max(1, n_clusters)
        labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust")
        return labels
    
    def compute_target_clusters(
        self,
        total_duration: float
    ) -> Dict[str, int]:
        fine_clusters = max(2, int(total_duration / self.fine_duration))
        medium_clusters = max(2, int(total_duration / self.medium_duration))
        coarse_clusters = max(1, int(total_duration / self.coarse_duration))
        
        return {
            "fine": fine_clusters,
            "medium": medium_clusters,
            "coarse": coarse_clusters
        }
    
    def create_chunks_from_labels(
        self,
        shots: List[Shot],
        labels: np.ndarray,
        level: str
    ) -> List[Chunk]:
        cluster_shots = {}
        for shot, label in zip(shots, labels):
            if label not in cluster_shots:
                cluster_shots[label] = []
            cluster_shots[label].append(shot)
        
        chunks = []
        
        for i, (label, shot_list) in enumerate(sorted(cluster_shots.items())):
            shot_list.sort(key=lambda s: s.start_time)
            
            chunk_id = f"{level}_{i}"
            start_time = shot_list[0].start_time
            end_time = shot_list[-1].end_time
            shot_ids = [s.shot_id for s in shot_list]
            
            transcripts = [s.transcript for s in shot_list if s.transcript]
            combined_transcript = " ".join(transcripts)
            
            embeddings = [
                np.array(s.fused_embedding) 
                for s in shot_list 
                if s.fused_embedding is not None
            ]
            if embeddings:
                avg_embedding = np.mean(embeddings, axis=0)
                avg_embedding = normalize_embedding(avg_embedding)
            else:
                avg_embedding = np.zeros(1792)
            
            chunk = Chunk(
                chunk_id=chunk_id,
                level=level,
                start_time=start_time,
                end_time=end_time,
                shot_ids=shot_ids,
                transcript=combined_transcript,
                embedding=avg_embedding.tolist(),
                reasoning=f"Grouped {len(shot_ids)} shots via hierarchical clustering at {level} level."
            )
            chunks.append(chunk)
        
        chunks.sort(key=lambda c: c.start_time)
        for i, chunk in enumerate(chunks):
            chunk.chunk_id = f"{level}_{i}"
        
        return chunks
    
    def enforce_temporal_contiguity(
        self,
        shots: List[Shot],
        labels: np.ndarray
    ) -> np.ndarray:
        n = len(shots)
        new_labels = np.zeros(n, dtype=int)
        
        current_label = 0
        new_labels[0] = current_label
        
        for i in range(1, n):
            if labels[i] != labels[i - 1]:
                current_label += 1
            new_labels[i] = current_label
        
        return new_labels
    
    def merge_small_clusters(
        self,
        shots: List[Shot],
        labels: np.ndarray,
        min_size: int
    ) -> np.ndarray:
        unique_labels = np.unique(labels)
        cluster_sizes = {l: np.sum(labels == l) for l in unique_labels}
        
        new_labels = labels.copy()
        
        for label in unique_labels:
            if cluster_sizes[label] < min_size:
                indices = np.where(labels == label)[0]
                
                for idx in indices:
                    if idx > 0:
                        new_labels[idx] = new_labels[idx - 1]
                    elif idx < len(labels) - 1:
                        new_labels[idx] = labels[idx + 1]
        
        return new_labels
    
    def chunk_video(
        self,
        shots: List[Shot]
    ) -> Dict[str, List[Chunk]]:
        if len(shots) < 2:
            logger.warning("Too few shots for clustering")
            chunk = Chunk(
                chunk_id="fine_0",
                level="fine",
                start_time=shots[0].start_time if shots else 0,
                end_time=shots[-1].end_time if shots else 0,
                shot_ids=[s.shot_id for s in shots],
                transcript=" ".join([s.transcript for s in shots if s.transcript])
            )
            return {
                "fine": [chunk],
                "medium": [chunk],
                "coarse": [chunk]
            }
        
        distance_matrix = self.compute_distance_matrix(shots)
        linkage_matrix, _ = self.perform_clustering(distance_matrix)
        
        total_duration = shots[-1].end_time - shots[0].start_time
        target_clusters = self.compute_target_clusters(total_duration)
        
        logger.info(f"Target clusters - Fine: {target_clusters['fine']}, "
                   f"Medium: {target_clusters['medium']}, "
                   f"Coarse: {target_clusters['coarse']}")
        
        chunks = {}
        
        for level in ["fine", "medium", "coarse"]:
            n_clusters = min(target_clusters[level], len(shots))
            
            labels = self.get_cluster_assignments(linkage_matrix, n_clusters)
            
            labels = self.enforce_temporal_contiguity(shots, labels)
            
            labels = self.merge_small_clusters(shots, labels, self.min_shots_per_chunk)
            
            level_chunks = self.create_chunks_from_labels(shots, labels, level)
            chunks[level] = level_chunks
            
            logger.info(f"{level.capitalize()} level: {len(level_chunks)} chunks")
        
        return chunks
    
    def build_hierarchy(
        self,
        chunks: Dict[str, List[Chunk]]
    ) -> Dict[str, Dict[str, str]]:
        hierarchy = {
            "fine_to_medium": {},
            "medium_to_coarse": {}
        }
        
        fine_chunks = chunks.get("fine", [])
        medium_chunks = chunks.get("medium", [])
        coarse_chunks = chunks.get("coarse", [])
        
        for fine_chunk in fine_chunks:
            fine_mid = (fine_chunk.start_time + fine_chunk.end_time) / 2
            
            best_medium = None
            for medium_chunk in medium_chunks:
                if medium_chunk.start_time <= fine_mid <= medium_chunk.end_time:
                    best_medium = medium_chunk.chunk_id
                    break
            
            if best_medium is None and medium_chunks:
                min_dist = float("inf")
                for medium_chunk in medium_chunks:
                    med_mid = (medium_chunk.start_time + medium_chunk.end_time) / 2
                    dist = abs(fine_mid - med_mid)
                    if dist < min_dist:
                        min_dist = dist
                        best_medium = medium_chunk.chunk_id
            
            if best_medium:
                hierarchy["fine_to_medium"][fine_chunk.chunk_id] = best_medium
                fine_chunk.parent_chunk_id = best_medium
        
        for medium_chunk in medium_chunks:
            med_mid = (medium_chunk.start_time + medium_chunk.end_time) / 2
            
            best_coarse = None
            for coarse_chunk in coarse_chunks:
                if coarse_chunk.start_time <= med_mid <= coarse_chunk.end_time:
                    best_coarse = coarse_chunk.chunk_id
                    break
            
            if best_coarse is None and coarse_chunks:
                min_dist = float("inf")
                for coarse_chunk in coarse_chunks:
                    coarse_mid = (coarse_chunk.start_time + coarse_chunk.end_time) / 2
                    dist = abs(med_mid - coarse_mid)
                    if dist < min_dist:
                        min_dist = dist
                        best_coarse = coarse_chunk.chunk_id
            
            if best_coarse:
                hierarchy["medium_to_coarse"][medium_chunk.chunk_id] = best_coarse
                medium_chunk.parent_chunk_id = best_coarse
        
        return hierarchy


def compute_boundary_context(
    shots: List[Shot],
    chunk1: Chunk,
    chunk2: Chunk,
    context_seconds: float = 10.0
) -> Tuple[str, str]:
    boundary_time = chunk1.end_time
    
    before_text = []
    after_text = []
    
    for shot in shots:
        if boundary_time - context_seconds <= shot.end_time <= boundary_time:
            if shot.transcript:
                before_text.append(shot.transcript)
        elif boundary_time <= shot.start_time <= boundary_time + context_seconds:
            if shot.transcript:
                after_text.append(shot.transcript)
    
    return " ".join(before_text), " ".join(after_text)
