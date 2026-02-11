import logging
from typing import List, Dict, Any
from utils import Chunk
from config import GEMINI_MODEL, OLLAMA_MODEL, OPENROUTER_MODEL, LLM_PROVIDER, LLM_REFINEMENT_THRESHOLD
from caption_generation import CaptionGenerator

logger = logging.getLogger("hybrid_hierarchical")

class BoundaryRefiner:
    def __init__(self):
        if LLM_PROVIDER == "ollama":
            model = OLLAMA_MODEL
        elif LLM_PROVIDER == "openrouter":
            model = OPENROUTER_MODEL
        else:
            model = GEMINI_MODEL
        self.caption_gen = CaptionGenerator(
            provider=LLM_PROVIDER,
            model=model
        )

    def refine(self, chunks_dict: Dict[str, List[Chunk]], video_path: str) -> Dict[str, List[Chunk]]:
        logger.info("Starting Stage 5: Selective Refinement and Captioning")
        
        # 1. Selective Boundary Refinement (Placeholder: Trust codec for now)
        # In a full impl, we would send low-confidence boundaries to LLM
        
        # 2. Labeling (Hybrid: Only Coarse chunks get full VLM treatment)
        if "coarse" in chunks_dict:
            from config import OLLAMA_MAX_CHUNKS, LLM_PROVIDER
            
            chunks_to_process = chunks_dict["coarse"]
            # Limit chunks if using Ollama to save CPU
            if LLM_PROVIDER == "ollama" and OLLAMA_MAX_CHUNKS:
                chunks_to_process = chunks_to_process[:OLLAMA_MAX_CHUNKS]
                logger.info(f"Processing {len(chunks_to_process)}/{len(chunks_dict['coarse'])} coarse chunks (limited for CPU conservation)")
            else:
                logger.info(f"Generating captions for {len(chunks_to_process)} coarse chunks")
            
            for i, chunk in enumerate(chunks_to_process, 1):
                try:
                    logger.info(f"Processing chunk {i}/{len(chunks_to_process)}: {chunk.chunk_id}")
                    caption = self.caption_gen.generate_caption(video_path, chunk)
                    chunk.caption = caption
                except Exception as e:
                    logger.error(f"Failed to generate caption for {chunk.chunk_id}: {e}")
                    chunk.caption = "[Caption generation failed]"
            
            # Set placeholder captions for remaining chunks
            if LLM_PROVIDER == "ollama" and OLLAMA_MAX_CHUNKS and len(chunks_dict["coarse"]) > OLLAMA_MAX_CHUNKS:
                for chunk in chunks_dict["coarse"][OLLAMA_MAX_CHUNKS:]:
                    chunk.caption = "[Caption skipped to conserve resources]"
        
        # 3. Propagate captions to children based on parent context
        coarse_chunks = chunks_dict.get("coarse", [])
        medium_chunks = chunks_dict.get("medium", [])
        
        for level, parent_level_chunks in [("medium", coarse_chunks), ("fine", medium_chunks)]:
            if level in chunks_dict:
                for chunk in chunks_dict[level]:
                    # Find parent by time overlap
                    mid_time = (chunk.start_time + chunk.end_time) / 2
                    parent_caption = None
                    for parent in parent_level_chunks:
                        if parent.start_time <= mid_time <= parent.end_time and parent.caption:
                            parent_caption = parent.caption[:150]
                            break
                    if parent_caption:
                        chunk.caption = f"[{level.capitalize()} detail] {parent_caption}..."
                    else:
                        chunk.caption = f"[{level.capitalize()} segment] {chunk.start_time:.1f}s - {chunk.end_time:.1f}s"
                    
        return chunks_dict
