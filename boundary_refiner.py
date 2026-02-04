import logging
from typing import List, Dict, Any
from utils import Chunk
from config import GEMINI_MODEL, OLLAMA_MODEL, LLM_PROVIDER, LLM_REFINEMENT_THRESHOLD
from caption_generation import CaptionGenerator

logger = logging.getLogger("hybrid_hierarchical")

class BoundaryRefiner:
    def __init__(self):
        # Initialize CaptionGenerator with configured provider
        model = OLLAMA_MODEL if LLM_PROVIDER == "ollama" else GEMINI_MODEL
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
        
        # 3. Propagate captions to children (simplified)
        for level in ["medium", "fine"]:
            if level in chunks_dict:
                for chunk in chunks_dict[level]:
                    chunk.caption = f"Detail of {level} segment."
                    
        return chunks_dict
