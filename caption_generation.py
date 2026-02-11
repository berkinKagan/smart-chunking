import os
import base64
import logging
from typing import List, Dict, Optional
from PIL import Image
import io

from utils import Chunk, Shot, extract_keyframes


logger = logging.getLogger("video_chunker")


class CaptionGenerator:
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None
    ):
        self.provider = provider
        self.model = model
        
        if provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        elif provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
            )
        elif provider == "ollama":
            import ollama
            self.client = ollama.Client(
                host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
                timeout=300.0  # 5 minute timeout to prevent hanging
            )
        elif provider == "gemini":
            from google import genai
            self.client = genai.Client(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
        elif provider == "openrouter":
            from openai import OpenAI
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key or os.environ.get("OPENROUTER_API_KEY")
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _encode_image(self, frame) -> str:
        if isinstance(frame, Image.Image):
            pil_image = frame
        else:
            pil_image = Image.fromarray(frame)
        
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def _get_prompt_for_level(
        self,
        level: str,
        duration: float,
        transcript: str
    ) -> str:
        duration_str = f"{duration:.1f} seconds" if duration < 60 else f"{duration/60:.1f} minutes"
        
        if level == "fine":
            return f"""These frames and transcript represent a {duration_str} video segment.

Transcript: {transcript if transcript else "[No speech detected]"}

Provide a 2-3 sentence description focusing on:
- Specific actions or events
- Key dialogue or narration
- Visual details

Description:"""
        
        elif level == "medium":
            return f"""This is a {duration_str} video segment.

Transcript: {transcript if transcript else "[No speech detected]"}

Provide a paragraph (4-6 sentences) covering:
- Main topic or theme
- Key events in chronological order
- Important dialogue or arguments
- How context flows within the segment

Description:"""
        
        else:
            return f"""This is a {duration_str} video section.

Transcript: {transcript if transcript else "[No speech detected]"}

Provide a comprehensive summary including:
1. Overarching theme or topic (1-2 sentences)
2. Major events or discussion points (bullet points)
3. How this section connects to the broader narrative
4. Key takeaways

Summary:"""
    
    def _get_num_frames_for_level(self, level: str) -> int:
        frame_counts = {
            "fine": 3,
            "medium": 5,
            "coarse": 8
        }
        # Further reduce for Ollama to save CPU
        if self.provider == "ollama":
            from config import OLLAMA_MAX_FRAMES
            return min(frame_counts.get(level, 3), OLLAMA_MAX_FRAMES)
        return frame_counts.get(level, 5)
    
    def generate_caption_openai(
        self,
        frames: List,
        prompt: str
    ) -> str:
        content = [{"type": "text", "text": prompt}]
        
        for frame in frames[:10]:
            base64_image = self._encode_image(frame)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "low"
                }
            })
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    
    def generate_caption_anthropic(
        self,
        frames: List,
        prompt: str
    ) -> str:
        content = []
        
        for frame in frames[:10]:
            base64_image = self._encode_image(frame)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64_image
                }
            })
        
        content.append({"type": "text", "text": prompt})
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ]
        )
        
        return response.content[0].text.strip()
    
    def generate_caption_ollama(
        self,
        frames: List,
        prompt: str
    ) -> str:
        images = []
        if frames:
            frame = frames[len(frames) // 2]
            if not isinstance(frame, Image.Image):
                frame = Image.fromarray(frame)
            
            buffer = io.BytesIO()
            frame.save(buffer, format="JPEG")
            images.append(buffer.getvalue())
        
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            images=images
        )
        
        return response.get('response', '').strip()
    
    def generate_caption_gemini(
        self,
        frames: List,
        prompt: str
    ) -> str:
        from google.genai import types
        
        contents = []
        
        for frame in frames[:10]:
            if isinstance(frame, Image.Image):
                pil_image = frame
            else:
                pil_image = Image.fromarray(frame)
            
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=85)
            image_bytes = buffer.getvalue()
            
            contents.append(types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/jpeg'
            ))
        
        contents.append(prompt)
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents
        )
        
        return response.text.strip()

    def generate_caption_openrouter(
        self,
        frames: List,
        prompt: str
    ) -> str:
        content = [{"type": "text", "text": prompt}]

        for frame in frames[:10]:
            base64_image = self._encode_image(frame)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "low"
                }
            })

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=500
        )

        return response.choices[0].message.content.strip()

    def generate_caption(
        self,
        video_path: str,
        chunk: Chunk
    ) -> str:
        num_frames = self._get_num_frames_for_level(chunk.level)
        
        # Try I-frame extraction first (much faster)
        try:
            from utils import extract_iframes_for_segment
            frames = extract_iframes_for_segment(
                video_path,
                chunk.start_time,
                chunk.end_time,
                max_frames=num_frames
            )
            if frames:
                logger.info(f"Using {len(frames)} I-frames for chunk {chunk.chunk_id}")
            else:
                # Fallback to standard keyframe extraction if no I-frames found
                frames = extract_keyframes(
                    video_path,
                    chunk.start_time,
                    chunk.end_time,
                    num_frames
                )
        except Exception as e:
            logger.warning(f"I-frame extraction failed, falling back: {e}")
            frames = extract_keyframes(
                video_path,
                chunk.start_time,
                chunk.end_time,
                num_frames
            )
        
        prompt = self._get_prompt_for_level(
            chunk.level,
            chunk.duration,
            chunk.transcript
        )
        
        if not frames:
            prompt = f"""Based on the following transcript from a {chunk.duration:.1f} second video segment, provide a description.

Transcript: {chunk.transcript if chunk.transcript else "[No speech detected]"}

Description:"""
        
        try:
            if self.provider == "openai":
                caption = self.generate_caption_openai(frames, prompt)
            elif self.provider == "anthropic":
                caption = self.generate_caption_anthropic(frames, prompt)
            elif self.provider == "ollama":
                caption = self.generate_caption_ollama(frames, prompt)
            elif self.provider == "gemini":
                caption = self.generate_caption_gemini(frames, prompt)
            elif self.provider == "openrouter":
                caption = self.generate_caption_openrouter(frames, prompt)
        except Exception as e:
            logger.error(f"Caption generation failed for {chunk.chunk_id}: {e}")
            caption = f"[Caption generation failed: {str(e)}]"
        
        return caption
    
    def generate_captions_for_chunks(
        self,
        video_path: str,
        chunks: Dict[str, List[Chunk]],
        levels: Optional[List[str]] = None
    ) -> Dict[str, List[Chunk]]:
        if levels is None:
            levels = ["fine", "medium", "coarse"]
        
        from tqdm import tqdm
        
        for level in levels:
            if level not in chunks:
                continue
            
            level_chunks = chunks[level]
            logger.info(f"Generating captions for {len(level_chunks)} {level} chunks")
            
            for chunk in tqdm(level_chunks, desc=f"{level} captions"):
                caption = self.generate_caption(video_path, chunk)
                chunk.caption = caption
        
        return chunks


class BoundaryRefiner:
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        merge_threshold: float = 0.7
    ):
        self.provider = provider
        self.model = model
        self.merge_threshold = merge_threshold
        
        if provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        elif provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
            )
        elif provider == "ollama":
            import ollama
            self.client = ollama.Client(
                host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
                timeout=300.0  # 5 minute timeout to prevent hanging
            )
    
    def compute_continuity_score(
        self,
        transcript_before: str,
        transcript_after: str
    ) -> float:
        if not transcript_before and not transcript_after:
            return 0.5
        
        prompt = f"""Segment A (before boundary): {transcript_before if transcript_before else "[No speech]"}
Segment B (after boundary): {transcript_after if transcript_after else "[No speech]"}

Question: Do these segments discuss the same topic or represent continuous action? 
Answer with ONLY a continuity score from 0.0 to 1.0, where:
- 1.0 = Same topic/action, should be merged
- 0.5 = Related but transitional
- 0.0 = Completely different topic/scene

Score:"""
        
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10
                )
                score_text = response.choices[0].message.content.strip()
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=10,
                    messages=[{"role": "user", "content": prompt}]
                )
                score_text = response.content[0].text.strip()
            elif self.provider == "ollama":
                response = self.client.generate(
                    model=self.model,
                    prompt=prompt
                )
                score_text = response.get('response', '').strip()
            
            score = float(score_text.replace(",", "."))
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.warning(f"Continuity score computation failed: {e}")
            return 0.5
    
    def refine_boundaries(
        self,
        shots: List[Shot],
        chunks: List[Chunk],
        context_seconds: float = 10.0
    ) -> List[Chunk]:
        from clustering import compute_boundary_context
        
        if len(chunks) < 2:
            return chunks
        
        logger.info(f"Refining boundaries for {len(chunks)} chunks")
        
        refined_chunks = list(chunks)
        merge_indices = []
        
        for i in range(len(refined_chunks) - 1):
            chunk1 = refined_chunks[i]
            chunk2 = refined_chunks[i + 1]
            
            before_text, after_text = compute_boundary_context(
                shots, chunk1, chunk2, context_seconds
            )
            
            score = self.compute_continuity_score(before_text, after_text)
            logger.debug(f"Boundary {chunk1.chunk_id}/{chunk2.chunk_id}: score={score:.2f}")
            
            if score > self.merge_threshold:
                merge_indices.append(i)
        
        for idx in reversed(merge_indices):
            chunk1 = refined_chunks[idx]
            chunk2 = refined_chunks[idx + 1]
            
            merged = Chunk(
                chunk_id=chunk1.chunk_id,
                level=chunk1.level,
                start_time=chunk1.start_time,
                end_time=chunk2.end_time,
                shot_ids=chunk1.shot_ids + chunk2.shot_ids,
                transcript=f"{chunk1.transcript} {chunk2.transcript}".strip(),
                reasoning=f"Merged {chunk1.chunk_id} and {chunk2.chunk_id} (Continuity Score: {score:.2f}) based on LLM boundary refinement."
            )
            
            refined_chunks[idx] = merged
            del refined_chunks[idx + 1]
        
        for i, chunk in enumerate(refined_chunks):
            chunk.chunk_id = f"{chunk.level}_{i}"
        
        logger.info(f"Merged {len(merge_indices)} boundaries, "
                   f"resulting in {len(refined_chunks)} chunks")
        
        return refined_chunks
