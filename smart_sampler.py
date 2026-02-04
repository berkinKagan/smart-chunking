import logging
from typing import List, Dict, Any
from utils import SamplingWindow, BoundaryCandidate
from config import (
    HIGH_PRIORITY_FPS, MEDIUM_PRIORITY_FPS, STABLE_REGION_FPS,
    MAX_TOTAL_SAMPLES
)

logger = logging.getLogger("hybrid_hierarchical")

class SmartSampler:
    def create_plan(self, codec_data: Dict[str, Any], max_samples: int = MAX_TOTAL_SAMPLES) -> List[SamplingWindow]:
        duration = codec_data["duration"]
        candidates: List[BoundaryCandidate] = codec_data["candidates"]
        
        windows = []
        
        # 1. High-priority windows around candidates
        for c in candidates:
            if c.confidence == "high":
                windows.append(SamplingWindow(
                    start_time=max(0, c.time - 2.0),
                    end_time=min(duration, c.time + 2.0),
                    fps=HIGH_PRIORITY_FPS,
                    priority="high"
                ))
            elif c.confidence == "medium":
                windows.append(SamplingWindow(
                    start_time=max(0, c.time - 4.0),
                    end_time=min(duration, c.time + 4.0),
                    fps=MEDIUM_PRIORITY_FPS,
                    priority="medium"
                ))
        
        # 2. Add stable region windows for gaps > 30s
        windows.sort(key=lambda w: w.start_time)
        stable_windows = []
        last_end = 0.0
        
        for w in windows:
            if w.start_time - last_end > 30.0:
                # Sample the middle of large gaps
                mid = (last_end + w.start_time) / 2
                stable_windows.append(SamplingWindow(
                    start_time=max(0, mid - 5.0),
                    end_time=min(duration, mid + 5.0),
                    fps=STABLE_REGION_FPS,
                    priority="low"
                ))
            last_end = w.end_time
            
        if duration - last_end > 30.0:
            mid = (last_end + duration) / 2
            stable_windows.append(SamplingWindow(
                start_time=max(0, mid - 5.0),
                end_time=min(duration, mid + 5.0),
                fps=STABLE_REGION_FPS,
                priority="low"
            ))
            
        all_windows = windows + stable_windows
        all_windows.sort(key=lambda w: w.start_time)
        
        # 3. Budget check (very simple estimation)
        total_est = sum((w.end_time - w.start_time) * w.fps for w in all_windows)
        if total_est > max_samples:
            logger.warning(f"Estimated samples {total_est} exceeds budget {max_samples}. Pruning low priority.")
            all_windows = [w for w in all_windows if w.priority != "low"]
            
        logger.info(f"Smart sampling plan created: {len(all_windows)} windows.")
        return all_windows
