import subprocess
import json
import logging
import numpy as np
from typing import List, Dict, Any
from utils import GOPInfo, BoundaryCandidate
from config import CODEC_HIGH_CONFIDENCE_THRESHOLD

logger = logging.getLogger("hybrid_hierarchical")

class CodecAnalyzer:
    def analyze(self, video_path: str) -> Dict[str, Any]:
        logger.info(f"Analyzing codec for {video_path}")
        
        # Run ffprobe to get frame data
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_frames", "-show_entries", "frame=pict_type,pts_time,pkt_size,key_frame",
            "-of", "json", video_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
        except Exception as e:
            logger.error(f"FFprobe failed: {e}")
            raise

        frames = data.get("frames", [])
        if not frames:
            raise ValueError("No frames found in video")

        # Process frames into GOPs
        gops = []
        current_gop = None
        iframe_times = []
        
        for i, f in enumerate(frames):
            is_key = f.get("key_frame") == 1 or f.get("pict_type") == "I"
            pts = float(f.get("pts_time", 0))
            size = int(f.get("pkt_size", 0))

            if is_key:
                iframe_times.append(pts)
                if current_gop is not None:
                    current_gop.end_time = pts
                    gops.append(current_gop)
                
                current_gop = GOPInfo(
                    gop_id=len(gops),
                    start_time=pts,
                    end_time=pts,  # updated next frame
                    avg_packet_size=size,
                    packet_size_variance=0,
                    motion_magnitude=0,
                    iframe_pts=pts,
                    iframe_size=size
                )
            elif current_gop:
                current_gop.avg_packet_size += size
                # Rough motion proxy: large P/B frames relative to I
                current_gop.motion_magnitude = max(current_gop.motion_magnitude, size / current_gop.iframe_size)
        
        if current_gop:
            current_gop.end_time = float(frames[-1]["pts_time"])
            gops.append(current_gop)

        # Identify boundary candidates via packet size jumps
        candidates = []
        for i in range(1, len(gops)):
            prev = gops[i-1]
            curr = gops[i]
            
            # Jump in I-frame size or motion is a good scene change signal
            jump = (curr.iframe_size - prev.iframe_size) / max(prev.iframe_size, 1)
            
            if jump > CODEC_HIGH_CONFIDENCE_THRESHOLD:
                conf = "high"
            elif jump > 0.3:
                conf = "medium"
            else:
                continue
                
            candidates.append(BoundaryCandidate(
                time=curr.start_time,
                confidence=conf,
                packet_size_jump=jump,
                reason=f"Codec jump: {jump:.2f}"
            ))

        logger.info(f"Codec analysis complete. Found {len(gops)} GOPs and {len(candidates)} candidates.")
        
        return {
            "duration": float(frames[-1]["pts_time"]),
            "iframe_times": iframe_times,
            "gops": gops,
            "candidates": candidates
        }
