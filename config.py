import os

# Stage 1: Codec Analysis
CODEC_HIGH_CONFIDENCE_THRESHOLD = 0.7  # Relative jump in packet size
GOP_ANALYSIS_WINDOW = 5  # Number of GOPs to look at for local stats

# Stage 2: Smart Sampling
HIGH_PRIORITY_FPS = 1.0       # Sampling rate near high-confidence boundaries
MEDIUM_PRIORITY_FPS = 0.5     # Sampling rate near medium-confidence boundaries
STABLE_REGION_FPS = 0.1       # Sampling rate in quiet regions
MAX_TOTAL_SAMPLES = 1000      # Global frame budget for multimodal extraction
MIN_SAMPLES_PER_WINDOW = 3     # Minimum frames to sample if a window is chosen

# Stage 3: Multimodal Models
CLIP_MODEL = "openai/clip-vit-large-patch14"
CLAP_MODEL = "laion/clap-htsat-fused"
WHISPER_MODEL = "base"
SBERT_MODEL = "all-MiniLM-L6-v2"

# Stage 4: Clustering
TEMPORAL_WEIGHT = 0.3
CODEC_ALIGNMENT_WEIGHT = 0.2
DEFAULT_LEVELS = ["fine", "medium", "coarse"]
TARGET_DURATIONS = {
    "fine": 30.0,
    "medium": 120.0,
    "coarse": 300.0
}

# Stage 5: LLM Refinement
LLM_PROVIDER = "ollama"  # Options: "gemini", "ollama", "openai", "anthropic"
GEMINI_MODEL = "gemini-3-flash-preview"
OLLAMA_MODEL = "llama3.2-vision:11b"
LLM_REFINEMENT_THRESHOLD = 0.5  # Only refine if codec/semantic confidence is low
SELECTIVE_REFINEMENT = True

# Ollama optimization (reduce CPU usage)
OLLAMA_MAX_FRAMES = 5  # Reduce frames for Ollama to save resources (default: 20 for coarse)
OLLAMA_MAX_CHUNKS = 20  # Limit max chunks to caption (None for no limit)

# IO
CACHE_DIR = "cache/hybrid_hierarchical"
os.makedirs(CACHE_DIR, exist_ok=True)
