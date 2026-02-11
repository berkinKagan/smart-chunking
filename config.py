import os

CODEC_HIGH_CONFIDENCE_THRESHOLD = 0.7
GOP_ANALYSIS_WINDOW = 5

HIGH_PRIORITY_FPS = 1.0
MEDIUM_PRIORITY_FPS = 0.5
STABLE_REGION_FPS = 0.1
MAX_TOTAL_SAMPLES = 1000
MIN_SAMPLES_PER_WINDOW = 3
CLIP_MODEL = "openai/clip-vit-large-patch14"
CLAP_MODEL = "laion/clap-htsat-fused"
WHISPER_MODEL = "base"
SBERT_MODEL = "all-MiniLM-L6-v2"


TEMPORAL_WEIGHT = 0.3
CODEC_ALIGNMENT_WEIGHT = 0.2
DEFAULT_LEVELS = ["fine", "medium", "coarse"]
TARGET_DURATIONS = {
    "fine": 30.0,
    "medium": 120.0,
    "coarse": 300.0
}

LLM_PROVIDER = "openrouter"
GEMINI_MODEL = "gemini-3-flash-preview"
OLLAMA_MODEL = "llama3.2-vision:11b"
OPENROUTER_MODEL = "google/gemini-2.0-flash-001"
LLM_REFINEMENT_THRESHOLD = 0.5
SELECTIVE_REFINEMENT = True

OLLAMA_MAX_FRAMES = 5
OLLAMA_MAX_CHUNKS = 20

CACHE_DIR = "cache/hybrid_hierarchical"
os.makedirs(CACHE_DIR, exist_ok=True)
