# Context-Preserving Video Chunking Pipeline

A production-ready video chunking pipeline that creates semantically coherent, context-preserving chunks at multiple granularity levels (fine/medium/coarse) for long-form videos (2+ hours). Designed for downstream video question-answering systems.

## Features

- **Shot Detection**: Automatic scene/shot boundary detection using PySceneDetect
- **Multimodal Feature Extraction**: 
  - Visual: CLIP (openai/clip-vit-large-patch14)
  - Audio: CLAP (laion/clap-htsat-unfused)
  - Text: Whisper ASR + Sentence Transformers
- **Hierarchical Clustering**: Temporal-constrained HAC with multi-level chunking
- **Boundary Refinement**: Optional LLM-based boundary verification
- **Caption Generation**: VLM-based descriptive captions for each chunk
- **Resumability**: Checkpoint-based processing for long videos

## Installation

```bash
pip install -r requirements.txt
```

### Requirements
- Python 3.10+
- CUDA-capable GPU (recommended for large videos)
- FFmpeg installed on system

## Quick Start

### Basic Usage

```bash
python chunk_video.py \
  --video_path "path/to/video.mp4" \
  --output_dir "output/chunks/"
```

### Full Featured Run

```bash
python chunk_video.py \
  --video_path "path/to/video.mp4" \
  --output_dir "output/chunks/" \
  --temporal_weight 0.3 \
  --use_boundary_refinement \
  --generate_captions \
  --device "cuda"
```

### Resume Interrupted Processing

```bash
python chunk_video.py \
  --video_path "path/to/video.mp4" \
  --output_dir "output/chunks/" \
  --resume_from "output/chunks/checkpoint.pkl"
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--video_path` | **required** | Path to input video file |
| `--output_dir` | **required** | Directory for output files |
| `--temporal_weight` | 0.3 | Weight for temporal distance (λ) |
| `--fine_duration` | 30 | Target duration for fine chunks (seconds) |
| `--medium_duration` | 120 | Target duration for medium chunks (seconds) |
| `--coarse_duration` | 300 | Target duration for coarse chunks (seconds) |
| `--use_boundary_refinement` | false | Enable LLM boundary verification |
| `--generate_captions` | false | Enable VLM caption generation |
| `--skip_audio` | false | Skip audio processing |
| `--device` | cuda | Device for inference (cuda/cpu) |
| `--resume_from` | - | Checkpoint file to resume from |
| `--caption_provider` | openai | LLM provider (openai/anthropic) |
| `--caption_model` | gpt-4o-mini | Model for captions |
| `--shot_threshold` | 27.0 | Shot detection threshold |
| `--batch_size` | 32 | Batch size for feature extraction |
| `--log_level` | INFO | Logging level |

## Output Format

The pipeline outputs a JSON file (`chunks.json`) with the following structure:

```json
{
  "video_metadata": {
    "video_path": "path/to/video.mp4",
    "duration": 7200.0,
    "total_shots": 450,
    "processed_date": "2024-01-28T12:00:00"
  },
  "shots": [
    {
      "shot_id": 0,
      "start_time": 0.0,
      "end_time": 5.2,
      "visual_embedding": [...],
      "audio_embedding": [...],
      "text_embedding": [...],
      "transcript": "Welcome to today's lecture..."
    }
  ],
  "chunks": {
    "fine": [
      {
        "chunk_id": "fine_0",
        "level": "fine",
        "start_time": 0.0,
        "end_time": 28.5,
        "duration": 28.5,
        "num_shots": 5,
        "shot_ids": [0, 1, 2, 3, 4],
        "caption": "The speaker introduces...",
        "transcript": "Full transcript...",
        "embedding": [...],
        "parent_chunk_id": "medium_0"
      }
    ],
    "medium": [...],
    "coarse": [...]
  },
  "hierarchy": {
    "fine_to_medium": {"fine_0": "medium_0", ...},
    "medium_to_coarse": {"medium_0": "coarse_0", ...}
  },
  "quality_metrics": {
    "fine_intra_coherence": 0.85,
    "fine_inter_diversity": 0.72,
    ...
  }
}
```

## Chunk Levels

| Level | Target Duration | Use Case |
|-------|----------------|----------|
| **Fine** | ~30 seconds | Specific actions, dialogue exchanges, single ideas |
| **Medium** | ~2 minutes | Complete scenes, topic discussions, procedural steps |
| **Coarse** | ~5 minutes | Major themes, chapters, acts |

## Architecture

```
┌─────────────────┐
│   Video Input   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Shot Detection │ (PySceneDetect)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│     Multimodal Feature Extraction   │
│  ┌───────┐  ┌───────┐  ┌─────────┐ │
│  │ CLIP  │  │ CLAP  │  │ Whisper │ │
│  │Visual │  │ Audio │  │   ASR   │ │
│  └───┬───┘  └───┬───┘  └────┬────┘ │
│      │          │           │       │
│      └──────────┼───────────┘       │
│                 ▼                   │
│         Fused Embedding             │
│         (1792 dimensions)           │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Hierarchical Agglomerative         │
│  Clustering (Ward Linkage)          │
│  + Temporal Constraints             │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Multi-Level Chunking               │
│  ┌───────┐  ┌────────┐  ┌────────┐ │
│  │ Fine  │  │ Medium │  │ Coarse │ │
│  │ ~30s  │  │  ~2min │  │  ~5min │ │
│  └───────┘  └────────┘  └────────┘ │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────┐     ┌──────────────────┐
│    Boundary     │     │     Caption      │
│   Refinement    │     │   Generation     │
│   (Optional)    │     │   (Optional)     │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
            ┌─────────────────┐
            │   JSON Output   │
            └─────────────────┘
```

## Performance Optimization

- **GPU Acceleration**: Use `--device cuda` for faster inference
- **Batch Processing**: Frames processed in batches (configurable via `--batch_size`)
- **Caching**: Embeddings cached to disk to avoid recomputation
- **Checkpointing**: Progress saved every 100 shots for resumability
- **Memory Management**: Audio segments processed and cleaned up per-shot

## Environment Variables

For cloud-based caption generation and boundary refinement:

```bash
export OPENAI_API_KEY="your-api-key"
# OR
export ANTHROPIC_API_KEY="your-api-key"
```

For local LLM support via Ollama:
- Ensure Ollama is running (`ollama serve`)
- Models like `llama3.2-vision:11b` (for captions) or `llama3.1:8b` (for boundaries) should be pulled.
- Default Ollama host is `http://localhost:11434`, but can be overridden:
```bash
export OLLAMA_HOST="http://your-server:11434"
```

Example using local models:
```bash
python chunk_video.py \
  --video_path "video.mp4" \
  --output_dir "output/" \
  --generate_captions \
  --caption_provider "ollama" \
  --caption_model "llama3.2-vision:11b"
```

## Quality Metrics

The pipeline computes and reports:
- **Intra-chunk coherence**: Average cosine similarity within chunks (higher = better)
- **Inter-chunk diversity**: 1 - average similarity between adjacent chunks (higher = better)
- **Duration statistics**: Min, max, mean, std for each chunk level
- **Hierarchy consistency**: Validates parent-child relationships

## File Structure

```
smart_chunking/
├── chunk_video.py          # Main CLI entry point
├── feature_extraction.py   # Shot detection + embeddings
├── clustering.py           # HAC + multi-level chunking
├── caption_generation.py   # VLM captions + boundary refinement
├── utils.py                # Helper functions + data classes
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## License

MIT License
