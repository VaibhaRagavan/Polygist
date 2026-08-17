# Polygist

A multimodal summarization app built with Streamlit. Upload a PDF, image, audio/video file, or paste a URL (including YouTube) and get an AI-generated summary — or ask questions against a PDF using RAG.

## Features

- **Summary tab** — supports PDF, image, audio, video, and direct URLs (including YouTube)
- **Q&A tab** — PDF-only, RAG-based chat (one PDF per session; refresh to switch)
- Domain-aware summaries for video/audio transcripts (auto-detects meeting / hospital review / lecture / other)
- Session-scoped Pinecone namespaces, with scheduled cleanup via GitHub Actions
- Full request tracing via LangSmith (`@traceable`)

## Flow

```mermaid
flowchart TD
    A[User Input] --> B{Input Type}

    B -->|PDF upload| C[pdfplumber extraction]
    B -->|Image upload/URL| D[Bedrock Nova - single shot]
    B -->|Audio/Video upload| E[ffprobe duration check]
    B -->|URL| F{urlparse netloc}

    F -->|YouTube| G[yt-dlp download]
    F -->|Other URL| H{Content-Type header}
    H -->|image| D
    H -->|pdf| C
    H -->|audio/video| E

    G --> E
    E --> I[S3 upload]
    I --> J[AWS Transcribe job]
    J --> K[Poll for completion]
    K --> L[Delete from S3]
    L --> M["generate_summary - media_type=video"]

    C --> N["generate_summary - media_type=pdf"]
    D --> O[Summary Result]
    M --> O
    N --> O

    C --> P[Chunk Text]
    P --> Q[Titan Embed v2]
    Q --> R[Pinecone upsert - session namespace]
    R --> S[User Question]
    S --> T[Embed query]
    T --> U[Pinecone top-3 retrieve]
    U --> V[Bedrock QA answer]
    V --> W[Q&A Result]
```

## Tech Stack

| Layer | Tool |
|---|---|
| Frontend | Streamlit |
| LLM | AWS Bedrock (Nova Lite for generation, Titan Embed Text v2 for embeddings) |
| Vector store | Pinecone (session-based UUID namespaces) |
| Audio/video transcription | AWS Transcribe + S3 (staging) |
| PDF parsing | pdfplumber |
| Media validation | ffprobe |
| YouTube ingestion | yt-dlp (local only — see Known Issues) |
| Tracing | LangSmith |
| Cleanup | GitHub Actions scheduled workflow (cron, UTC) |

## Architecture Notes

- Images and video use single-shot Bedrock calls; only PDFs go through full RAG.
- URL routing checks `urlparse(url).netloc` **before** making any `requests.get()` call — YouTube URLs return `text/html`, so content-type-based routing alone can't distinguish them.
- Pinecone records carry a `created_at` timestamp in metadata; cleanup is a scheduled job filtering with `$lte`, one namespace at a time (4MB upsert limit means batching ≤500 vectors/call).

## Known Issues

- **yt-dlp works locally but fails on Streamlit Cloud (HTTP 403).** Streamlit Cloud runs on datacenter IP ranges that YouTube blocks for direct downloads. This affects the YouTube branch in `app.py`'s URL-handling block.
  - **Fix in progress:** switch to `youtube-transcript-api`, which pulls captions directly (no download), avoiding the IP block. Preferred first-line solution for captioned videos.
  - **Still needs handling:** videos with no captions available, and auto-generated caption quality/formatting.
- Pipeline is slow for short videos — no per-stage timing yet to confirm, but Transcribe polling is the suspected bottleneck.

## Setup

```bash
pip install -r requirements.txt
```

Also requires `ffmpeg` (see `packages.txt` — needed for Streamlit Cloud's apt-based installs).

### Environment variables (`.env`)
- `S3_BUCKET_NAME`
- `PINECONE_API_KEY`
- `LANGSMITH_API_KEY`

### Run
```bash
streamlit run app.py
```

## Roadmap
- [ ] Replace yt-dlp with `youtube-transcript-api` for cloud compatibility
- [ ] Per-stage pipeline timing to diagnose slow processing
- [ ] Chat memory (last ~10 exchanges) in Q&A tab
- [ ] Adding Q/A for image and videos 
