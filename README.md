# MAVSPOI: Multi-Agent Voting Scheme for real-time POI recommendation

Current Peogress: CoMaPOI-styled Variant (Frozen LLM, Yelp)

This repository now includes a CoMaPOI-styled multi-agent variant for:

- Query-based real-time POI recommendation
- Frozen LLM via OpenAI API (no fine-tuning)
- Yelp Open Dataset as POI source

## 1) Architecture Mapping (CoMaPOI -> This Variant)

- `Profiler` (kept): converts request/context into:
  - long-term profile (`P_u`)
  - short-term intent pattern (`M_u`)
- `Forecaster` (kept): refines retrieval candidates into:
  - long-term candidate set (`C_H`)
  - short-term candidate set (`C_C`)
- `Predictor` (kept): integrates `P_u`, `M_u`, `C_H`, `C_C` and outputs final ranking.

Removed from original paper:

- RRF / LoRA / model fine-tuning.

## 2) Project Layout

- `src/`: generic components
  - `config.py`: `.env` settings
  - `schemas.py`: shared dataclasses
  - `openai_client.py`: OpenAI chat + embeddings
  - `yelp_loader.py`: Yelp JSONL loader
  - `retrieval.py`: retrieval and score fusion
  - `request_simulator.py`: generate time+location query samples
- `utils/`
  - `geo.py`: distance utility
- `CoMaPOI_styled/`: CoMaPOI-specific system
  - `prompts.py`: structured prompts
  - `agents.py`: profiler/forecaster/predictor
  - `pipeline.py`: orchestration
  - `run_query_reco.py`: CLI entry

## 3) Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env` from `.env.example`, then fill values:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `OPENAI_EMBED_MODEL`
- Yelp paths (`YELP_BUSINESS_JSON`, optional city filter)

## 4) Yelp Data Placement

Put Yelp dataset files under a local path, for example:

- `data/yelp/yelp_academic_dataset_business.json`
- `data/yelp/yelp_academic_dataset_review.json` (optional for future extensions)

Current pipeline uses business file directly and builds embedding cache at:

- `data/cache/yelp_business_embeddings.jsonl`

## 5) Run Example

```bash
python CoMaPOI_styled/run_query_reco.py ^
  --query "Need a quiet cafe with good wifi for 2 hours" ^
  --city "Las Vegas" ^
  --state "NV" ^
  --lat 36.1147 ^
  --lon -115.1728 ^
  --local-time "2026-03-09T14:10" ^
  --long-term-notes "Prefers coffee shops and light meals." ^
  --recent-activity-notes "Just finished lunch in downtown area."
```

The output includes:

- intermediate Profiler/Forecaster outputs
- final ranked recommendations with reasons

## 6) Notes

- Structured prompts are enforced in every agent with JSON-only output contracts.
- This design is intentionally split so you can later plug in your own non-CoMaPOI system for fair comparison.
