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
  - `profile_loader.py`: load per-user profiles from JSONL
  - `yelp_loader.py`: Yelp JSONL loader
  - `retrieval.py`: retrieval and score fusion
  - `request_simulator.py`: generate time+location query samples
- `utils/`
  - `geo.py`: distance utility
  - `extract_by_city.py`: extract a city-level consistent Yelp subset
  - `build_user_profile.py`: build per-user profile JSONL from subset files
  - `build_eval_protocol.py`: build held-out test queries/candidates and train split
- `CoMaPOI_styled/`: CoMaPOI-specific system
  - `prompts.py`: structured prompts
  - `agents.py`: profiler/forecaster/predictor
  - `pipeline.py`: orchestration
  - `run_query_reco.py`: CLI entry
  - `run_eval.py`: batch evaluation on processed eval files

## 3) Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env` from `.env.example`, then fill values:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `OPENAI_EMBED_MODEL`
- Yelp paths (`YELP_BUSINESS_JSON`, `YELP_PROFILE_JSON`)

## 4) Yelp Data Placement

For the CoMaPOI-styled inference pipeline, only two files are required:

- `data/<prefix>-business.jsonl`
- `data/<prefix>-profile.jsonl`

Current pipeline reads business + profile, and builds embedding cache at:

- `data/cache/yelp_business_embeddings.jsonl`

If you want a city-level subset:

```bash
python utils/extract_by_city.py --city Indianapolis --input-dir data --output-dir data
```

This creates:

- `data/yelp-indianapolis-business.jsonl`
- `data/yelp-indianapolis-review.jsonl`
- `data/yelp-indianapolis-user.jsonl`
- `data/yelp-indianapolis-checkin.jsonl`
- `data/yelp-indianapolis-tip.jsonl`

## 5) Run Example

```bash
python CoMaPOI_styled/run_query_reco.py ^
  --query "Need a quiet cafe with good wifi for 2 hours" ^
  --user-id "some_existing_user_id" ^
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
- compact user profile features injected into the pipeline
- final ranked recommendations with reasons

The pipeline now supports two retrieval modes:

- `full_corpus`: retrieve from all train businesses
- `candidate_constrained`: retrieve only within candidate IDs from eval protocol files

## 6) Build User Profiles (No LLM)

Build deterministic per-user profiles from city subset files:

```bash
python utils/build_user_profile.py --data-prefix yelp-indianapolis --data-dir data
```

Input resolution rules:

- Prefer `data/<data_prefix>-*.jsonl`
- Fallback to `data/<data_prefix>-*.json`

Output:

- `data/<data_prefix>-profile.jsonl`

This file is directly consumed by `CoMaPOI_styled` inference through `YELP_PROFILE_JSON`.

Each profile row includes:

- `support`: explicit `zero_shot` / `few_shot` / `warm` evidence level
- `coverage`: interaction availability and checkin-context availability
- `static`: user metadata (fans, friends, elite years, etc.)
- `rating_behavior`: review-star statistics and feedback signals
- `temporal_pref`: hour/weekday distributions from review+tip timestamps
- `category_pref`: top categories and diversity
- `price_pref`: price-level distribution from business attributes
- `spatial_pref`: center and movement radius statistics
- `business_quality_pref`: visited-business quality statistics
- `checkin_context_pref`: aggregated checkin distributions of visited businesses

This profile design supports fair and quantifiable comparison across:

- `zero_shot` users (no observed interactions)
- `few_shot` users (limited interactions)
- `warm` users (sufficient interaction history)

## 7) Notes

- Structured prompts are enforced in every agent with JSON-only output contracts.
- This design is intentionally split so you can later plug in your own non-CoMaPOI system for fair comparison.

## 8) Build Evaluation Protocol

This pipeline creates a reproducible test protocol and an updated train split:

1. Find each user's latest visit from `review + tip` as ground truth.
2. Hold out that single latest visit from train interactions.
3. Simulate a real-time query by:
   - rolling back time by 5-30 minutes
   - perturbing location by 100-500 meters
4. Build candidate sets for each query with guaranteed GT inclusion.
5. Rebuild train-side user profiles from updated train files.

Run (template query mode, deterministic):

```bash
python utils/build_eval_protocol.py --data-prefix yelp-indianapolis --data-dir data
```

By default this randomly samples `500` users for protocol generation.  
Use `--sample-size 0` for full users, or another number (for example `--sample-size 1000`).
By default outputs go to `data/train/` and `data/eval/`.

Use local small model (llama.cpp OpenAI-compatible server on port 1025):

```bash
python utils/build_eval_protocol.py ^
  --data-prefix yelp-indianapolis ^
  --data-dir data ^
  --sample-size 500 ^
  --query-generator llm ^
  --llm-base-url http://127.0.0.1:1025/v1 ^
  --llm-model local-llm ^
  --llm-temperature 0.7
```

Main outputs:

- `data/eval/<prefix>-eval-queries.jsonl`
- `data/eval/<prefix>-eval-candidates.jsonl`
- `data/eval/<prefix>-eval-meta.json`
- Updated train files:
  - `data/train/<prefix>-train-business.jsonl`
  - `data/train/<prefix>-train-review.jsonl`
  - `data/train/<prefix>-train-tip.jsonl`
  - `data/train/<prefix>-train-user.jsonl`
  - `data/train/<prefix>-train-checkin.jsonl`
  - `data/train/<prefix>-train-profile.jsonl`

Important protocol behavior:

- Users with no valid visits are removed from train user file.
- Users with exactly one visit become `zero_shot` after holdout (kept for evaluation).

## 9) Evaluate CoMaPOI-styled on Processed Dataset

Run constrained evaluation (recommended, uses `data/eval/*` candidates):

```bash
python CoMaPOI_styled/run_eval.py ^
  --eval-queries data/eval/yelp-indianapolis-eval-queries.jsonl ^
  --eval-candidates data/eval/yelp-indianapolis-eval-candidates.jsonl ^
  --mode constrained ^
  --k-values 1,5,10
```

Run full-corpus evaluation:

```bash
python CoMaPOI_styled/run_eval.py ^
  --eval-queries data/eval/yelp-indianapolis-eval-queries.jsonl ^
  --mode full ^
  --k-values 1,5,10
```

You can save per-query predictions:

```bash
python CoMaPOI_styled/run_eval.py ^
  --mode constrained ^
  --save-predictions data/eval/yelp-indianapolis-preds.jsonl
```
