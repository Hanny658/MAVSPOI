# CoMaPOI-styled Reimplementation Notes

## 1. Scope and Positioning
This project is a **CoMaPOI-style variant** for `Query-based real-time POI Recommendation` on Yelp data.

What is preserved from CoMaPOI:
- 3-agent decomposition: `Profiler -> Forecaster -> Predictor`
- Two-layer preference modeling idea: long-term + short-term
- Candidate refinement before final ranking
- Adapted trajectory-style tool summaries (`freq/cat/time/loc/poi`) for Profiler prompts
- Candidate-stage observability (`initial/CH/CC/merged` hit-rate in eval output)

What is intentionally changed:
- No trajectory prediction objective
- No LoRA / no fine-tuning (frozen LLM via API)
- Yelp Open Dataset instead of Foursquare

Core runtime dependency is reduced to:
- `business` data
- `user profile` data

## 2. Directory and File Conventions
### 2.1 Runtime modules
- `CoMaPOI_styled/pipeline.py`: end-to-end orchestration
- `CoMaPOI_styled/agents.py`: Profiler/Forecaster/Predictor logic + fallback guards
- `CoMaPOI_styled/prompts.py`: structured prompts (JSON schema constrained)
- `CoMaPOI_styled/run_query_reco.py`: single-query CLI
- `CoMaPOI_styled/run_eval.py`: batch evaluation CLI

### 2.2 Generic modules
- `src/config.py`: `.env` loading and defaults
- `src/openai_client.py`: embeddings + JSON chat wrapper
- `src/retrieval.py`: FAISS-backed retrieval and score fusion
- `src/yelp_loader.py`: business loader
- `src/profile_loader.py`: profile loader
- `src/schemas.py`: dataclasses (`UserQueryContext`, `BusinessPOI`, `CandidateScore`)

### 2.3 Data build scripts
- `utils/extract_by_city.py`: city-consistent extraction
- `utils/build_eval_protocol.py`: holdout-based eval/train split generation
- `utils/build_user_profile.py`: deterministic profile building (no LLM)

### 2.4 Current processed data layout
- Train split: `data/train/<prefix>-train-*.jsonl`
- Eval split: `data/eval/<prefix>-eval-*.jsonl`
- Embedding index: `data/cache/*.faiss` + `*.faiss.ids.json`

## 3. Data Processing Pipeline
## 3.1 City-consistent subset extraction
Script: `utils/extract_by_city.py`

Input (raw Yelp files in `data/`):
- `yelp_academic_dataset_business.json`
- `yelp_academic_dataset_review.json`
- `yelp_academic_dataset_user.json`
- `yelp_academic_dataset_checkin.json`
- `yelp_academic_dataset_tip.json`

Process:
1. Filter `business` by exact city match.
2. Keep `review/tip` rows whose `business_id` is in selected city businesses.
3. Build `user_id` union from kept `review/tip`.
4. Filter `user` by this union.
5. Filter `checkin` by selected city businesses.

Output:
- `data/<prefix>-business.jsonl`
- `data/<prefix>-review.jsonl`
- `data/<prefix>-user.jsonl`
- `data/<prefix>-checkin.jsonl`
- `data/<prefix>-tip.jsonl`
- `data/<prefix>-meta.json`

Consistency property:
- All interactions reference in-city businesses.
- User file is aligned with review/tip evidence.

## 3.2 Eval protocol + train holdout generation
Script: `utils/build_eval_protocol.py`

Goal:
- Build fair offline protocol for real-time query recommendation.

Main logic:
1. For each user, find latest visit event from `review + tip`.
2. Hold out this latest event as GT.
3. Build synthetic query by:
- rolling back time by `5-30` minutes
- random location perturbation `100-500m`
- query text by template or optional local LLM
4. Build per-query candidate set with GT included:
- hard negatives: same city/category priority
- medium negatives: same city
- easy negatives: random global fill
5. Write eval files to `data/eval`.
6. Remove held-out interaction rows and write train files to `data/train`.
7. Re-run profile builder on the updated train split.

Important behavior:
- Users with no valid visit are not used.
- Users with one visit become `zero_shot` after holdout.
- Default sampling is `--sample-size 500` to control evaluation size.

## 3.3 User profile building
Script: `utils/build_user_profile.py`

Input resolution rule:
- Prefer `*.jsonl`, fallback to `*.json`.

Accepted sources:
- `business/review/tip/user/checkin`

Output:
- `data/<prefix>-profile.jsonl`

Per-user profile fields:
- `support`: support level + evidence counts
- `coverage`: interaction/checkin coverage
- `static`: user-side static attributes
- `rating_behavior`: rating stats + social feedback
- `temporal_pref`: hour/weekday behavior
- `category_pref`: top categories + diversity
- `price_pref`: price distribution + dominant level
- `spatial_pref`: activity center + mobility radius
- `business_quality_pref`: quality tendency of visited places
- `checkin_context_pref`: checkin context aggregated from visited businesses

Current support bucket rule:
- `zero_shot`: interaction count `<= 0`
- `few_shot`: interaction count `1-10`
- `warm`: interaction count `>= 11`

## 4. Runtime Recommendation Pipeline
Entry orchestrator: `CoMaPOI_styled/pipeline.py`

Initialization:
1. Load settings from `.env` (`src/config.py`).
2. Load businesses (`src/yelp_loader.py`).
3. Load profiles (`src/profile_loader.py`).
4. Initialize `CandidateRetriever` (`src/retrieval.py`).
5. Initialize three agents (`ProfilerAgent`, `ForecasterAgent`, `PredictorAgent`).

Context enrichment:
- Runtime query context is merged with compact profile-derived hints:
- long-term notes: support level, top categories, price/radius tendencies
- short-term hints: active hours, weekend ratio, checkin tendency

Two retrieval modes:
- `full_corpus`: search all train businesses
- `candidate_constrained`: search only inside provided candidate IDs

## 4.1 FAISS retrieval implementation details
Module: `src/retrieval.py`

Embedding/index lifecycle:
1. Encode business retrieval text with OpenAI embedding model.
2. Normalize vectors (L2) and store in `FAISS IndexFlatIP`.
3. Persist index to `*.faiss`.
4. Persist ordered business ID mapping to `*.faiss.ids.json`.
5. Incremental append when new businesses are missing from existing index.

Query scoring:
- Semantic similarity from FAISS inner product.
- Popularity score:
`0.5 * (stars / 5) + 0.5 * log1p(review_count) / log1p(max_reviews)`
- Geo score (if query has location):
`exp(-distance_km / 8.0)`

Final fusion:
- With location:
`0.65 * semantic + 0.20 * geo + 0.15 * popularity`
- Without location:
`0.80 * semantic + 0.20 * popularity`
- Plus city bonus `+0.05` when business city equals query city.

## 4.2 Agent-level implementation details
Module: `CoMaPOI_styled/agents.py`

All agents:
- Receive structured JSON payload.
- Must output JSON.
- Have robust fallback defaults if model fails or outputs invalid structure.

Profiler:
- Input: context + compact profile + retrieval snapshot
- Input also includes adapted tool summaries approximating trajectory statistics
- Output:
- `long_term_profile`
- `short_term_pattern`
- `hard_constraints`

Forecaster:
- Input: context + profiler output + initial candidates
- Input also includes `short_term_initial_candidate_ids` (adapted `C_C,init`)
- Output:
- `long_term_candidate_ids` (`C_H`)
- `short_term_candidate_ids` (`C_C`)
- `merged_candidate_ids`
- `rerank_rationale`
- Safety: candidate IDs are sanitized to allowed retrieval IDs only.

Predictor:
- Input: context + profiler output + forecaster output + final candidate list
- Input also includes a broader candidate universe for optional out-of-set fallback
- Output:
- final ordered recommendations (`business_id`, `score`, `reason`, `fit_tags`)
- `final_summary`
- Safety: keeps IDs in allowed set by default; optional low-confidence out-of-set fallback is constrained to candidate universe and penalized.

Prompting strategy:
- `CoMaPOI_styled/prompts.py` defines strict JSON schema contracts for each agent.
- This is the main structured-prompt optimization layer in current implementation.

## 5. Evaluation and Metrics
Script: `CoMaPOI_styled/run_eval.py`

Inputs:
- `data/eval/<prefix>-eval-queries.jsonl`
- `data/eval/<prefix>-eval-candidates.jsonl` (for constrained mode)

Supports:
- `--mode constrained`: use candidate-constrained retrieval
- `--mode full`: full-corpus retrieval

Metrics:
- `Hit@K`
- `Recall@K` (single-GT setting equals Hit@K)
- `NDCG@K`
- `MRR@K`

Reports:
- overall metrics
- `by_support_level` metrics (`zero_shot/few_shot/warm`) from eval query slice metadata

## 6. Current Design Limits
- `hard_constraints` are now applied with runtime filtering and a relaxed fallback when filtering is over-aggressive.
- Open-hour and distance constraints are still lightweight and do not use a full business-hour engine from raw hour ranges.
- In full mode, performance may look high on smaller/easier sampled subsets; use larger samples and harder candidates for robust comparison.

## 7. Reproducible Command Sequence (uv)
1. City extraction
```bash
python -m uv run utils/extract_by_city.py --city Indianapolis --input-dir data --output-dir data
```

2. Build eval protocol + train holdout + rebuilt train profile
```bash
python -m uv run utils/build_eval_protocol.py --data-prefix yelp-indianapolis --data-dir data --sample-size 500
```

3. Single-query inference
```bash
python -m uv run CoMaPOI_styled/run_query_reco.py --query "Need coffee near me" --user-id "<uid>" --city Indianapolis --state IN --lat 39.7684 --lon -86.1581
```

4. Batch evaluation (constrained)
```bash
python -m uv run CoMaPOI_styled/run_eval.py --eval-queries data/eval/yelp-indianapolis-eval-queries.jsonl --eval-candidates data/eval/yelp-indianapolis-eval-candidates.jsonl --mode constrained --k-values 1,5,10
```

## 8. Why This Structure Is Useful for Future Comparison
The repo keeps:
- generic utilities in `src/` and `utils/`
- CoMaPOI-specific logic in `CoMaPOI_styled/`

So future architectures can reuse the same:
- train/eval split
- user profile definition
- candidate protocol
- metrics code

This isolates architecture differences and keeps comparison fair.
