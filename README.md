# MAVSPOI

**Mixture-of-Agents Voting Scheme for POI Recommendation that is context-aware**

This repository contains:
- A working **CoMaPOI-styled baseline** (`Profiler -> Forecaster -> Predictor`) on Yelp Open Dataset.
- A documented target architecture for the next stage: **Router + Sparse Voting Agents + Aggregator**.

## 1. Current Status
Current implemented system:
- Frozen LLM via OpenAI API (no fine-tuning).
- FAISS-based retrieval over Yelp businesses.
- Structured JSON agent outputs with fallback guards.
- Offline evaluation pipeline with reproducible holdout protocol.

Baseline runtime modules:
- `CoMaPOI_styled/pipeline.py`
- `CoMaPOI_styled/agents.py`
- `CoMaPOI_styled/prompts.py`
- `CoMaPOI_styled/run_query_reco.py`
- `CoMaPOI_styled/run_eval.py`

Shared modules:
- `src/retrieval.py`, `src/openai_client.py`, `src/schemas.py`
- `utils/build_eval_protocol.py`, `utils/build_user_profile.py`

## 2. Fair-Comparison Boundary
To ensure fair comparison between baseline and MAVSPOI target architecture, keep these fixed:
- Same train/eval split protocol from `utils/build_eval_protocol.py`.
- Same user-profile construction from `utils/build_user_profile.py`.
- Same candidate protocol (`data/eval/*-eval-candidates.jsonl`).
- Same metrics: `Hit@K`, `Recall@K`, `NDCG@K`, `MRR@K`.

## 3. Target MAVSPOI Architecture

### 3.1 End-to-End Flow
1. Retrieval: FAISS candidate generation (shared).
2. Hard constraints: deterministic filtering before voting.
3. Router Agent: sparse activation of experts.
4. Voting Agents: activated experts score candidates in parallel.
5. Aggregator Agent: deterministic fusion + explanation.

### 3.2 Router Agent
Input:
- Query/session context (time, location, city/state, user id, query text).
- Compact user profile signals.
- Candidate summary statistics.
- Agent registry.

Output:
- Activated agents with weight/confidence.
- Global constraints (`max_distance_km`, `open_now`, city constraints).
- Risk flags (`low_profile_support`, `low_geo_precision`, etc.).

Recommended policy:
- Multi-label sparse activation (`min_agents`, `max_agents`).
- Confidence-threshold activation.
- Low-confidence fallback bundle: `A1 + A3 + A4 + A6`.

### 3.3 Voting Agents (A1-A7)
All agents output normalized candidate scores in `[0,1]` with confidence in `[0,1]`.

- `A1` Spatial Feasibility Expert  
  Spatial reachability and detour cost.

- `A2` Temporal Feasibility Expert  
  Time-window fit, urgency, and period suitability.

- `A3` Intent Matching Expert  
  Short-term query/session intent alignment.

- `A4` Stable Preference Expert  
  Long-term interests and habitual preference alignment.

- `A5` Exploration Expert  
  Novelty/diversity control with repetition penalty.

- `A6` Availability-Reliability Expert  
  Candidate validity, data freshness, cold-start risk handling.

- `A7` Purpose-Modality Expert  
  Purpose fit (social/study/work) and online/offline modality fit.

### 3.4 Unified Voting Output Contract
```json
{
  "agent_id": "A1",
  "results": [
    {
      "business_id": "...",
      "score": 0.73,
      "confidence": 0.88,
      "evidence_tags": ["nearby", "low_detour"],
      "notes": "optional"
    }
  ],
  "agent_status": "ok"
}
```

### 3.5 Aggregator Agent
Primary strategy:
- Deterministic weighted fusion for ranking.
- LLM used mainly for concise explanation and tie rationale.

Reference scoring form:
$$FinalScore(i) = w_r * retrieval(i) + Σ_a [w_a * conf_a(i) * score_a(i)] + bias(i)$$

Where:
- `w_r` = retrieval baseline weight.
- `w_a` = router-provided or configured agent weight.
- `bias(i)` = deterministic correction (for example diversity bonus/repetition penalty).

## 4. Hard Constraints and Safety
Hard checks (recommended strict):
- City/state mismatch removal.
- Max-distance threshold when location is available.
- Open-now strict filtering when reliable hour data is available.

Soft checks (recommended down-weight):
- Missing hours.
- Sparse metadata.
- Cold-start uncertainty.

## 5. Evaluation Plan
Primary metrics:
- `Hit@K`, `Recall@K`, `NDCG@K`, `MRR@K`

Additional diagnostics:
- Per-agent activation rate.
- Per-agent contribution share.
- Router fallback rate.
- Query latency and token cost.
- Slice metrics by `zero_shot/few_shot/warm`.

Suggested ablations:
1. Full MAVSPOI (`Router + A1..A7 + Aggregator`).
2. Router off (all experts always on).
3. Leave-one-out (`-A1` ... `-A7`).
4. Deterministic aggregation vs LLM-heavy aggregation.

## 6. Data Pipeline

### 6.1 Build city-level subset (optional)
```bash
python utils/extract_by_city.py --city Indianapolis --input-dir data --output-dir data
```

### 6.2 Build eval protocol and train split
```bash
python utils/build_eval_protocol.py --data-prefix yelp-indianapolis --data-dir data --sample-size 500
```

Outputs:
- `data/eval/<prefix>-eval-queries.jsonl`
- `data/eval/<prefix>-eval-candidates.jsonl`
- `data/eval/<prefix>-eval-meta.json`
- `data/train/<prefix>-train-*.jsonl`

### 6.3 Build user profiles
```bash
python utils/build_user_profile.py --data-prefix yelp-indianapolis --data-dir data
```

## 7. Run Baseline Inference
```bash
python CoMaPOI_styled/run_query_reco.py ^
  --query "Need a quiet cafe with good wifi for 2 hours" ^
  --user-id "some_existing_user_id" ^
  --city "Indianapolis" ^
  --state "IN" ^
  --lat 39.7684 ^
  --lon -86.1581
```

## 8. Run Baseline Evaluation
Constrained mode:
```bash
python CoMaPOI_styled/run_eval.py ^
  --eval-queries data/eval/yelp-indianapolis-eval-queries.jsonl ^
  --eval-candidates data/eval/yelp-indianapolis-eval-candidates.jsonl ^
  --mode constrained ^
  --k-values 1,5,10
```

Full-corpus mode:
```bash
python CoMaPOI_styled/run_eval.py ^
  --eval-queries data/eval/yelp-indianapolis-eval-queries.jsonl ^
  --mode full ^
  --k-values 1,5,10
```

## 9. Environment
Create `.env` from `.env.example` and set:
- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `OPENAI_EMBED_MODEL`
- `YELP_BUSINESS_JSON`
- `YELP_PROFILE_JSON`

Optional controls:
- `RETRIEVAL_TOP_K`
- `FORECASTER_TOP_K`
- `FINAL_TOP_K`
- `EMBED_CACHE_PATH`

## 10. Roadmap
1. Freeze Router/Voting/Aggregator JSON contracts.
2. Implement minimum expert set: `A1 + A3 + A4 + A6`.
3. Add `A2 + A5 + A7` and contribution tracing.
4. Run full ablations under the same eval protocol.
5. Report baseline vs MAVSPOI with slice-level analysis.
