# MAVSPOI

**MAVSPOI = Mixture-of-Agents Voting Scheme for context-aware query-based POI Recommendation**

This repo now contains:
- A preserved **CoMaPOI-styled baseline** (`CoMaPOI_styled/*`).
- A modular **MAVSPOI implementation in `src/`** with:
  - Hybrid Calibration Router
  - Chain-of-thought (internal) Voting Agents A1~A7
  - Parallel voting execution
  - Deterministic Aggregator

## 1. What Is Implemented Now

### 1.1 MAVSPOI Runtime (src-level)
- `src/mavspoi_pipeline.py`: end-to-end orchestration
- `src/agents/router_agent.py`: Hybrid Calibration Router
- `src/agents/agent_registry.py`: voting-agent registration for Router/Aggregator
- `src/agents/aggregator_agent.py`: deterministic score fusion
- `src/agents/llm_voting_base.py`: shared CoT voting base
- `src/agents/*_agent.py`: A1~A7 expert modules

### 1.2 Shared Components Reused
- `src/retrieval.py`: FAISS retrieval + retrieval score fusion
- `src/openai_client.py`: OpenAI chat JSON + embeddings
- `src/profile_loader.py`, `src/yelp_loader.py`, `src/schemas.py`
- `utils/build_eval_protocol.py`, `utils/build_user_profile.py`

### 1.3 Main Entrypoint
- `main.py` provides:
  - `query` subcommand
  - `eval` subcommand

## 2. Architecture (Current)

### 2.1 End-to-End Flow
1. Retrieval from FAISS (`src/retrieval.py`)
2. Router decides active experts with Hybrid Calibration
3. Activated voting experts run in parallel
4. Aggregator fuses retrieval + votes into final ranking

### 2.2 Router: Hybrid Calibration
Router combines:
- heuristic routing score
- LLM routing score (optional, enabled by default)
- prior from configured agent weights

It calibrates mixture weights by:
- profile reliability (`warm` / `few_shot` / `zero_shot`)
- query clarity (token richness)

Output:
- `activated_agents` with calibrated weight/confidence
- `global_constraints` (`city/state/open_now/max_distance_km`)
- `risk_flags`

### 2.3 Voting Agents: CoT Reasoning + Calibration
All A1~A7 experts use:
- LLM internal chain-of-thought style reasoning (hidden, JSON output only)
- heuristic fallback scoring
- per-agent blend of LLM score and heuristic score

Experts:
- A1 Spatial Feasibility
- A2 Temporal Feasibility
- A3 Intent Matching
- A4 Stable Preference
- A5 Exploration
- A6 Availability-Reliability
- A7 Purpose-Modality

### 2.4 Parallel Voting
Voting is executed with a thread pool in `src/mavspoi_pipeline.py`:
- configurable `parallel_workers`
- per-agent failure isolation (failed agent does not break full request)

### 2.5 Aggregator
Deterministic weighted fusion:
- retrieval component
- activated expert contributions (`weight * confidence * score`)
- optional diversity penalty via greedy rerank

## 3. Agent Registry

`src/agents/agent_registry.py` is the canonical registry.

It:
- registers all voting agents
- exposes specs to Router
- provides agent lookup for voting execution

Compatibility file for requested naming:
- `src/agents/agent-registry.py`

## 4. Config

Settings are read from `config.yaml` (`runtime` + `mavspoi` sections), with optional env override for runtime keys.

Key MAVSPOI controls:
```yaml
mavspoi:
  router:
    enabled_agents: [A1, A2, A3, A4, A5, A6, A7]
    min_agents: 3
    max_agents: 5
    activation_threshold: 0.45
    fallback_agents: [A1, A3, A4, A6]
    default_max_distance_km: 10.0
    use_llm: true
    llm_temperature: 0.1
    hybrid_heuristic_base: 0.45
    hybrid_llm_base: 0.45
    hybrid_prior_base: 0.10

  voting:
    candidate_pool_size: 30
    parallel_workers: 7
    llm_enabled: true
    llm_temperature: 0.1
    llm_max_tokens: 1600
    llm_candidate_limit: 30
    llm_weight: 0.65
    heuristic_weight: 0.35

  aggregator:
    retrieval_weight: 0.25
    diversity_penalty: 0.04
    weights:
      A1: 0.16
      A2: 0.14
      A3: 0.20
      A4: 0.17
      A5: 0.09
      A6: 0.15
      A7: 0.09
```

Optional env var:
- `CONFIG_YAML_PATH=...` to point to another YAML file.

## 5. Run

### 5.1 Single Query
```bash
python main.py query ^
  --query "Need a quiet cafe with wifi for working near me" ^
  --user-id "<uid>" ^
  --city "Indianapolis" ^
  --state "IN" ^
  --lat 39.7684 ^
  --lon -86.1581
```

### 5.2 Batch Eval
Constrained mode:
```bash
python main.py eval --mode constrained --eval-queries data/eval/yelp-indianapolis-eval-queries.jsonl --eval-candidates data/eval/yelp-indianapolis-eval-candidates.jsonl --k-values 1,5,10
```

Full mode (--max-queries for smoke tests):
```bash
python main.py eval --mode full --eval-queries data/eval/yelp-indianapolis-eval-queries.jsonl --k-values 1,5,10 --max-queries 50
```

## 6. Evaluation Compatibility

Fair-comparison assets remain unchanged:
- Train/eval split protocol
- Candidate generation protocol
- Profile construction protocol
- Metrics (`Hit@K`, `Recall@K`, `NDCG@K`, `MRR@K`)

So results are directly comparable against `CoMaPOI_styled/*` and `SingleAgent/*` baselines.
