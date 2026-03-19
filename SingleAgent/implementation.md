# SingleAgent Baseline Implementation

## 1. Goal
This baseline implements a **single-agent recommendation system** with the same:
- dataset format
- train/eval protocol
- retrieval and evaluation interfaces

as the current CoMaPOI-styled pipeline, so results are directly comparable.

## 2. Design Overview
SingleAgent still follows a complete recommendation chain:
1. `RAG retrieval` from POI corpus (FAISS + score fusion)
2. `LLM reranking` by one unified agent
3. output final Top-K list with reasons

Difference from CoMaPOI-styled:
- No multi-agent decomposition (`Profiler/Forecaster/Predictor`).
- A single LLM agent performs final ranking over retrieved candidates.

## 3. Files
- `SingleAgent/pipeline.py`: end-to-end runtime pipeline
- `SingleAgent/agents.py`: single-agent reranker + optional planning
- `SingleAgent/prompts.py`: structured prompts and JSON schemas
- `SingleAgent/run_query_reco.py`: single-query CLI
- `SingleAgent/run_eval.py`: batch evaluation CLI

## 4. Data and Evaluation Compatibility
This implementation reuses:
- `src/config.py`
- `src/retrieval.py`
- `src/yelp_loader.py`
- `src/profile_loader.py`
- `src/schemas.py`

Default runtime data:
- business: `data/train/yelp-indianapolis-train-business.jsonl`
- profile: `data/train/yelp-indianapolis-train-profile.jsonl`

Evaluation inputs:
- queries: `data/eval/<prefix>-eval-queries.jsonl`
- candidates: `data/eval/<prefix>-eval-candidates.jsonl`

## 5. Retrieval + Reranking Flow
## 5.1 Retrieval stage
`SingleAgentRealtimeRecommender` calls `CandidateRetriever`:
- `full_corpus` mode: retrieve from full train business corpus
- `candidate_constrained` mode: retrieve only from provided candidate IDs

Retrieval uses the same FAISS and score fusion as CoMaPOI-styled for fairness.

## 5.2 Profile-aware context enrichment
Before reranking, context is enriched with compact profile signals:
- support level and interaction evidence
- top categories
- price/radius tendencies
- temporal activity hints

This keeps the same profile signal source as CoMaPOI-styled.

## 5.3 Single-agent reranking
`SingleRecommenderAgent` receives:
- enriched context
- user profile features
- rerank candidate pool (from retriever)

It returns JSON:
- ordered `recommendations`
- `final_summary`

Hard safety constraints:
- only candidate business IDs are accepted
- score is clipped into `[0, 1]`
- retrieval-order fallback is used on malformed/empty LLM output

## 6. Optional CoT Modes
Supported by `--cot-mode`:
- `off`: single pass ranking prompt
- `embedded`: single pass, prompt asks model to reason internally before output
- `two_pass`: first call builds concise planning JSON, second call ranks with that plan

`two_pass` is implemented in `SingleAgent/agents.py` with:
1. planner prompt (`planner_system_prompt`)
2. recommender prompt (`recommender_system_prompt`)

Both passes enforce JSON output contracts.

## 7. Output Contract
The runtime output includes:
- `task`, `system`, `cot_mode`
- `context`, `enriched_context`
- `retrieval_mode`
- `user_profile_features`
- `intermediate` (retrieval/rerank pool size and optional planning object)
- `recommendations` with retrieval component details
- `summary`

This structure is intentionally close to CoMaPOI-styled output for downstream comparison.

## 8. Commands
Single-query inference:
```bash
python -m uv run SingleAgent/run_query_reco.py --query "Need coffee near me" --user-id "<uid>" --city Indianapolis --state IN --lat 39.7684 --lon -86.1581 --cot-mode off
```

Batch eval (candidate constrained):
```bash
python -m uv run SingleAgent/run_eval.py --eval-queries data/eval/yelp-indianapolis-eval-queries.jsonl --eval-candidates data/eval/yelp-indianapolis-eval-candidates.jsonl --mode constrained --k-values 1,5,10 --cot-mode off
```

Batch eval (full corpus / max querie smoke test):
```bash
python -m uv run SingleAgent/run_eval.py --eval-queries data/eval/yelp-indianapolis-eval-queries.jsonl --mode full --k-values 1,5,10 --cot-mode two_pass --max-queries 50
```

