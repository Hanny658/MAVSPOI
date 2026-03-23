# Recall+Rerank (Non-LLM) Baseline

## Goal
Provide a fully non-LLM baseline with the same runtime/eval interfaces as existing methods:
- shared retrieval module (`src/retrieval.py`)
- same query/context schemas
- same recommendation output shape
- same evaluation metrics and protocol compatibility

## Method
1. **Recall stage**: reuse FAISS retrieval + retrieval score fusion from `src/retrieval.py`.
2. **Rerank stage**: linear listwise ranker trained by **cross-entropy over candidate sets**.
   - For each query, candidate logits are produced by a linear scorer.
   - Softmax over the candidate set.
   - Loss is `-log p(gt_business_id)`.

## Main Files
- `Recall_Rerank/pipeline.py`: runtime pipeline (`recommend` / `recommend_with_candidates`)
- `Recall_Rerank/features.py`: deterministic non-LLM feature extraction
- `Recall_Rerank/model.py`: listwise CE model + trainer + model IO
- `Recall_Rerank/train_listwise_ce.py`: model training entrypoint
- `Recall_Rerank/run_query_reco.py`: single-query inference CLI
- `Recall_Rerank/run_eval.py`: batch eval CLI

## Commands
Train model:
```bash
python -m uv run Recall_Rerank/train_listwise_ce.py \
  --train-queries data/eval/yelp-indianapolis-eval-queries.jsonl \
  --train-candidates data/eval/yelp-indianapolis-eval-candidates.jsonl \
  --mode constrained \
  --model-out Recall_Rerank/models/listwise_ce_model.json
```

Single query:
```bash
python -m uv run Recall_Rerank/run_query_reco.py \
  --query "Need a quiet cafe with wifi for work near me" \
  --user-id "<uid>" \
  --city Indianapolis \
  --state IN \
  --lat 39.7684 \
  --lon -86.1581
```

Batch eval:
```bash
python -m uv run Recall_Rerank/run_eval.py \
  --eval-queries data/eval/yelp-indianapolis-eval-queries.jsonl \
  --eval-candidates data/eval/yelp-indianapolis-eval-candidates.jsonl \
  --mode constrained \
  --k-values 1,5,10
```

## Output Contract
The runtime output keeps the same top-level fields:
- `task`, `system`
- `context`, `enriched_context`
- `retrieval_mode`
- `user_profile_features`
- `intermediate` (retrieval/rerank stats)
- `recommendations` (business + ranking_score + reason + fit_tags + retrieval_components)
- `summary`

