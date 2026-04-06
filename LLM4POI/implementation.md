# LLM4POI (API-LLM Adaptation)

This folder reimplements the core LLM4POI idea from
"Large Language Models for Next Point-of-Interest Recommendation" (SIGIR 2024),
adapted to this repository's query-based evaluation interface.

## What is preserved from the paper

- Trajectory prompting with block structure:
  - current trajectory block
  - historical trajectory block
  - instruction/question block
- Key-query trajectory similarity:
  - key: current trajectory prompt (without the last check-in sentence when possible)
  - query: full historical trajectory prompts
  - top-k similar trajectories injected into historical block
- Three paper variants:
  - `llm4poi_star`  : current trajectory only (no history)
  - `llm4poi_star2` : history from current user only, no key-query similarity
  - `llm4poi`       : full variant with key-query similarity and cross-user history

## What is changed (requested adaptation)

- The paper's supervised fine-tuned local model is replaced by API LLM inference.
- Similarity encoder is implemented with embedding API vectors
  (instead of hidden states from a fine-tuned Llama backbone).
- Output is constrained to the candidate pool from this repo's retriever/eval protocol.

## Prompt-engineering approximation to fine-tuning

To better approximate the original fine-tuned behavior, the runtime applies:

- Paper-style QA prompt blocks (`<question>/<answer>` semantics).
- Dynamic in-context few-shot examples from historical trajectories.
- Two-stage prompting:
  - Stage-1: predict single next POI id (QA-style).
  - Stage-2: rerank candidate POIs with stage-1 as prior.
- Light post-hoc calibration blending stage-2 confidence with retrieval prior.

## Runtime files

- `LLM4POI/pipeline.py`: main runtime implementation (`LLM4POIRealtimeRecommender`)
- `LLM4POI/run_query_reco.py`: single-query CLI
- `LLM4POI/run_eval.py`: batch evaluation CLI

## Example commands

Single query:

```bash
python -m uv run LLM4POI/run_query_reco.py \
  --query "Need a quiet cafe with wifi near me" \
  --user-id "<uid>" \
  --city Indianapolis \
  --state IN \
  --lat 39.7684 \
  --lon -86.1581 \
  --variant llm4poi
```

Batch eval (candidate constrained):

```bash
python -m uv run LLM4POI/run_eval.py \
  --eval-queries data/eval/yelp-indianapolis-eval-queries.jsonl \
  --eval-candidates data/eval/yelp-indianapolis-eval-candidates.jsonl \
  --mode constrained \
  --k-values 1,5,10 \
  --variant llm4poi
```
