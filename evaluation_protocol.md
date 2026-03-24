# Evaluation Protocol (Current Implementation)

This document describes the **current end-to-end evaluation pipeline** used in this repository, including protocol construction, query generation, candidate construction, and metric computation (hard GT + soft GT).

---

## 1. Scope and Entry Points

### Protocol Builder
- `utils/build_eval_protocol.py`
- Responsibilities:
1. Hold out each selected user's latest interaction as GT.
2. Generate cloud-LLM queries (query-based setting).
3. Build candidate sets.
4. Build `ground_truth_soft` (weighted multi-relevance labels).
5. Rebuild train split and train-side user profiles.

### Evaluation Runners
- MAVSPOI: `main.py eval`
- SingleAgent: `SingleAgent/run_eval.py`
- CoMaPOI-styled: `CoMaPOI_styled/run_eval.py`
- Recall+Rerank: `Recall_Rerank/run_eval.py`

All four runners now use aligned ranking metrics:
- Hard metrics: `hit`, `recall`, `ndcg`
- Soft metrics: `ndcg_soft`, `wrecall_soft`
- `mrr` is removed from runtime evaluation summaries.

---

## 2. Data Split and Holdout Protocol

Implemented in `utils/build_eval_protocol.py`.

### Inputs
Expected source files (under `data/` by prefix):
- `<prefix>-business.jsonl`
- `<prefix>-review.jsonl`
- `<prefix>-tip.jsonl`
- `<prefix>-user.jsonl`
- `<prefix>-checkin.jsonl`

Default prefix is `yelp-indianapolis`.

### Holdout Rule
For each user:
1. Merge valid events from `review` and `tip`.
2. Keep the latest timestamped interaction as held-out GT event.
3. Remaining events are used to define support level and train profile.

Support levels after holdout:
- `zero_shot`: remaining visits <= 0
- `few_shot`: 1..10
- `warm`: >10

### Sampling
- Controlled by `--sample-size` (`0` means all eligible users).
- If `--balance-support` is enabled (default), sampled users are balanced across:
  - `zero_shot`, `few_shot`, `warm`.

---

## 3. Query Generation (Cloud LLM, Query-Based)

Implemented by `CloudLLMQueryGenerator` in `utils/build_eval_protocol.py`.

### API / Model
- API key loaded from environment:
  - `OPENAI_API_KEY` (or fallback `OPENAI_KEY`)
- Model loaded from:
  - `OPENAI_MODEL` (default `gpt-4.1-mini`)
- Optional base URL:
  - `OPENAI_BASE_URL`

### Query Family Mix
Target ratio:
- `explicit_category`: 15%
- `soft_constraint`: 35%
- `generic_need`: 50%

Family semantics:
1. `explicit_category`:
   - Can inject GT category hint.
   - Should explicitly mention category intent.
2. `soft_constraint`:
   - Can use GT-derived latent traits (e.g., spicy/cheap/cozy).
   - Must avoid explicit cuisine/category mentions.
3. `generic_need`:
   - No GT injection.
   - Broad, ambiguous food-seeking intent.

### Per-query context perturbation
For realism:
- Query time = held-out interaction time minus random 5~30 minutes.
- Query location = random perturbation of GT POI location (100~500 meters).

---

## 4. Candidate Set Construction

Implemented in `make_candidate_set(...)` (`utils/build_eval_protocol.py`).

Given GT business and target candidate size:
1. Start with GT item.
2. Add hard negatives:
   - Prefer same city + overlapping categories.
   - Count controlled by `hard_negative_ratio`.
3. Fill with medium negatives:
   - Same city.
4. Fill remaining with global random negatives.
5. Shuffle final candidate list.

Outputs are written to:
- `data/eval/<prefix>-eval-candidates.jsonl`

---

## 5. Soft Ground Truth (`ground_truth_soft`)

Implemented in `build_soft_ground_truth(...)` (`utils/build_eval_protocol.py`).

Each query now includes:
- `ground_truth` (single anchor GT)
- `ground_truth_soft` (weighted multi-item relevance set)
- `ground_truth_soft_scope` (`candidate` or `full`)

Soft GT scope is controlled by protocol-builder arguments:
- `--soft-gt-scope`:
  - `candidate`: build soft GT only from constrained candidate set.
  - `full` (default): build soft GT from full-corpus pool.
- `--soft-gt-full-max-businesses`:
  - optional cap for full-scope pool size (`0` means all businesses).

`ground_truth_soft` schema:
```json
{
  "version": "soft_gt_v1",
  "family": "soft_constraint",
  "constraint_signals": ["spicy", "nearby"],
  "items": [
    {"business_id": "...", "relevance": 1.0},
    {"business_id": "...", "relevance": 0.73}
  ]
}
```

### Relevance Components
For each candidate item, relevance is computed from 4 parts:
1. **Intent consistency**
2. **Substitutability**
3. **Constraint satisfaction**
4. **Exposure correction**

Raw score:
- `raw = w_intent * intent + w_sub * sub + w_cons * cons + w_exp * exp`
- GT anchor gets an extra `+0.08` bias and is forced to relevance `1.0`.

### Family-specific weights
- `explicit_category`:  
  - intent `0.45`, substitutability `0.30`, constraint `0.20`, exposure `0.05`  
  - `min_relevance=0.20`, `max_items=15`
- `soft_constraint`:  
  - intent `0.30`, substitutability `0.25`, constraint `0.35`, exposure `0.10`  
  - `min_relevance=0.18`, `max_items=20`
- `generic_need`:  
  - intent `0.15`, substitutability `0.40`, constraint `0.15`, exposure `0.30`  
  - `min_relevance=0.12`, `max_items=25`

### Candidate-level component notes
- Intent consistency:
  - Query-token overlap with business name/category tokens.
- Substitutability:
  - Category Jaccard similarity + geo similarity + star similarity.
- Constraint satisfaction:
  - Detect constraint signals in query text (budget/spicy/quick/healthy/cozy/hearty/dessert/caffeine/highly_rated/nearby).
  - Evaluate item against signal-specific rules.
- Exposure correction:
  - Uses normalized `log(review_count)` inverse to encourage long-tail alternatives.

---

## 6. Evaluation Modes

All evaluators support:
- `--mode constrained`
- `--mode full`

### Constrained Mode
- Uses candidate list from `eval-candidates.jsonl`.
- Ranking happens only within candidate subset.

### Full Mode
- No candidate-file restriction at runtime.
- Recommender ranks from full retrieval corpus.

---

## 7. Metrics (Current)

### Hard Metrics (single anchor GT)
At each `K`:
- `hit@K`: whether anchor GT appears in top-K.
- `recall@K`: same as hit under single-GT protocol.
- `ndcg@K`: `1/log2(rank+1)` if hit, else `0`.

### Soft Metrics (weighted multi-GT)
At each `K`:
- `ndcg_soft@K`:
  - DCG uses graded gain `(2^rel - 1)/log2(rank+1)`.
  - IDCG is computed from ideal sorting of `ground_truth_soft.items`.
- `wrecall_soft@K`:
  - Sum relevance covered in top-K divided by total relevance mass in soft GT.

### Backward compatibility
If a query row has no `ground_truth_soft`:
- Evaluator falls back to `{ground_truth.business_id: 1.0}`.

---

## 8. Output Files

Protocol builder output (`data/eval/`):
- `<prefix>-eval-queries.jsonl`
- `<prefix>-eval-candidates.jsonl`
- `<prefix>-eval-meta.json`

Each evaluator prints one JSON summary with:
- `overall.metrics[K]`
- `by_support_level`
- optional per-method extras:
  - CoMaPOI-styled includes `candidate_stage_hit`

All evaluators support optional prediction dump:
- `--save-predictions <path>`

---

## 9. Progress and Runtime UX

Evaluation loops now provide streaming progress bars (stderr):
- processed/total
- percentage
- queries/sec
- ETA

---

## 10. Example Commands

### Build protocol
```bash
python utils/build_eval_protocol.py --data-prefix yelp-indianapolis --sample-size 500 --candidate-size 100 --hard-negative-ratio 0.5 \
--balance-support
```

### Evaluate MAVSPOI
```bash
python main.py eval \
  --eval-queries data/eval/yelp-indianapolis-eval-queries.jsonl \
  --eval-candidates data/eval/yelp-indianapolis-eval-candidates.jsonl \
  --mode constrained \
  --k-values 1,5,10
```

### Evaluate baselines
```bash
python SingleAgent/run_eval.py --mode constrained --k-values 1,5,10
python CoMaPOI_styled/run_eval.py --mode constrained --k-values 1,5,10
python Recall_Rerank/run_eval.py --mode constrained --k-values 1,5,10
```
