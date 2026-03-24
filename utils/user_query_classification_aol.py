from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

# Canonical labels ordered by information richness (high -> low).
LABEL_CONCISE = "concise requirements"
LABEL_SOFT = "soft preference"
LABEL_ENTITY = "entity lookup"
LABEL_GENERAL = "general search"

TOKEN_PATTERN = re.compile(r"[a-z0-9$%]+")
SPACE_PATTERN = re.compile(r"\s+")

CONCISE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "numeric_budget",
        re.compile(
            r"\b(under|below|less than|at most|no more than|over|above|at least|between)\s+\$?\d+(?:[.,]\d+)?\b"
        ),
    ),
    (
        "price_constraint",
        re.compile(
            r"\$(?:\d+(?:[.,]\d+)?)|\b(price|cost|budget|affordable|cheap|expensive|discount)\b"
        ),
    ),
    (
        "location_constraint",
        re.compile(r"\b(near|nearby|closest|within|miles?|km|in\s+[a-z]{2,}(?:\s+[a-z]{2,}){0,3})\b"),
    ),
    (
        "time_constraint",
        re.compile(
            r"\b(today|tonight|tomorrow|this weekend|weekend|open now|hours|24/7|24 hour|now)\b|\b(19|20)\d{2}\b|\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b"
        ),
    ),
    (
        "attribute_constraint",
        re.compile(
            r"\b(with|without)\s+[a-z0-9-]+|\b(non[- ]?smoking|pet[- ]?friendly|vegan|gluten[- ]?free|wheelchair|for\s+(kids|children|family|beginners|seniors))\b"
        ),
    ),
    (
        "route_constraint",
        re.compile(r"\bfrom\s+[a-z]{2,}(?:\s+[a-z]{2,}){0,2}\s+to\s+[a-z]{2,}(?:\s+[a-z]{2,}){0,2}\b"),
    ),
]

SOFT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "quality_preference",
        re.compile(r"\b(best|better|top|favorite|popular|recommended|famous|quality)\b"),
    ),
    (
        "taste_preference",
        re.compile(r"\b(romantic|cozy|quiet|fun|luxury|premium|authentic|friendly|healthy)\b"),
    ),
    (
        "comparison_intent",
        re.compile(r"\b(vs\.?|versus|compare|comparison|review|reviews|rating|ratings)\b"),
    ),
]

SOFT_BRAND_COLLISIONS = {
    "best western",
    "quality inn",
    "comfort inn",
    "holiday inn",
}

SOFT_CONTEXT_TYPE_PATTERN = re.compile(
    r"\b("
    r"restaurant|restaurants|cafe|coffee|bar|pub|bistro|diner|bakery|pizza|burger|bbq|sushi|ramen|"
    r"steakhouse|buffet|brunch|breakfast|lunch|dinner|"
    r"hotel|motel|inn|hostel|resort|lodging|"
    r"museum|gallery|park|zoo|aquarium|beach|trail|campground|"
    r"cinema|movie theater|theater|"
    r"gym|fitness|yoga|spa|salon|"
    r"mall|shopping center|supermarket|grocery"
    r")\b"
)

GENERAL_OVERRIDES = {
    "best buy",
    "best western",
    "top gun",
    "cheap trick",
    "popular science",
}

POI_TYPE_PATTERN = re.compile(
    r"\b("
    r"restaurant|restaurants|cafe|coffee|bar|pub|bistro|diner|bakery|pizza|burger|bbq|sushi|ramen|"
    r"steakhouse|buffet|brunch|breakfast|lunch|dinner|food|takeout|delivery|"
    r"hotel|motel|inn|hostel|resort|lodging|airbnb|"
    r"museum|gallery|park|zoo|aquarium|beach|trail|campground|"
    r"cinema|movie theater|theater|stadium|arena|"
    r"gym|fitness|yoga|spa|salon|barber|nail salon|massage|"
    r"dentist|doctor|clinic|hospital|pharmacy|veterinary|vet|"
    r"bank|atm|gas station|car wash|auto repair|mechanic|"
    r"mall|shopping center|supermarket|grocery|market|bookstore|store|shop|"
    r"school|college|university|library|church|temple"
    r")\b"
)
POI_LOOKUP_PATTERN = re.compile(
    r"\b(address|phone|phone number|hours|open now|menu|reservation|reservations|booking|ticket|tickets|directions|map|parking)\b"
)
POI_LOCAL_INTENT_PATTERN = re.compile(r"\b(near me|nearby|closest|in [a-z]{2,}(?: [a-z]{2,}){0,3})\b")
POI_BRAND_PATTERN = re.compile(
    r"\b("
    r"mcdonalds|starbucks|subway|kfc|pizza hut|dominos|taco bell|"
    r"walmart|target|costco|ikea|home depot|walgreens|cvs|"
    r"hilton|marriott|hyatt|holiday inn|best western"
    r")\b"
)
AMBIGUOUS_POI_TYPES = {
    "food",
    "school",
    "college",
    "university",
    "library",
    "bank",
    "church",
    "hospital",
    "clinic",
    "doctor",
    "dentist",
    "beach",
    "park",
    "market",
    "store",
    "shop",
}
POI_ENTITY_SUFFIXES = {
    "restaurant",
    "cafe",
    "bar",
    "pub",
    "bakery",
    "hotel",
    "motel",
    "inn",
    "resort",
    "museum",
    "park",
    "zoo",
    "theater",
    "stadium",
    "arena",
    "gym",
    "spa",
    "salon",
    "dentist",
    "doctor",
    "clinic",
    "hospital",
    "pharmacy",
    "bank",
    "store",
    "shop",
    "mall",
    "market",
    "supermarket",
    "school",
    "college",
    "university",
    "library",
    "church",
    "temple",
}
NON_ENTITY_INTENT_TOKENS = {
    "near",
    "nearby",
    "closest",
    "best",
    "better",
    "top",
    "cheap",
    "affordable",
    "hours",
    "menu",
    "reservation",
    "reservations",
    "booking",
    "directions",
    "map",
    "review",
    "reviews",
    "rating",
    "ratings",
    "open",
    "today",
    "tonight",
    "tomorrow",
}
GENERIC_DESCRIPTOR_TOKENS = {
    "new",
    "old",
    "local",
    "public",
    "city",
    "county",
    "state",
    "national",
    "downtown",
    "north",
    "south",
    "east",
    "west",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze AOL Query Log (data/aol-queries.txt by default) and print the "
            "distribution of query information richness."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/aol-queries.txt"),
        help="Path to AOL query log text file (tab-separated).",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding (default: utf-8).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500000,
        help="Print progress every N lines (0 to disable).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Number of sample queries to print for each class.",
    )
    parser.add_argument(
        "--poi-gate",
        choices=("strict", "legacy"),
        default="strict",
        help=(
            "POI intent gate mode. strict=(type/brand) AND (local/lookup) with "
            "extra ambiguity guard; legacy keeps historical broad type matching."
        ),
    )
    parser.add_argument(
        "--deduplicate",
        dest="deduplicate",
        action="store_true",
        help="Deduplicate normalized queries before POI filtering/classification.",
    )
    parser.add_argument(
        "--no-deduplicate",
        dest="deduplicate",
        action="store_false",
        help="Disable deduplication and count each log row.",
    )
    parser.add_argument(
        "--split-entity-lookup",
        dest="split_entity_lookup",
        action="store_true",
        help="Split explicit POI entity-name lookup into a separate category.",
    )
    parser.add_argument(
        "--no-split-entity-lookup",
        dest="split_entity_lookup",
        action="store_false",
        help="Merge entity lookup into concise requirements (legacy behavior).",
    )
    parser.set_defaults(deduplicate=True, split_entity_lookup=True)
    return parser.parse_args()


def normalize_query(text: str) -> str:
    lowered = text.strip().lower()
    return SPACE_PATTERN.sub(" ", lowered)


def iter_queries(path: Path, encoding: str) -> Iterable[str]:
    with path.open("r", encoding=encoding, errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            query = (row.get("Query") or row.get("query") or "").strip()
            if query:
                yield query


def has_soft_poi_context(normalized: str) -> bool:
    # Soft preference should be tied to POI-seeking context rather than generic "best/review".
    return bool(SOFT_CONTEXT_TYPE_PATTERN.search(normalized) or POI_BRAND_PATTERN.search(normalized))


def classify_query(query: str, *, split_entity_lookup: bool = True) -> tuple[str, list[str]]:
    normalized = normalize_query(query)
    if not normalized:
        return LABEL_GENERAL, ["empty"]
    if normalized in GENERAL_OVERRIDES:
        return LABEL_GENERAL, ["general_override"]

    concise_reasons: list[str] = [
        reason for reason, pattern in CONCISE_PATTERNS if pattern.search(normalized)
    ]
    raw_soft_reasons: list[str] = [
        reason for reason, pattern in SOFT_PATTERNS if pattern.search(normalized)
    ]
    if "quality_preference" in raw_soft_reasons and any(c in normalized for c in SOFT_BRAND_COLLISIONS):
        raw_soft_reasons = [reason for reason in raw_soft_reasons if reason != "quality_preference"]
    soft_reasons: list[str] = raw_soft_reasons if has_soft_poi_context(normalized) else []

    tokens = TOKEN_PATTERN.findall(normalized)
    token_count = len(tokens)

    concise_score = len(concise_reasons)
    soft_score = len(soft_reasons)
    is_entity_lookup = is_specific_poi_target(normalized, tokens)

    if is_entity_lookup and not split_entity_lookup:
        concise_score += 1
        concise_reasons.append("specific_poi_target")

    if concise_reasons and token_count >= 6:
        concise_score += 1
        concise_reasons.append("long_query_with_constraints")
    if soft_reasons and token_count >= 4:
        soft_score += 1
        soft_reasons.append("descriptive_preference_phrase")

    if concise_score > 0:
        return LABEL_CONCISE, concise_reasons
    if soft_score > 0:
        return LABEL_SOFT, soft_reasons
    if is_entity_lookup and split_entity_lookup:
        return LABEL_ENTITY, ["specific_poi_target"]
    return LABEL_GENERAL, ["no_preference_signal"]


def is_specific_poi_target(normalized: str, tokens: list[str]) -> bool:
    if not (2 <= len(tokens) <= 8):
        return False
    if any(token in NON_ENTITY_INTENT_TOKENS for token in tokens):
        return False

    # Brand-driven lookup is often an explicit target POI.
    if POI_BRAND_PATTERN.search(normalized) and len(tokens) <= 5:
        return True

    has_entity_suffix = any(token in POI_ENTITY_SUFFIXES for token in tokens)
    if not has_entity_suffix:
        return False

    informative_tokens = [
        token
        for token in tokens
        if token not in POI_ENTITY_SUFFIXES
        and token not in GENERIC_DESCRIPTOR_TOKENS
    ]
    return any(len(token) >= 3 for token in informative_tokens)


def detect_poi_intent(query: str, gate_mode: str = "strict") -> tuple[bool, list[str]]:
    normalized = normalize_query(query)
    if not normalized:
        return False, ["empty"]

    reasons: list[str] = []
    has_type = bool(POI_TYPE_PATTERN.search(normalized))
    has_lookup = bool(POI_LOOKUP_PATTERN.search(normalized))
    has_local = bool(POI_LOCAL_INTENT_PATTERN.search(normalized))
    has_brand = bool(POI_BRAND_PATTERN.search(normalized))

    if has_type:
        reasons.append("poi_type_term")
    if has_lookup:
        reasons.append("poi_lookup_intent")
    if has_local:
        reasons.append("poi_local_intent")
    if has_brand:
        reasons.append("poi_brand")

    if gate_mode == "legacy":
        # Historical broad gate: prone to false positives on generic web queries.
        if has_type:
            return True, reasons
        if has_brand and (has_lookup or has_local):
            return True, reasons
        return False, reasons or ["no_poi_signal"]

    tokens = TOKEN_PATTERN.findall(normalized)
    matched_types = {m for m in POI_TYPE_PATTERN.findall(normalized)}
    has_non_ambiguous_type = bool(matched_types - AMBIGUOUS_POI_TYPES)
    has_local_or_lookup = has_lookup or has_local
    is_entity_target = is_specific_poi_target(normalized, tokens)

    if has_local_or_lookup and (has_type or has_brand):
        if matched_types and matched_types.issubset(AMBIGUOUS_POI_TYPES) and not has_lookup:
            return False, reasons + ["ambiguous_type_without_lookup"]
        reasons.append("strict_gate_pass")
        return True, reasons

    if has_type and has_non_ambiguous_type and is_entity_target:
        reasons.append("strict_gate_entity_pass")
        return True, reasons

    return False, reasons or ["no_poi_signal"]


def print_summary(
    path: Path,
    total_rows: int,
    total_queries: int,
    poi_queries: int,
    label_counts: Counter[str],
    reason_counts: dict[str, Counter[str]],
    samples: dict[str, list[str]],
    poi_keep_reason_counts: Counter[str],
    filtered_out_non_poi: int,
    ordered_labels: list[str],
    deduplicate: bool,
    skipped_duplicates: int,
    poi_gate: str,
) -> None:
    print("=" * 72)
    print("AOL Query Information Richness Analysis")
    print("=" * 72)
    print(f"Input file        : {path}")
    print(f"Rows scanned      : {total_rows:,}")
    print(f"POI gate mode     : {poi_gate}")
    print(f"Deduplicate       : {deduplicate}")
    print(f"Queries analyzed  : {total_queries:,}")
    if deduplicate:
        print(f"Skipped duplicates: {skipped_duplicates:,}")
    poi_ratio = (poi_queries / total_queries) if total_queries else 0.0
    print(f"POI subset size   : {poi_queries:,} ({poi_ratio:.2%} of analyzed queries)")
    print()
    print("Category Distribution")
    print("-" * 72)
    name_width = max(len(label) for label in ordered_labels)

    for label in ordered_labels:
        count = label_counts.get(label, 0)
        ratio_in_poi = (count / poi_queries) if poi_queries else 0.0
        ratio_in_all = (count / total_queries) if total_queries else 0.0
        print(
            f"{label:<{name_width}}  count={count:>10,}  "
            f"ratio_in_poi={ratio_in_poi:>8.2%}  ratio_in_all={ratio_in_all:>8.2%}"
        )

    print()
    print("Top Rule Signals")
    print("-" * 72)
    print("[poi filter]")
    print(f"  kept_poi_queries{'':<22} {poi_queries:>10,}")
    print(f"  filtered_non_poi{'':<21} {filtered_out_non_poi:>10,}")
    for reason, count in poi_keep_reason_counts.most_common(5):
        print(f"  {reason:<34} {count:>10,}")
    for label in ordered_labels:
        top_reasons = reason_counts.get(label, Counter()).most_common(5)
        print(f"[{label}]")
        if not top_reasons:
            print("  (none)")
            continue
        for reason, count in top_reasons:
            print(f"  {reason:<34} {count:>10,}")

    print()
    print("Sample Queries")
    print("-" * 72)
    for label in ordered_labels:
        print(f"[{label}]")
        if not samples.get(label):
            print("  (none)")
            continue
        for query in samples[label]:
            print(f"  - {query}")

    print("=" * 72)


def main() -> None:
    args = parse_args()
    input_path: Path = args.input
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {input_path}. "
            "Please download AOL query log to this path or pass --input."
        )

    label_counts: Counter[str] = Counter()
    reason_counts: dict[str, Counter[str]] = defaultdict(Counter)
    samples: dict[str, list[str]] = defaultdict(list)
    seen_samples: dict[str, set[str]] = defaultdict(set)
    poi_keep_reason_counts: Counter[str] = Counter()
    filtered_out_non_poi = 0
    skipped_duplicates = 0
    seen_queries: set[str] = set()

    total_rows = 0
    total_queries = 0
    poi_queries = 0

    for query in iter_queries(input_path, encoding=args.encoding):
        total_rows += 1

        if args.progress_every > 0 and total_rows % args.progress_every == 0:
            print(f"[progress] scanned {total_rows:,} queries...")

        normalized = normalize_query(query)
        if args.deduplicate:
            if normalized in seen_queries:
                skipped_duplicates += 1
                continue
            seen_queries.add(normalized)
        total_queries += 1

        is_poi, poi_reasons = detect_poi_intent(query, gate_mode=args.poi_gate)
        if not is_poi:
            filtered_out_non_poi += 1
            continue
        for reason in poi_reasons:
            poi_keep_reason_counts[reason] += 1
        poi_queries += 1

        label, reasons = classify_query(query, split_entity_lookup=args.split_entity_lookup)
        label_counts[label] += 1
        for reason in reasons:
            reason_counts[label][reason] += 1

        if len(samples[label]) < args.sample_size:
            if normalized not in seen_samples[label]:
                seen_samples[label].add(normalized)
                samples[label].append(query.strip())

    ordered_labels = [LABEL_CONCISE, LABEL_SOFT]
    if args.split_entity_lookup:
        ordered_labels.append(LABEL_ENTITY)
    ordered_labels.append(LABEL_GENERAL)

    print_summary(
        path=input_path,
        total_rows=total_rows,
        total_queries=total_queries,
        poi_queries=poi_queries,
        label_counts=label_counts,
        reason_counts=reason_counts,
        samples=samples,
        poi_keep_reason_counts=poi_keep_reason_counts,
        filtered_out_non_poi=filtered_out_non_poi,
        ordered_labels=ordered_labels,
        deduplicate=args.deduplicate,
        skipped_duplicates=skipped_duplicates,
        poi_gate=args.poi_gate,
    )


if __name__ == "__main__":
    main()
