from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, TextIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract city-level consistent subset from Yelp Open Dataset JSONL files."
    )
    parser.add_argument(
        "--city",
        default="Indianapolis",
        help="City name to extract (case-insensitive exact match). Default: Indianapolis",
    )
    parser.add_argument(
        "--input-dir",
        default="data",
        help="Directory containing Yelp source JSON files. Default: data",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory to write extracted subset files. Default: data",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help=(
            "Output file prefix without trailing dash. "
            "Default: yelp-<city-slug> (for example yelp-indianapolis)"
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500000,
        help="Print progress every N lines. Default: 500000",
    )
    return parser.parse_args()


def slugify_city(city: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", city.strip().lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "city"


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def write_jsonl_row(writer: TextIO, row: dict) -> None:
    writer.write(json.dumps(row, ensure_ascii=False) + "\n")


def log(message: str) -> None:
    print(message, file=sys.stderr)


def build_input_paths(input_dir: Path) -> dict[str, Path]:
    return {
        "business": input_dir / "yelp_academic_dataset_business.json",
        "review": input_dir / "yelp_academic_dataset_review.json",
        "user": input_dir / "yelp_academic_dataset_user.json",
        "checkin": input_dir / "yelp_academic_dataset_checkin.json",
        "tip": input_dir / "yelp_academic_dataset_tip.json",
    }


def build_output_paths(output_dir: Path, prefix: str) -> dict[str, Path]:
    return {
        "business": output_dir / f"{prefix}-business.jsonl",
        "review": output_dir / f"{prefix}-review.jsonl",
        "user": output_dir / f"{prefix}-user.jsonl",
        "checkin": output_dir / f"{prefix}-checkin.jsonl",
        "tip": output_dir / f"{prefix}-tip.jsonl",
        "meta": output_dir / f"{prefix}-meta.json",
    }


def ensure_inputs_exist(input_paths: dict[str, Path]) -> None:
    missing = [name for name, path in input_paths.items() if not path.exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Missing input files for: {joined}")


def extract_city_subset(
    city: str,
    input_paths: dict[str, Path],
    output_paths: dict[str, Path],
    progress_every: int = 500000,
) -> dict:
    city_norm = city.strip().lower()
    business_ids: set[str] = set()
    user_ids: set[str] = set()

    counters = {
        "business_in": 0,
        "business_out": 0,
        "review_in": 0,
        "review_out": 0,
        "tip_in": 0,
        "tip_out": 0,
        "user_in": 0,
        "user_out": 0,
        "checkin_in": 0,
        "checkin_out": 0,
    }

    # 1) Business by city
    log(f"[1/5] Filtering business rows for city='{city}'...")
    with output_paths["business"].open("w", encoding="utf-8") as fw:
        for i, row in enumerate(iter_jsonl(input_paths["business"]), 1):
            counters["business_in"] += 1
            row_city = str(row.get("city", "")).strip().lower()
            if row_city == city_norm:
                bid = str(row.get("business_id", "")).strip()
                if bid:
                    business_ids.add(bid)
                    write_jsonl_row(fw, row)
                    counters["business_out"] += 1
            if i % progress_every == 0:
                log(f"  business processed={i:,}, matched={counters['business_out']:,}")
    log(f"  business done: matched={counters['business_out']:,}")

    # 2) Review by selected business_id; collect review users
    log("[2/5] Filtering review rows by selected business_id...")
    with output_paths["review"].open("w", encoding="utf-8") as fw:
        for i, row in enumerate(iter_jsonl(input_paths["review"]), 1):
            counters["review_in"] += 1
            bid = str(row.get("business_id", "")).strip()
            if bid in business_ids:
                uid = str(row.get("user_id", "")).strip()
                if uid:
                    user_ids.add(uid)
                write_jsonl_row(fw, row)
                counters["review_out"] += 1
            if i % progress_every == 0:
                log(
                    f"  review processed={i:,}, matched={counters['review_out']:,}, "
                    f"users={len(user_ids):,}"
                )
    log(
        f"  review done: matched={counters['review_out']:,}, "
        f"users_from_review={len(user_ids):,}"
    )

    # 3) Tip by selected business_id; collect tip users for consistency
    log("[3/5] Filtering tip rows by selected business_id...")
    with output_paths["tip"].open("w", encoding="utf-8") as fw:
        for i, row in enumerate(iter_jsonl(input_paths["tip"]), 1):
            counters["tip_in"] += 1
            bid = str(row.get("business_id", "")).strip()
            if bid in business_ids:
                uid = str(row.get("user_id", "")).strip()
                if uid:
                    user_ids.add(uid)
                write_jsonl_row(fw, row)
                counters["tip_out"] += 1
            if i % progress_every == 0:
                log(
                    f"  tip processed={i:,}, matched={counters['tip_out']:,}, "
                    f"users_union={len(user_ids):,}"
                )
    log(
        f"  tip done: matched={counters['tip_out']:,}, "
        f"users_union={len(user_ids):,}"
    )

    # 4) User by union(review users + tip users)
    log("[4/5] Filtering user rows by user_ids observed in review/tip...")
    with output_paths["user"].open("w", encoding="utf-8") as fw:
        for i, row in enumerate(iter_jsonl(input_paths["user"]), 1):
            counters["user_in"] += 1
            uid = str(row.get("user_id", "")).strip()
            if uid in user_ids:
                write_jsonl_row(fw, row)
                counters["user_out"] += 1
            if i % progress_every == 0:
                log(f"  user processed={i:,}, matched={counters['user_out']:,}")
    log(f"  user done: matched={counters['user_out']:,}")

    # 5) Checkin by selected business_id
    log("[5/5] Filtering checkin rows by selected business_id...")
    with output_paths["checkin"].open("w", encoding="utf-8") as fw:
        for i, row in enumerate(iter_jsonl(input_paths["checkin"]), 1):
            counters["checkin_in"] += 1
            bid = str(row.get("business_id", "")).strip()
            if bid in business_ids:
                write_jsonl_row(fw, row)
                counters["checkin_out"] += 1
            if i % progress_every == 0:
                log(f"  checkin processed={i:,}, matched={counters['checkin_out']:,}")
    log(f"  checkin done: matched={counters['checkin_out']:,}")

    summary = {
        "city": city,
        "business_ids": len(business_ids),
        "user_ids": len(user_ids),
        "counts": counters,
        "outputs": {k: str(v) for k, v in output_paths.items()},
    }

    with output_paths["meta"].open("w", encoding="utf-8") as fw:
        json.dump(summary, fw, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    args = parse_args()
    city = args.city.strip()
    if not city:
        raise ValueError("--city cannot be empty")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix.strip() or f"yelp-{slugify_city(city)}"
    input_paths = build_input_paths(input_dir)
    output_paths = build_output_paths(output_dir, prefix)
    ensure_inputs_exist(input_paths)

    summary = extract_city_subset(
        city=city,
        input_paths=input_paths,
        output_paths=output_paths,
        progress_every=args.progress_every,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

