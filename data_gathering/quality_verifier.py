"""
Quality Verifier — Post-curation dataset QA
=============================================
Loads the curated Disney dataset (.pt files) and the QA report, then runs
deeper verification checks and produces a human-readable review summary.

Checks performed:
  1. Format consistency — every example has all required keys
  2. Syllable ratio distribution — flag outliers
  3. Melody coverage — notes per target syllable
  4. Token length sanity — src_ids and tgt_ids within bounds
  5. Duplicate detection — same text appearing multiple times
  6. Cross-check with QA flags from the curator

Usage:
  cd musically-aligned-translation/
  python -m data_gathering.quality_verifier
  python -m data_gathering.quality_verifier --data data_gathering/output/disney_train_data.pt
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict


REQUIRED_KEYS = {
    "src_ids", "tgt_ids", "melody_features", "src_syllables",
    "tgt_syllables", "num_notes", "song_name", "english_text",
    "hindi_text", "stress_pattern"
}


def verify_dataset(data_path: str, qa_report_path: str = None,
                   verbose: bool = True) -> Dict:
    """
    Run all quality checks on a .pt dataset file.

    Returns a summary dict with pass/fail counts and flagged examples.
    """
    data = torch.load(data_path, weights_only=False)
    print(f"Loaded {len(data)} examples from {data_path}\n")

    summary = {
        "total_examples": len(data),
        "format_errors": [],
        "syllable_outliers": [],
        "melody_outliers": [],
        "token_outliers": [],
        "duplicates": [],
        "per_song_stats": {},
        "overall_pass": True,
    }

    # ------------------------------------------------------------------
    # Check 1: Format consistency
    # ------------------------------------------------------------------
    print("Check 1: Format consistency")
    for i, ex in enumerate(data):
        missing = REQUIRED_KEYS - set(ex.keys())
        if missing:
            summary["format_errors"].append({
                "index": i,
                "song_name": ex.get("song_name", "?"),
                "missing_keys": list(missing),
            })
    n_fmt = len(summary["format_errors"])
    print(f"  {'✓' if n_fmt == 0 else '✗'} {n_fmt} format errors\n")

    # ------------------------------------------------------------------
    # Check 2: Syllable ratio distribution
    # ------------------------------------------------------------------
    print("Check 2: Syllable ratios (Hindi / English)")
    syl_ratios = []
    for i, ex in enumerate(data):
        en_syl = ex.get("src_syllables", 0)
        hi_syl = ex.get("tgt_syllables", 0)
        if en_syl > 0:
            ratio = hi_syl / en_syl
            syl_ratios.append(ratio)
            if ratio > 2.5 or ratio < 0.3:
                summary["syllable_outliers"].append({
                    "index": i,
                    "song_name": ex.get("song_name", "?"),
                    "en_syl": en_syl, "hi_syl": hi_syl,
                    "ratio": round(ratio, 2),
                    "english": ex.get("english_text", "")[:80],
                    "hindi": ex.get("hindi_text", "")[:80],
                })

    if syl_ratios:
        arr = np.array(syl_ratios)
        print(f"  Mean ratio : {arr.mean():.2f}")
        print(f"  Std        : {arr.std():.2f}")
        print(f"  Range      : [{arr.min():.2f}, {arr.max():.2f}]")
    print(f"  {'✓' if len(summary['syllable_outliers']) == 0 else '⚠'} "
          f"{len(summary['syllable_outliers'])} outliers (ratio <0.3 or >2.5)\n")

    # ------------------------------------------------------------------
    # Check 3: Melody coverage (notes per Hindi syllable)
    # ------------------------------------------------------------------
    print("Check 3: Melody coverage")
    note_ratios = []
    for i, ex in enumerate(data):
        hi_syl = ex.get("tgt_syllables", 0)
        n_notes = ex.get("num_notes", 0)
        if hi_syl > 0 and n_notes > 0:
            ratio = n_notes / hi_syl
            note_ratios.append(ratio)
            if ratio < 0.2 or ratio > 8.0:
                summary["melody_outliers"].append({
                    "index": i,
                    "song_name": ex.get("song_name", "?"),
                    "notes": n_notes, "hi_syl": hi_syl,
                    "ratio": round(ratio, 2),
                })

    if note_ratios:
        arr = np.array(note_ratios)
        print(f"  Mean notes/syl : {arr.mean():.2f}")
        print(f"  Std            : {arr.std():.2f}")
    print(f"  {'✓' if len(summary['melody_outliers']) == 0 else '⚠'} "
          f"{len(summary['melody_outliers'])} outliers\n")

    # ------------------------------------------------------------------
    # Check 4: Token length sanity
    # ------------------------------------------------------------------
    print("Check 4: Token lengths")
    src_lens = []
    tgt_lens = []
    for i, ex in enumerate(data):
        sl = ex["src_ids"].size(0) if torch.is_tensor(ex.get("src_ids")) else 0
        tl = ex["tgt_ids"].size(0) if torch.is_tensor(ex.get("tgt_ids")) else 0
        src_lens.append(sl)
        tgt_lens.append(tl)
        if sl > 120 or tl > 120 or sl < 3 or tl < 3:
            summary["token_outliers"].append({
                "index": i,
                "song_name": ex.get("song_name", "?"),
                "src_len": sl, "tgt_len": tl,
            })

    print(f"  Src tokens : mean={np.mean(src_lens):.1f}, "
          f"max={np.max(src_lens)}, min={np.min(src_lens)}")
    print(f"  Tgt tokens : mean={np.mean(tgt_lens):.1f}, "
          f"max={np.max(tgt_lens)}, min={np.min(tgt_lens)}")
    print(f"  {'✓' if len(summary['token_outliers']) == 0 else '⚠'} "
          f"{len(summary['token_outliers'])} outliers\n")

    # ------------------------------------------------------------------
    # Check 5: Duplicate detection
    # ------------------------------------------------------------------
    print("Check 5: Duplicates")
    en_counter = Counter(ex.get("english_text", "") for ex in data)
    hi_counter = Counter(ex.get("hindi_text", "") for ex in data)
    en_dups = {t: c for t, c in en_counter.items() if c > 1 and t}
    hi_dups = {t: c for t, c in hi_counter.items() if c > 1 and t}
    summary["duplicates"] = {
        "english_duplicates": len(en_dups),
        "hindi_duplicates": len(hi_dups),
        "examples": list(en_dups.items())[:10],
    }
    total_dups = len(en_dups) + len(hi_dups)
    print(f"  English text duplicates: {len(en_dups)}")
    print(f"  Hindi text duplicates  : {len(hi_dups)}")
    print(f"  {'✓' if total_dups == 0 else '⚠'} {total_dups} duplicate groups\n")

    # ------------------------------------------------------------------
    # Per-song breakdown
    # ------------------------------------------------------------------
    print("Per-song breakdown:")
    song_groups = defaultdict(list)
    for i, ex in enumerate(data):
        base_name = ex.get("song_name", "?").split("_line")[0]
        song_groups[base_name].append(ex)

    for song, examples in sorted(song_groups.items()):
        avg_syl_ratio = np.mean([
            ex["tgt_syllables"] / max(ex["src_syllables"], 1) for ex in examples
        ])
        avg_notes = np.mean([ex["num_notes"] for ex in examples])
        summary["per_song_stats"][song] = {
            "num_lines": len(examples),
            "avg_syllable_ratio": round(avg_syl_ratio, 2),
            "avg_notes_per_line": round(avg_notes, 1),
            "source": examples[0].get("source", "unknown"),
        }
        print(f"  {song:35s} {len(examples):3d} lines, "
              f"syl_ratio={avg_syl_ratio:.2f}, "
              f"avg_notes={avg_notes:.1f}")

    # ------------------------------------------------------------------
    # Cross-check with QA report
    # ------------------------------------------------------------------
    if qa_report_path and Path(qa_report_path).exists():
        print(f"\nCross-checking with QA report: {qa_report_path}")
        with open(qa_report_path) as f:
            qa_report = json.load(f)
        total_flags = sum(len(entry.get("flags", [])) for entry in qa_report)
        error_flags = sum(
            1 for entry in qa_report
            for flag in entry.get("flags", [])
            if flag.get("severity") == "error"
        )
        print(f"  Total QA flags: {total_flags}")
        print(f"  Error flags   : {error_flags}")
        print(f"  Warning flags : {total_flags - error_flags}")

    # ------------------------------------------------------------------
    # Overall verdict
    # ------------------------------------------------------------------
    has_errors = (
        len(summary["format_errors"]) > 0
        or len(summary["token_outliers"]) > len(data) * 0.2
    )
    summary["overall_pass"] = not has_errors

    print(f"\n{'='*60}")
    if summary["overall_pass"]:
        print("  ✓ DATASET PASSES basic quality checks")
    else:
        print("  ✗ DATASET HAS ISSUES — review flagged examples above")
    print(f"{'='*60}\n")

    # Save verification report
    report_path = Path(data_path).parent / "verification_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"Report saved → {report_path}")

    return summary


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    import os

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    os.chdir(PROJECT_ROOT)

    parser = argparse.ArgumentParser(
        description="Verify quality of curated Disney song dataset"
    )
    parser.add_argument(
        "--data", type=str,
        default="data_gathering/output/disney_train_data.pt",
        help="Path to .pt dataset file"
    )
    parser.add_argument(
        "--qa-report", type=str,
        default="data_gathering/output/qa_report.json",
        help="Path to QA report from curator"
    )
    args = parser.parse_args()

    verify_dataset(args.data, args.qa_report)
