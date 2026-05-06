"""Check a preregistered prediction YAML against measured experimental
results. Each hypothesis specifies a measurement path
(file::key | nested.path) and a fail_state_threshold; this script reports
PASS / FAIL per hypothesis without exiting non-zero on hypothesis failure
(failure ≠ bug; failure = empirical reality contradicts prior belief).

Usage:
    python scripts/check_preregistered.py \
        experiments/preregistered/predictions_p1_1_naive_baselines.yaml

Adopts the discipline rule from artifacts/audit/AUDIT_PLAYBOOK.md: this
runs OUTSIDE the pytest CI gate so a falsified hypothesis triggers
human review (paper restructure or revision) rather than a CI break.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print("PyYAML not installed; run: pip install pyyaml", file=sys.stderr)
    sys.exit(2)

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class HypothesisResult:
    id: str
    claim: str
    measurement: str
    measured_value: Any
    reference_value: Any
    threshold: Any
    fail_state_name: str
    verdict: str  # PASS / FAIL / SKIP
    notes: list[str] = field(default_factory=list)


def _load_measurement(spec: str, base: Path) -> Any:
    """Resolve a 'path/to/file.json :: key' or '... :: nested.path' spec."""
    if "::" not in spec:
        return None
    file_part, key_part = [s.strip() for s in spec.split("::", 1)]
    file_path = base / file_part
    if not file_path.exists():
        return None
    if file_path.suffix == ".json":
        data = json.loads(file_path.read_text())
    else:
        return None
    # Support nested key paths: foo.bar.baz
    cur = data
    for token in key_part.split("."):
        if isinstance(cur, dict) and token in cur:
            cur = cur[token]
        else:
            return None
    return cur


def _evaluate(h: dict, base: Path) -> HypothesisResult:
    h_id = h.get("id", "?")
    claim = h.get("claim", "")
    measurement = h.get("measurement", "")
    threshold = h.get("fail_state_threshold")
    fail_name = h.get("fail_state_name", "FAIL_STATE")
    reference = h.get("reference")

    measured = _load_measurement(measurement, base)
    if measured is None:
        return HypothesisResult(
            id=h_id, claim=claim, measurement=measurement,
            measured_value=None, reference_value=reference,
            threshold=threshold, fail_state_name=fail_name,
            verdict="SKIP",
            notes=["measurement not yet computed or file missing"],
        )

    # Skip the special "ALREADY_CONFIRMED" status — used when an audit
    # invariant already proves the hypothesis (no need to re-evaluate).
    if h.get("status") == "ALREADY_CONFIRMED":
        return HypothesisResult(
            id=h_id, claim=claim, measurement=measurement,
            measured_value=measured, reference_value=reference,
            threshold=threshold, fail_state_name=fail_name,
            verdict="PASS",
            notes=["status=ALREADY_CONFIRMED in YAML"],
        )

    # Heuristic verdict: claim-string parsing for common patterns.
    notes = []
    verdict = "PASS"
    if reference is not None and threshold is not None and isinstance(measured, (int, float)):
        gap = float(reference) - float(measured)
        notes.append(f"reference={reference}, measured={measured}, gap={gap:+.4f}, threshold={threshold}")
        if "by at least" in claim or "≥" in claim or "by " in claim.lower():
            if gap < float(threshold):
                verdict = "FAIL"
                notes.append(f"FAIL: gap {gap:.4f} < required {threshold}")
        elif "above" in claim.lower() or ">" in claim:
            if measured < float(threshold):
                verdict = "FAIL"
                notes.append(f"FAIL: measured {measured} < threshold {threshold}")
    elif threshold is not None and isinstance(measured, (int, float)):
        # No reference, just check measured against threshold
        notes.append(f"measured={measured}, threshold={threshold}")
        if "above" in claim.lower() or "> " in claim:
            if measured < float(threshold):
                verdict = "FAIL"
                notes.append(f"FAIL: measured {measured} < threshold {threshold}")
        elif "less than" in claim.lower() or "<" in claim:
            if measured >= float(threshold):
                verdict = "FAIL"
                notes.append(f"FAIL: measured {measured} >= threshold {threshold}")
    else:
        verdict = "SKIP"
        notes.append("auto-evaluation not implemented for this claim shape")

    return HypothesisResult(
        id=h_id, claim=claim, measurement=measurement,
        measured_value=measured, reference_value=reference,
        threshold=threshold, fail_state_name=fail_name,
        verdict=verdict, notes=notes,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml_path", type=Path,
                    help="Preregistered predictions YAML")
    ap.add_argument("--base", type=Path, default=REPO_ROOT,
                    help="Base path for resolving measurement file refs")
    args = ap.parse_args()

    if not args.yaml_path.exists():
        print(f"Not found: {args.yaml_path}", file=sys.stderr)
        return 2

    spec = yaml.safe_load(args.yaml_path.read_text())
    experiment = spec.get("experiment", "?")
    hypotheses = spec.get("hypothesis_set", [])

    print(f"=== Pre-registered check: {experiment} ===")
    print(f"YAML:     {args.yaml_path}")
    print(f"Hypotheses: {len(hypotheses)}")
    print()

    n_pass, n_fail, n_skip = 0, 0, 0
    failed_states: list[str] = []
    for h in hypotheses:
        r = _evaluate(h, args.base)
        marker = {"PASS": "OK  ", "FAIL": "FAIL", "SKIP": "skip"}[r.verdict]
        print(f"[{marker}] {r.id}: {r.claim[:80]}{'...' if len(r.claim)>80 else ''}")
        for note in r.notes:
            print(f"         {note}")
        if r.verdict == "PASS":
            n_pass += 1
        elif r.verdict == "FAIL":
            n_fail += 1
            failed_states.append(r.fail_state_name)
        else:
            n_skip += 1
        print()

    print(f"Summary: {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP")
    if failed_states:
        print(f"Failed fail-states triggered: {failed_states}")
        print()
        print("NOTE: A failed pre-registered hypothesis is NOT a code bug — "
              "it means empirical reality contradicts the prior belief. "
              "Per AUDIT_PLAYBOOK, this triggers paper-revision review, "
              "not test failure. Exit code is 0 regardless.")

    return 0  # always exit 0; failures are findings, not errors


if __name__ == "__main__":
    raise SystemExit(main())
