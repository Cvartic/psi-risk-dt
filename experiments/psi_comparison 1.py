"""
psi_comparison.py
=================
Experimental comparison between the Neural-Only and Neuro-Symbolic (NeSy) pipelines
as defined in Vartic's thesis, Sections 4.2, 5.2, and 5.5.2.

Overview
--------
The Ψ operator (Psi) maps an entropic deviation signal ΔHt and a symbolic gate Gt
to a risk activation at and an updated gate state Gt+1:

    Ψ : (ΔHt, Gt) → (at, Gt+1)

Gating condition (NeSy only):
    ΔHt > τs  AND  neural_score > θ_pre_gate

Two configurations are compared per dataset tier:
    - neural-only    : risk_score = weighted_sum(features), no symbolic gating
    - neurosymbolic  : same neural core + modulating symbolic gate

DISCLAIMER / PLACEHOLDER NOTE
------------------------------
The "neural" component used here is a *calibrated weighted-sum placeholder* that
approximates the ARNN (Attention-based Recurrent Neural Network) described in the
thesis.  It is NOT a trained model.  Its weights and normalisation constants are
chosen so that:
    - normal_traffic and udp_flood_low → score < 0.5  (benign baseline)
    - escalation during stealth/rampup  → score 0.4–0.7 (ambiguous zone)
    - saturation/flood peak             → score > 0.7  (clear alert)
This makes the comparison meaningful and consistent with the behavioural claims of
the thesis without requiring a trained ARNN checkpoint.

Usage
-----
    python psi_comparison.py --data_dir data/sliding_windows --out_dir results

Or, using the programmatic API:
    from psi_comparison import get_risk_timeline, run_comparison
    timeline = get_risk_timeline("escalation", "mid", "data/sliding_windows")
    run_comparison("data/sliding_windows", "results")
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants — Θ and weights (thesis §4.2 / §5.2)
# ---------------------------------------------------------------------------

# Neural activation threshold (thesis §4.2)
THETA: float = 0.7

# Pre-gate threshold used by the symbolic modulator.
# Deliberately set lower than Θ so the gate can boost borderline detections.
THETA_PRE_GATE: float = 0.5

# Neural feature weights: [f1_norm, f2, f3, f4, f5_norm]
# f1 = traffic_rate_pps  (dominant predictor of flood intensity)
# f2 = entropy_dst_port  (high during scans / low during floods)
# f3 = entropy_src_ip    (spoofing signal in UDP floods)
# f4 = entropy_payload_sz
# f5 = std_jitter_ms     (normalised by 1000)
#
# Weight rationale:
#   f1 (traffic rate) carries the strongest flood signal and is given the
#   largest weight (0.45).  The entropy features contribute the remaining 0.45
#   equally, and jitter adds a small tail (0.10).
#   With f1_max calibrated to the observed dataset peak, a saturation window
#   with f1 ≈ 90% of max and moderate entropies yields score > 0.70, while a
#   benign window with f1 < 5% of max stays below 0.50.
NEURAL_WEIGHTS: list[float] = [0.45, 0.20, 0.15, 0.10, 0.10]
assert abs(sum(NEURAL_WEIGHTS) - 1.0) < 1e-9, "Weights must sum to 1"

# Normalisation cap for f1 (traffic_rate_pps); set to peak observed across the
# full dataset.  Updated dynamically during dataset loading — see _build_scaler.
_F1_MAX_FALLBACK: float = 10_000.0   # pps — conservative default
_F5_DIVISOR: float = 1_000.0         # ms → normalised jitter

# Symbolic gate modulation factors (thesis MSU logic, §5.2)
GATE_BOOST: float = 1.30   # when exceedsTauS AND neural > θ_pre_gate
GATE_SUPPRESS: float = 0.85  # when NOT exceedsTauS (noise suppression)

# Known dataset identifiers
SCENARIOS: list[str] = [
    "udp_flood",
    "escalation",
    "zero_day_like",
    "normal_traffic",
]
TIERS: list[str] = ["low", "mid", "high"]

# Phases that represent stealth / silence windows (thesis §5.5.2)
STEALTH_PHASES: set[str] = {"stealth", "silence", "rampup_low"}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Window:
    """Single sliding window parsed from a *_windows.jsonl file."""

    window_id: str
    t_start: float
    t_end: float
    n_packets: int
    f1: float          # traffic_rate_pps
    f2: float          # entropy_dst_port
    f3: float          # entropy_src_ip
    f4: float          # entropy_payload_sz
    f5: float          # std_jitter_ms
    delta_h: float     # ΔHt — entropic deviation
    exceeds_tau_s: bool  # pre-computed ΔHt > τs boolean
    phase: str           # ground-truth phase label
    config: str          # "neural-only" | "neurosymbolic"

    # Derived (filled during scoring)
    neural_score: float = 0.0
    neural_only_risk: float = 0.0
    neurosymbolic_risk: float = 0.0
    gate_fired: bool = False


@dataclass
class DatasetResult:
    """Aggregated results for one scenario+tier pair."""

    scenario: str
    tier: str
    windows: list[Window] = field(default_factory=list)

    # Metric values
    detection_latency_neural: int | None = None
    detection_latency_nesy: int | None = None
    fnr_neural: float | None = None
    fnr_nesy: float | None = None
    consistency_neural: float | None = None
    consistency_nesy: float | None = None
    symbolic_activation_rate: float | None = None


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _file_name(scenario: str, tier: str) -> str:
    """Return the expected JSONL filename for a given scenario/tier pair."""
    if scenario == "normal_traffic":
        return "normal_traffic_windows.jsonl"
    return f"{scenario}_{tier}_windows.jsonl"


def load_windows(data_dir: str | Path, scenario: str, tier: str) -> list[Window]:
    """Parse all windows from a *_windows.jsonl file.

    Parameters
    ----------
    data_dir:
        Directory containing the JSONL files.
    scenario:
        One of 'udp_flood', 'escalation', 'zero_day_like', 'normal_traffic'.
    tier:
        One of 'low', 'mid', 'high' (ignored for normal_traffic).

    Returns
    -------
    list[Window]
        Parsed window objects, in file order.
    """
    path = Path(data_dir) / _file_name(scenario, tier)
    if not path.exists():
        raise FileNotFoundError(f"JSONL not found: {path}")

    windows: list[Window] = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON parse error at {path}:{line_no}: {exc}") from exc

            # Robust boolean parsing for exceedsTauS
            ets_raw = raw.get("exceeds_tau_s", raw.get("exceedsTauS", False))
            if isinstance(ets_raw, str):
                exceeds_tau_s = ets_raw.lower() in {"true", "1", "yes"}
            else:
                exceeds_tau_s = bool(ets_raw)

            w = Window(
                window_id=str(raw.get("window_id", f"w{line_no}")),
                t_start=float(raw.get("t_start", 0.0)),
                t_end=float(raw.get("t_end", 0.0)),
                n_packets=int(raw.get("n_packets", 0)),
                f1=float(raw.get("f1", 0.0)),
                f2=float(raw.get("f2", 0.0)),
                f3=float(raw.get("f3", 0.0)),
                f4=float(raw.get("f4", 0.0)),
                f5=float(raw.get("f5", 0.0)),
                delta_h=float(raw.get("delta_h", 0.0)),
                exceeds_tau_s=exceeds_tau_s,
                phase=str(raw.get("phase", "unknown")),
                config=str(raw.get("config", "neural-only")),
            )
            windows.append(w)

    return windows


def _collect_f1_max(all_windows: list[Window]) -> float:
    """Return the global maximum f1 across all loaded windows for normalisation."""
    if not all_windows:
        return _F1_MAX_FALLBACK
    return max(w.f1 for w in all_windows) or _F1_MAX_FALLBACK


# ---------------------------------------------------------------------------
# Neural scoring (ARNN placeholder)
# ---------------------------------------------------------------------------

def _neural_score(w: Window, f1_max: float) -> float:
    """Compute the placeholder neural risk score for a window.

    This is a calibrated weighted sum of five normalised features.
    It stands in for the ARNN output (thesis §4.2) and is NOT a trained model.

    Feature normalisation:
        f1_norm = f1 / f1_max          (traffic rate, capped at observed max)
        f2      = f2  [0, 1] assumed]   (entropy already normalised in JSONL)
        f3      = f3  [0, 1] assumed]
        f4      = f4  [0, 1] assumed]
        f5_norm = f5 / 1000            (jitter in ms → approx [0, 1])

    Expected behaviour (calibration targets):
        normal_traffic      → score < 0.50
          typical: f1 ≈ 2% of max, entropy features ≈ 0.30
        udp_flood_low       → score < 0.50
          typical: f1 ≈ 5% of max, entropy features ≈ 0.25
        escalation stealth  → score 0.40–0.70
          typical: f1 ≈ 25–30% of max, entropy features ≈ 0.55–0.65
        flood/saturation    → score > 0.70
          typical: f1 ≈ 90–100% of max, any entropy features

    Note: f1_max is calibrated across the full loaded dataset so that
    saturation windows (peak traffic rate) normalise to ≈ 1.0.  Scores
    outside the target ranges are possible for atypical feature combinations.
    """
    f1_norm = min(w.f1 / f1_max, 1.0)
    f2_norm = min(max(w.f2, 0.0), 1.0)
    f3_norm = min(max(w.f3, 0.0), 1.0)
    f4_norm = min(max(w.f4, 0.0), 1.0)
    f5_norm = min(w.f5 / _F5_DIVISOR, 1.0)

    features = [f1_norm, f2_norm, f3_norm, f4_norm, f5_norm]
    score = sum(wt * ft for wt, ft in zip(NEURAL_WEIGHTS, features))
    return round(min(max(score, 0.0), 1.0), 6)


# ---------------------------------------------------------------------------
# Scoring: neural-only vs. neurosymbolic
# ---------------------------------------------------------------------------

def score_window(w: Window, f1_max: float) -> None:
    """Populate neural_only_risk, neurosymbolic_risk, and gate_fired in-place.

    Neural-only (thesis §5.2 baseline):
        risk_score = weighted_sum(features)
        No symbolic gating; the score passes through unchanged.

    Neurosymbolic (thesis §4.2 / §5.2, MSU logic):
        Same neural core, then modulated by the symbolic gate:
          - IF exceedsTauS AND neural_score > θ_pre_gate:
                risk_ns = min(neural_score * GATE_BOOST, 1.0)   [boost]
                gate_fired = True
          - ELSE:
                risk_ns = neural_score * GATE_SUPPRESS          [suppress]
                gate_fired = False

        Rationale: the gate is not a hard binary switch but a modulating
        operator (MSU, §5.2). Borderline detections are reinforced when
        symbolic context confirms anomaly; noise is damped otherwise.
    """
    ns = _neural_score(w, f1_max)
    w.neural_score = ns
    w.neural_only_risk = ns

    if w.exceeds_tau_s and ns > THETA_PRE_GATE:
        w.neurosymbolic_risk = round(min(ns * GATE_BOOST, 1.0), 6)
        w.gate_fired = True
    else:
        w.neurosymbolic_risk = round(ns * GATE_SUPPRESS, 6)
        w.gate_fired = False


def score_all(windows: list[Window], f1_max: float) -> None:
    """Score all windows in-place."""
    for w in windows:
        score_window(w, f1_max)


# ---------------------------------------------------------------------------
# Metrics (thesis §5.5.2)
# ---------------------------------------------------------------------------

def metric_detection_latency(windows: list[Window], use_nesy: bool) -> int | None:
    """Return the index (0-based) of the first window where risk score > θ.

    A lower value means earlier detection.  None indicates no detection.
    (thesis §5.5.2, metric 1)
    """
    for idx, w in enumerate(windows):
        score = w.neurosymbolic_risk if use_nesy else w.neural_only_risk
        if score > THETA:
            return idx
    return None


def metric_false_negative_rate(windows: list[Window], use_nesy: bool) -> float:
    """Proportion of stealth/silence windows where risk score < θ (missed detections).

    Stealth windows: phases labelled 'stealth', 'silence', 'rampup_low' —
    these are the low-visibility anomaly phases defined in §5.5.2.
    FNR = |{stealth windows with score < θ}| / |{stealth windows}|
    Returns NaN if no stealth windows exist in this dataset.
    """
    stealth = [w for w in windows if w.phase.lower() in STEALTH_PHASES]
    if not stealth:
        return float("nan")
    missed = sum(
        1
        for w in stealth
        if (w.neurosymbolic_risk if use_nesy else w.neural_only_risk) < THETA
    )
    return round(missed / len(stealth), 6)


def metric_risk_score_consistency(windows: list[Window], use_nesy: bool) -> float:
    """Standard deviation of risk scores within each distinct phase, averaged.

    Lower std dev → more consistent / stable predictions within a phase.
    (thesis §5.5.2, metric 3)

    Returns the mean per-phase std dev across all phases present in the dataset.
    """
    phase_scores: dict[str, list[float]] = {}
    for w in windows:
        score = w.neurosymbolic_risk if use_nesy else w.neural_only_risk
        phase_scores.setdefault(w.phase, []).append(score)

    per_phase_stds: list[float] = []
    for scores in phase_scores.values():
        if len(scores) >= 2:
            per_phase_stds.append(statistics.stdev(scores))
        else:
            per_phase_stds.append(0.0)

    if not per_phase_stds:
        return float("nan")
    return round(statistics.mean(per_phase_stds), 6)


def metric_symbolic_activation_rate(windows: list[Window]) -> float:
    """Fraction of windows where the symbolic gate fired (NeSy only).

    gate_fired ↔ exceedsTauS AND neural_score > θ_pre_gate
    (thesis §5.5.2, metric 4)
    """
    if not windows:
        return float("nan")
    activated = sum(1 for w in windows if w.gate_fired)
    return round(activated / len(windows), 6)


def compute_metrics(windows: list[Window]) -> dict[str, float | None]:
    """Compute all four §5.5.2 metrics for both configurations.

    Returns a flat dict keyed by 'neural_<metric>' and 'nesy_<metric>'.
    """
    return {
        "detection_latency_neural": metric_detection_latency(windows, use_nesy=False),
        "detection_latency_nesy": metric_detection_latency(windows, use_nesy=True),
        "fnr_neural": metric_false_negative_rate(windows, use_nesy=False),
        "fnr_nesy": metric_false_negative_rate(windows, use_nesy=True),
        "consistency_neural": metric_risk_score_consistency(windows, use_nesy=False),
        "consistency_nesy": metric_risk_score_consistency(windows, use_nesy=True),
        "symbolic_activation_rate": metric_symbolic_activation_rate(windows),
    }


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

CSV_COLUMNS: list[str] = [
    "dataset",
    "tier",
    "scenario",
    "metric_name",
    "neural_only_value",
    "neurosymbolic_value",
    "delta",
]

METRIC_NAMES: list[str] = [
    "detection_latency",
    "false_negative_rate",
    "risk_score_consistency",
    "symbolic_activation_rate",
]


def _delta(neural: float | None, nesy: float | None) -> str:
    """Compute delta = nesy - neural; returns 'N/A' when either is None/NaN."""
    if neural is None or nesy is None:
        return "N/A"
    try:
        if neural != neural or nesy != nesy:  # NaN check
            return "N/A"
        return str(round(nesy - neural, 6))
    except TypeError:
        return "N/A"


def _fmt(val: float | int | None) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, float) and val != val:  # NaN
        return "N/A"
    return str(val)


def write_csv(results: list[DatasetResult], out_dir: str | Path) -> Path:
    """Write the comparison metrics CSV to out_dir/psi_comparison_metrics.csv.

    Returns the path to the written file.
    """
    out_path = Path(out_dir) / "psi_comparison_metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for r in results:
            # Build display strings
            dataset_label = (
                f"{r.scenario}_{r.tier}" if r.scenario != "normal_traffic" else "normal_traffic"
            )
            common = {
                "dataset": dataset_label,
                "tier": r.tier,
                "scenario": r.scenario,
            }

            metrics_rows = [
                {
                    "metric_name": "detection_latency",
                    "neural_only_value": _fmt(r.detection_latency_neural),
                    "neurosymbolic_value": _fmt(r.detection_latency_nesy),
                    "delta": _delta(r.detection_latency_neural, r.detection_latency_nesy),
                },
                {
                    "metric_name": "false_negative_rate",
                    "neural_only_value": _fmt(r.fnr_neural),
                    "neurosymbolic_value": _fmt(r.fnr_nesy),
                    "delta": _delta(r.fnr_neural, r.fnr_nesy),
                },
                {
                    "metric_name": "risk_score_consistency",
                    "neural_only_value": _fmt(r.consistency_neural),
                    "neurosymbolic_value": _fmt(r.consistency_nesy),
                    "delta": _delta(r.consistency_neural, r.consistency_nesy),
                },
                {
                    "metric_name": "symbolic_activation_rate",
                    "neural_only_value": "N/A",
                    "neurosymbolic_value": _fmt(r.symbolic_activation_rate),
                    "delta": "N/A",
                },
            ]

            for row in metrics_rows:
                writer.writerow({**common, **row})

    return out_path


# ---------------------------------------------------------------------------
# Public API — get_risk_timeline
# ---------------------------------------------------------------------------

def get_risk_timeline(
    scenario: str,
    tier: str,
    data_dir: str | Path,
) -> dict[str, list[Any]]:
    """Return the per-window risk score timeline for both pipeline configurations.

    Parameters
    ----------
    scenario:
        One of 'udp_flood', 'escalation', 'zero_day_like', 'normal_traffic'.
    tier:
        One of 'low', 'mid', 'high'.  Ignored for normal_traffic.
    data_dir:
        Path to the directory containing *_windows.jsonl files.

    Returns
    -------
    dict with keys:
        'windows'       : list of window_id strings
        'neural_only'   : list of float risk scores (neural-only config)
        'neurosymbolic' : list of float risk scores (NeSy config)
        'phases'        : list of ground-truth phase labels
        'gate_fired'    : list of bool — True when symbolic gate activated
        't_start'       : list of float timestamps
        't_end'         : list of float timestamps
    """
    all_windows = _load_and_score(scenario, tier, data_dir)
    return {
        "windows": [w.window_id for w in all_windows],
        "neural_only": [w.neural_only_risk for w in all_windows],
        "neurosymbolic": [w.neurosymbolic_risk for w in all_windows],
        "phases": [w.phase for w in all_windows],
        "gate_fired": [w.gate_fired for w in all_windows],
        "t_start": [w.t_start for w in all_windows],
        "t_end": [w.t_end for w in all_windows],
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_and_score(
    scenario: str,
    tier: str,
    data_dir: str | Path,
    f1_max: float | None = None,
) -> list[Window]:
    """Load windows for one scenario/tier, compute f1_max if needed, score all."""
    windows = load_windows(data_dir, scenario, tier)
    if f1_max is None:
        f1_max = _collect_f1_max(windows)
    score_all(windows, f1_max)
    return windows


def _load_all_windows(data_dir: str | Path) -> dict[tuple[str, str], list[Window]]:
    """Load every available JSONL file, skip missing ones gracefully.

    Returns a dict keyed by (scenario, tier).
    """
    all_data: dict[tuple[str, str], list[Window]] = {}
    combos: list[tuple[str, str]] = []
    for s in SCENARIOS:
        if s == "normal_traffic":
            combos.append((s, ""))
        else:
            for t in TIERS:
                combos.append((s, t))

    for scenario, tier in combos:
        try:
            windows = load_windows(data_dir, scenario, tier)
            all_data[(scenario, tier)] = windows
        except FileNotFoundError:
            pass  # silently skip missing datasets

    return all_data


def _global_f1_max(all_data: dict[tuple[str, str], list[Window]]) -> float:
    """Compute f1 max across the entire dataset collection."""
    all_windows = [w for wins in all_data.values() for w in wins]
    return _collect_f1_max(all_windows)


# ---------------------------------------------------------------------------
# run_comparison — main experiment entry point
# ---------------------------------------------------------------------------

def run_comparison(
    data_dir: str | Path = "data/sliding_windows",
    out_dir: str | Path = "results",
    verbose: bool = True,
) -> list[DatasetResult]:
    """Run the full Ψ comparison experiment.

    Loads all available JSONL datasets, scores every window with both
    neural-only and neurosymbolic configurations, computes the four §5.5.2
    metrics, writes psi_comparison_metrics.csv, and returns the result list.

    Parameters
    ----------
    data_dir:
        Directory containing *_windows.jsonl files.
    out_dir:
        Output directory for psi_comparison_metrics.csv.
    verbose:
        Print progress and a summary table to stdout.
    """
    if verbose:
        print(f"[psi_comparison] Loading datasets from: {data_dir}")

    all_data = _load_all_windows(data_dir)
    if not all_data:
        raise RuntimeError(
            f"No JSONL files found in '{data_dir}'. "
            "Check the path and ensure *_windows.jsonl files are present."
        )

    f1_max = _global_f1_max(all_data)
    if verbose:
        print(f"[psi_comparison] Global f1_max (normalisation cap): {f1_max:.2f} pps")
        print(f"[psi_comparison] Loaded {len(all_data)} dataset(s).")

    results: list[DatasetResult] = []

    for (scenario, tier), windows in sorted(all_data.items()):
        score_all(windows, f1_max)
        m = compute_metrics(windows)

        r = DatasetResult(
            scenario=scenario,
            tier=tier if tier else "—",
            windows=windows,
            detection_latency_neural=m["detection_latency_neural"],
            detection_latency_nesy=m["detection_latency_nesy"],
            fnr_neural=m["fnr_neural"],
            fnr_nesy=m["fnr_nesy"],
            consistency_neural=m["consistency_neural"],
            consistency_nesy=m["consistency_nesy"],
            symbolic_activation_rate=m["symbolic_activation_rate"],
        )
        results.append(r)

        if verbose:
            label = f"{scenario}_{tier}" if tier else scenario
            print(
                f"  [{label:30s}] "
                f"latency neural={_fmt(r.detection_latency_neural):>6} "
                f"nesy={_fmt(r.detection_latency_nesy):>6} | "
                f"FNR neural={_fmt(r.fnr_neural):>8} "
                f"nesy={_fmt(r.fnr_nesy):>8} | "
                f"gate_rate={_fmt(r.symbolic_activation_rate):>8}"
            )

    csv_path = write_csv(results, out_dir)
    if verbose:
        print(f"[psi_comparison] Results written to: {csv_path}")

    return results


# ---------------------------------------------------------------------------
# Self-test (calibration verification)
# ---------------------------------------------------------------------------

def self_test() -> None:
    """Run a quick sanity-check on the scoring calibration.

    Uses synthetic windows with known feature profiles to verify that the
    placeholder neural scores match the behavioural targets described in the
    thesis and in the docstring of _neural_score.

    Raises AssertionError on failure.
    """
    # Build a synthetic dataset where f1_max is the peak (saturation) window.
    # Feature profiles are chosen to match typical real-data values per phase.
    windows = [
        Window("t_normal",   0, 5, 100,  f1=180,  f2=0.30, f3=0.28, f4=0.32, f5=25,
               delta_h=0.05, exceeds_tau_s=False, phase="normal",     config="neural-only"),
        Window("t_ufl",      0, 5, 500,  f1=450,  f2=0.22, f3=0.38, f4=0.24, f5=55,
               delta_h=0.18, exceeds_tau_s=False, phase="flood_low",  config="neural-only"),
        Window("t_stealth",  0, 5, 900,  f1=2500, f2=0.65, f3=0.60, f4=0.55, f5=180,
               delta_h=0.42, exceeds_tau_s=False, phase="stealth",    config="neural-only"),
        Window("t_peak",     0, 5, 5000, f1=9200, f2=0.45, f3=0.55, f4=0.40, f5=820,
               delta_h=2.10, exceeds_tau_s=True,  phase="saturation", config="neural-only"),
    ]
    f1_max = _collect_f1_max(windows)
    score_all(windows, f1_max)

    by_phase = {w.phase: w for w in windows}

    # Calibration target assertions
    assert by_phase["normal"].neural_only_risk < 0.50, (
        f"normal score {by_phase['normal'].neural_only_risk:.4f} should be < 0.50"
    )
    assert by_phase["flood_low"].neural_only_risk < 0.50, (
        f"flood_low score {by_phase['flood_low'].neural_only_risk:.4f} should be < 0.50"
    )
    assert 0.40 <= by_phase["stealth"].neural_only_risk <= 0.70, (
        f"stealth score {by_phase['stealth'].neural_only_risk:.4f} should be 0.40–0.70"
    )
    assert by_phase["saturation"].neural_only_risk > 0.70, (
        f"saturation score {by_phase['saturation'].neural_only_risk:.4f} should be > 0.70"
    )

    # Gate logic assertions
    assert by_phase["saturation"].gate_fired is True, "Gate should fire at saturation"
    assert by_phase["saturation"].neurosymbolic_risk >= by_phase["saturation"].neural_only_risk
    assert by_phase["normal"].gate_fired is False, "Gate must not fire on normal traffic"
    assert by_phase["normal"].neurosymbolic_risk <= by_phase["normal"].neural_only_risk


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="psi_comparison.py",
        description=(
            "Ψ operator comparison: Neural-Only vs Neuro-Symbolic pipeline.\n"
            "Reads *_windows.jsonl from data_dir, writes metrics CSV to out_dir."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        default="data/sliding_windows",
        help="Directory containing *_windows.jsonl files (default: data/sliding_windows)",
    )
    parser.add_argument(
        "--out_dir",
        default="results",
        help="Output directory for psi_comparison_metrics.csv (default: results)",
    )
    parser.add_argument(
        "--timeline",
        nargs=2,
        metavar=("SCENARIO", "TIER"),
        help=(
            "Print the risk timeline for a specific scenario and tier, "
            "e.g. --timeline escalation mid"
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        dest="self_test",
        help="Run calibration self-test and exit",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("[psi_comparison] Self-test passed.")
        return

    if args.timeline:
        scenario, tier = args.timeline
        timeline = get_risk_timeline(scenario, tier, args.data_dir)
        print(f"\nRisk timeline for {scenario}_{tier}:")
        print(f"{'#':>4}  {'window_id':>20}  {'phase':>20}  "
              f"{'neural_only':>12}  {'neurosymbolic':>14}  {'gate_fired':>10}")
        print("-" * 90)
        for i, (wid, ph, no, ns, gf) in enumerate(zip(
            timeline["windows"],
            timeline["phases"],
            timeline["neural_only"],
            timeline["neurosymbolic"],
            timeline["gate_fired"],
        )):
            print(
                f"{i:>4}  {wid:>20}  {ph:>20}  "
                f"{no:>12.4f}  {ns:>14.4f}  {str(gf):>10}"
            )
    else:
        run_comparison(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
