import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Generator, Iterator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_KEYS = [
    "traffic_rate_pps",
    "entropy_dst_port",
    "entropy_src_ip",
    "entropy_payload_sz",
    "std_ipg_ms",
]

DEFAULT_FIELDS = {
    "timestamp":    "timestamp",
    "src_ip":       "src_ip",
    "dst_ip":       "dst_ip",
    "dst_port":     "dst_port",
    "payload_size": "payload_size",
    "scenario":     "scenario",
}


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def shannon_entropy(values: list) -> float:
    """
    Shannon entropy H = -Σ p_i * log2(p_i)  (bits).
    Returns 0.0 for empty or single-element sequences.
    """
    n = len(values)
    if n == 0:
        return 0.0
    counts = Counter(values)
    return -sum(
        (c / n) * math.log2(c / n)
        for c in counts.values()
        if c > 0
    )


def std_dev(values: list[float]) -> float:
    """
    Population standard deviation.
    Returns 0.0 for sequences shorter than 2.
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    return math.sqrt(variance)


def inter_packet_gaps_ms(timestamps: list[int]) -> list[int]:
    """
    Given an ordered list of packet timestamps,
    return the list of consecutive inter-packet gaps in milliseconds.
    """
    sorted_ts = sorted(timestamps)
    return [(sorted_ts[i + 1] - sorted_ts[i])
            for i in range(len(sorted_ts) - 1)]


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def iter_packets(
    path: Path
) -> Iterator[dict]:
    """
    Yield normalised packet dicts from a JSONL file.
    Skips blank lines and lines that cannot be parsed.
    """
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"[WARN] line {lineno}: JSON parse error – {exc}", file=sys.stderr)
                continue

            # Normalise time to seconds (from ms)
            #obj["timestamp"] = obj["timestamp"] / 1000.0
            yield obj


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def sliding_windows(
    packets: list[dict],
    window_ms: int,
    stride_ms: int,
) -> Generator[tuple[float, float, list[dict]], None, None]:
    """
    Yield (t_start, t_end, window_packets) for a sliding window over
    a sorted list of packet dicts.
    """
    if not packets:
        return

    t_min = packets[0]["timestamp"]
    t_max = packets[-1]["timestamp"]

    t_start = t_min
    t_end = t_start + window_ms
        
    while t_end != t_max:
        t_end = t_start + window_ms
        if t_end > t_max:
            t_end = t_max
    
        window = [p for p in packets if t_start <= p["timestamp"] < t_end]
        yield t_start, t_end, window
        t_start += stride_ms


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(
    t_start: int,
    t_end: int,
    window: list[dict],
    window_id: int
) -> dict | None:
    """
    Compute the 5 Ψ-Risk-DT features for one time window.
    Returns None if the window has fewer than min_packets packets.
    """
    n = len(window)

    duration = t_end - t_start  # seconds

    # ── Feature 1: Traffic rate (pkt/s) ──────────────────────────────────
    traffic_rate = n / duration if duration > 0 else 0.0

    # ── Feature 2: Shannon entropy of destination ports ───────────────────
    dst_ports = [p["dst_port"] for p in window]
    h_dst_port = shannon_entropy(dst_ports)

    # ── Feature 3: Shannon entropy of source IPs ──────────────────────────
    src_ips = [p["src_ip"] for p in window]
    h_src_ip = shannon_entropy(src_ips)

    # ── Feature 4: Shannon entropy of payload sizes ───────────────────────
    payload_sizes = [p["payload_size"] for p in window]
    h_payload = shannon_entropy(payload_sizes)

    # ── Feature 5: Std-dev of inter-packet gaps (ms) ──────────────────────
    timestamps = [p["timestamp"] for p in window]
    ipgs = inter_packet_gaps_ms(timestamps)
    print(f"DEBUG: window {window_id}  ipgs={ipgs}", file=sys.stderr)
    std_ipg = std_dev(ipgs)

    return {
        "window_id":          window_id,
        "t_start":            round(t_start / 1000.0, 4),
        "t_end":              round(t_end / 1000.0, 4),
        "n_packets":          n,
        "traffic_rate_pps":   round(traffic_rate),
        "entropy_dst_port":   round(h_dst_port, 6),
        "entropy_src_ip":     round(h_src_ip, 6),
        "entropy_payload_sz": round(h_payload, 6),
        "std_jitter_ms":      round(std_ipg)
    }


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(records: list[dict], input_path: Path, output_path: Path) -> None:
    if not records:
        print("[INFO] No windows produced.", file=sys.stderr)
        return

    def col(key: str) -> list[float]:
        return [r[key] for r in records]

    def _stats(values: list[float]) -> str:
        n = len(values)
        mn, mx = min(values), max(values)
        mean = sum(values) / n
        return f"min={mn:.3f}  mean={mean:.3f}  max={mx:.3f}"

    lines = [
        "",
        "═" * 60,
        f"  batch_formatter  –  summary",
        "═" * 60,
        f"  Input  : {input_path}",
        f"  Output : {output_path}",
        f"  Windows: {len(records)}",
        f"  Packets/window  {_stats(col('n_packets'))}",
        "",
        "  Features:",
        f"    traffic_rate_pps   {_stats(col('traffic_rate_pps'))}",
        f"    entropy_dst_port   {_stats(col('entropy_dst_port'))}",
        f"    entropy_src_ip     {_stats(col('entropy_src_ip'))}",
        f"    entropy_payload_sz {_stats(col('entropy_payload_sz'))}",
        f"    std_jitter_ms      {_stats(col('std_jitter_ms'))}",
        "═" * 60,
        "",
    ]
    print("\n".join(lines), file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_output_path(input_path: Path, output_arg: str | None) -> Path:
    if output_arg:
        p = Path(output_arg)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    stem = input_path.stem
    return input_path.with_name(f"{stem}_features.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw attack-scenario JSONL → Ψ-Risk-DT feature windows.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input",
        metavar="INPUT.jsonl",
        help="Raw packet-event JSONL file to process",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="OUTPUT.jsonl",
        default=None,
        help="Output path (default: <input_stem>_features.jsonl alongside input)",
    )
    parser.add_argument(
        "--window", "-w",
        type=float,
        default=1.0,
        metavar="MILLISECONDS",
        help="Sliding window width in ms (default: 1.0)",
    )
    parser.add_argument(
        "--stride", "-s",
        type=float,
        default=None,
        metavar="MILLISECONDS",
        help="Window stride in ms (default: equal to --window, i.e. tumbling)",
    )
    args = parser.parse_args()

    # ── Resolve paths ──────────────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"[FATAL] Input file not found: {input_path}")

    output_path = build_output_path(input_path, args.output)
    stride = args.stride if args.stride is not None else round(args.window / 2)

    if stride <= 0 or args.window <= 0:
        sys.exit("[FATAL] --window and --stride must be positive.")

    # ── Load & sort packets ────────────────────────────────────────────────
    print(f"[INFO] Reading {input_path} …", file=sys.stderr)
    try:
        packets = list(iter_packets(input_path))
    except ValueError as exc:
        sys.exit(f"[FATAL] {exc}")

    if not packets:
        sys.exit("[FATAL] Input file is empty or contains no valid packets.")

    #packets.sort(key=lambda p: p["timestamp"])
    print(f"[INFO] Loaded {len(packets):,} packets  "
          f"(T={packets[0]['timestamp'] / 1000.0:.3f}s … {packets[-1]['timestamp'] / 1000.0:.3f}s)",
          file=sys.stderr)
    print(f"[INFO] Window={args.window / 1000.0}s  Stride={stride / 1000.0}s  ",
          file=sys.stderr)

    # ── Extract windows ────────────────────────────────────────────────────
    records: list[dict] = []

    with open(output_path, "w", encoding="utf-8") as out_fh:
        for wid, (t_start, t_end, window) in enumerate(
            sliding_windows(packets, args.window, stride)
        ):
            rec = extract_features(t_start, t_end, window, wid)
            if rec is None:
                continue
            records.append(rec)
            out_fh.write(json.dumps(rec, indent=None, ensure_ascii=False))
            out_fh.write("\n")

    # ── Summary ────────────────────────────────────────────────────────────
    print_summary(records, input_path, output_path)
    #print(f"[INFO] Wrote {len(records):,} windows → {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()