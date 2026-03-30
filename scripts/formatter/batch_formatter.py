import argparse
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# The 9 canonical campaign filenames
# ---------------------------------------------------------------------------
CAMPAIGN_FILES = [
    "udp_flood_low.jsonl",
    "udp_flood_mid.jsonl",
    "udp_flood_high.jsonl",
    "escalation_low.jsonl",
    "escalation_mid.jsonl",
    "escalation_high.jsonl",
    "zero_day_like_low.jsonl",
    "zero_day_like_mid.jsonl",
    "zero_day_like_high.jsonl",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_output_name(campaign_file: str) -> str:
    """udp_flood_low.jsonl  →  udp_flood_low_windows.jsonl"""
    stem = Path(campaign_file).stem
    return f"{stem}_windows.jsonl"


def run_formatter(
    formatter: Path,
    input_file: Path,
    output_file: Path,
    window: float,
    stride: float,
) -> bool:
    """
    Call formatter.py as a subprocess.
    Returns success.
    """
    cmd = [
        sys.executable, str(formatter),
        str(input_file),
        "--output", str(output_file),
        "--window", str(window),
        "--stride", str(stride),
    ]

    print(f"  $ {' '.join(cmd)}")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Forward stderr (formatter prints its summary there)
    if result.stderr:
        for line in result.stderr.splitlines():
            print(f"  {line}")

    if result.returncode != 0:
        print(f"  [ERROR] exit code\n")
        return False

    print(f"  [OK] → {output_file}\n")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import os
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    os.chdir("../..")  # project root
    print(f"Current working directory: {Path.cwd()}")
    
    parser = argparse.ArgumentParser(
        description="Run batch_formatter.py on all 9 campaign JSONL files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--campaigns",
        default="data/attack_scenarios",
        metavar="DIR",
        help="Directory containing the 9 campaign JSONL files (default: data/attack_scenarios/)",
    )
    parser.add_argument(
        "--features",
        default="data/sliding_windows",
        metavar="DIR",
        help="Output directory for feature JSONL files (default: data/sliding_windows/)",
    )
    parser.add_argument(
        "--window", "-w",
        type=float,
        default=5000.0,
        metavar="SECONDS",
        help="Sliding window width in seconds (default: 5000.0)",
    )
    parser.add_argument(
        "--stride", "-s",
        type=float,
        default= None,
        metavar="SECONDS",
        help="Window stride in seconds (default: equal to --window / 2)",
    )
    args = parser.parse_args()

    campaigns_dir = Path(args.campaigns)
    features_dir  = Path(args.features)
    formatter     = Path("scripts/formatter/formatter.py")
    stride        = args.stride if args.stride is not None else args.window / 2

    # ── Validate ────────────────────────────────────────────────────────────
    if not formatter.exists():
        sys.exit(f"[FATAL] formatter.py not found at: {formatter}")
    if args.window <= 0 or stride <= 0:
        sys.exit("[FATAL] --window and --stride must be positive.")

    features_dir.mkdir(parents=True, exist_ok=True)

    # ── Header ──────────────────────────────────────────────────────────────
    print()
    print("═" * 64)
    print("  run_all_campaigns  –  batch feature extraction")
    print("═" * 64)
    print(f"  Campaigns dir : {campaigns_dir.resolve()}")
    print(f"  Features dir  : {features_dir.resolve()}")
    print(f"  Formatter     : {formatter.resolve()}")
    print(f"  Window / stride: {args.window}s / {stride}s")
    print("═" * 64)
    print()

    # ── Process each campaign ────────────────────────────────────────────────
    results: list[tuple[str, bool, str]] = []   # (name, ok, secs, note)

    for filename in CAMPAIGN_FILES:
        input_file  = campaigns_dir / filename
        output_file = features_dir  / build_output_name(filename)

        print(f"▶  {filename}")

        if not input_file.exists():
            note = "SKIPPED – file not found"
            print(f"  [WARN] {input_file} not found, skipping.\n")
            results.append((filename, False, 0.0, note))
            continue

        ok = run_formatter(
            formatter, input_file, output_file,
            args.window, stride,
        )
        results.append((filename, ok, "OK" if ok else "FAILED"))

    # ── Final summary table ──────────────────────────────────────────────────
    passed  = sum(1 for _, ok, _ in results if ok)
    failed  = sum(1 for _, ok, note in results if not ok and note != "SKIPPED – file not found")
    skipped = sum(1 for _, _, note in results if note == "SKIPPED – file not found")

    print("═" * 64)
    print(f"  {'Campaign file':<35} {'Status':<10}")
    print(f"  {'─' * 35} {'─' * 10} {'─' * 6}")
    for name, ok, note in results:
        status = note if not ok else "OK"
        print(f"  {name:<35} {status:<10}")
    print("═" * 64)
    print(f"  {passed} succeeded  |  {failed} failed  |  {skipped} skipped")
    print(f"  Feature files in: {features_dir.resolve()}")
    print("═" * 64)
    print()

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()