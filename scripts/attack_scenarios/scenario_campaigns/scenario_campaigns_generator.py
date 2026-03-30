import copy
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml



GENERATOR_MAP = {
    "udp_flood":       "scenario_generator/udp_flood.py",
    "escalation_attack": "scenario_generator/escalation_attack.py",
    "zero_day_like":   "scenario_generator/entropy_anomaly.py",
}

CONFIG_SECTION_MAP = {
    "udp_flood":       "udp_flood",
    "escalation_attack": "escalation_attack",
    "zero_day_like":   "entropy_anomaly",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_run_config(base_config: dict, section_key: str, campaign_params: dict) -> dict:
    """
    Return a deep-copied base_config with the relevant section
    overridden by campaign_params.
    """
    cfg = copy.deepcopy(base_config)
    cfg = {**cfg.get(section_key, {}), **campaign_params}
    return cfg


def write_temp_config(cfg: dict) -> Path:
    """Write cfg to a temp YAML file and return its Path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="campaign_cfg_"
    )
    yaml.dump(cfg, tmp, default_flow_style=False, allow_unicode=True)
    tmp.close()
    return Path(tmp.name)


def run_generator(script: Path, config_path: Path, output_file: Path) -> bool:
    """
    Invoke `python <script> --config <config_path> --output <output_file>`.
    """
    cmd = [
        sys.executable, str(script),
        "--config", str(config_path),
        "--output", str(output_file),
    ]
    print(f"  CMD: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            print(f"  [ERROR] Generator exited {result.returncode}")
            print(f"  STDERR:\n{result.stderr}")
            return False
        print(f"  [OK] → {output_file}")
        return True
    except FileNotFoundError:
        print(f"  [ERROR] Script not found: {script}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # move to attack scenarios root for consistent paths
    import os
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    os.chdir("..") 

    
    campaigns_path = Path("scenario_campaigns_config.yaml")
    base_config_path = Path("scenario_config.yaml")
    output_dir = Path("../../data/attack_scenarios")

    # Validate inputs
    if not campaigns_path.exists():
        sys.exit(f"[FATAL] Campaign file not found: {campaigns_path}")
    if not base_config_path.exists():
        sys.exit(f"[FATAL] Base config not found: {base_config_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    campaigns = load_yaml(campaigns_path)
    base_cfg   = load_yaml(base_config_path)

    # Strip internal YAML anchors / meta keys that start with '_'
    scenario_keys = [k for k in campaigns if not k.startswith("_")]

    passed = 0
    failed = 0

    print(f"\n{'='*60}")
    print(f"  Campaign runner  –  {len(scenario_keys)} scenarios")
    print(f"  Output dir: {output_dir.resolve()}")

    for scenario_key, tiers in campaigns.items():
        if scenario_key.startswith("_") or not isinstance(tiers, dict):
            continue

        script_rel  = GENERATOR_MAP.get(scenario_key)
        config_sect = CONFIG_SECTION_MAP.get(scenario_key)

        if script_rel is None:
            print(f"[WARN] No generator mapped for '{scenario_key}' – skipping\n")
            continue

        script_path = Path(script_rel)

        for tier_name, tier_def in tiers.items():
            output_filename = tier_def.get("output_file", f"{scenario_key}_{tier_name}.jsonl")
            campaign_params = tier_def.get("params", {})

            output_file = output_dir / output_filename

            print(f"▶  {scenario_key} / {tier_name}")
            print(f"   Params: {campaign_params}")

            # Build merged config and write to temp file
            run_cfg      = build_run_config(base_cfg, config_sect, campaign_params)
            tmp_cfg_path = write_temp_config(run_cfg)

            try:
                ok = run_generator(script_path, tmp_cfg_path, output_file)
            finally:
                tmp_cfg_path.unlink(missing_ok=True)   # clean up temp

            if ok:
                passed += 1
            else:
                failed += 1
            print()

    print(f"{'='*60}")
    print(f"  Done  –  {passed} succeeded, {failed} failed")
    print(f"  JSONL files in: {output_dir.resolve()}")
    print(f"{'='*60}\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()