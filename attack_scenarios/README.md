# attack_scenarios

Attack scenario generators for stress testing the Ψ-Risk-DT framework.

## Structure
- `scenario_config.yaml` - configuration file of the base generators
- `scenario_campaigns_config.yaml` - configuration file of the scenario campaigns generator
- `scenario_generator/` - attack modules
  - `udp_flood.py` - UDP volumetric flood attack pattern for high packet rate stress testing
  - `entropy_anomaly.py` - zero-day like attack with unusual entropy characteristics
  - `escalation_attack.py` - structured attack divided in 3 distinct phases (Stealth, Ramp-up, Saturation)
- `scenario_campaigns/`
  - `scenario_capaigns_generator.py` - 3 different attack scenarios with various intensity applied to each generator, produces 9 output files

## Usage
1. Install `requirements.txt`
2. Adjust `scenario_config.yaml` and `scenario_campaigns_config.yaml` if necessary.
3. Run the campaigns generator module with `python scenario_campaigns/scenario_campaigns_generator.py`.
