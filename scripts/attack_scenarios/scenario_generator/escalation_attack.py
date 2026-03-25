import random
import json
import argparse
import yaml
from dataclasses import dataclass, asdict
from typing import List

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PacketEvent:
    timestamp:    float
    src_ip:       str
    dst_ip:       str
    src_port:     int
    dst_port:     int
    protocol:     str
    payload_size: int
    
    # Metadata for debugging
    bot_id:       int
    phase:        int    # 0=stealth, 1=rampup, 2=saturation
    phase_name:   str


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def rand_known_port() -> int:
    return random.choice([80, 443, 53])

def rand_uniform_port() -> int:
    return random.randint(1024, 65535)


def rand_mixed_port(t: int, phase_duration: int) -> int:
    """
    During ramp-up, mix between known ports and random uniform distributions.
    """
    alpha = t / phase_duration  # 0.0 → 1.0
    if random.random() > alpha:
        return rand_known_port()
    else:
        return rand_uniform_port()

def get_random_ip() -> str:
    return f"{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"


# ---------------------------------------------------------------------------
# Phase generators
# ---------------------------------------------------------------------------

def generate_stealth_phase(config: dict, t_offset: int,
                            bot_ips: List[str]) -> List[PacketEvent]:
    """
    Phase 0 — Stealth
    Low rate, normal port distribution.
    """
    events: List[PacketEvent] = []
    duration   = config["phase_stealth_duration_s"] * 1000
    n_bots     = config["stealth_n_bots"]
    rate       = config["stealth_rate_pps"]
    
    t = t_offset
    dt = round(1000 / rate)  # inter-packet gap (ms)
    
    while t < t_offset + duration:
        for bot_id in range(n_bots):
            jitter = random.randint(0, dt)
            events.append(PacketEvent(
                timestamp    = t + jitter,
                src_ip       = bot_ips[bot_id],
                dst_ip       = config["target_ip"],
                src_port     = rand_uniform_port(),
                dst_port     = rand_known_port(),
                protocol     = "UDP",
                payload_size = random.randint(
                                   config["stealth_payload_min"],
                                   config["stealth_payload_max"]),
                bot_id       = bot_id,
                phase        = 0,
                phase_name   = "stealth",
            ))
            t += dt

    return events


def generate_rampup_phase(config: dict, t_offset: int,
                           bot_ips: List[str]) -> List[PacketEvent]:
    """
    Phase 1 — Ramp-up
    Rate and active bots grow linearly. Port distribution shifts from
    normal to uniform.
    """
    events: List[PacketEvent] = []
    duration     = config["phase_rampup_duration_s"] * 1000
    n_bots_start = config["stealth_n_bots"]
    n_bots_end   = config["saturation_n_bots"]
    rate_start   = config["stealth_rate_pps"]
    rate_end     = config["saturation_rate_pps"]

    t = t_offset
    end_t = t_offset + duration

    while t < end_t:
        alpha        = (t - t_offset) / duration
        n_bots_now   = int(n_bots_start + alpha * (n_bots_end - n_bots_start))
        rate_now     = rate_start + alpha * (rate_end - rate_start)
        dt_now       = round(1000 / rate_now)  # inter-packet gap (ms)
        elapsed_t = t - t_offset

        for bot_id in range(n_bots_now):
            jitter = random.randint(0, dt_now)
            events.append(PacketEvent(
                timestamp    = t + jitter,
                src_ip       = bot_ips[bot_id] if bot_id < len(bot_ips)
                                else get_random_ip(),
                dst_ip       = config["target_ip"],
                src_port     = rand_uniform_port(),
                dst_port     = rand_mixed_port(elapsed_t, duration),
                protocol     = "UDP",
                payload_size = random.randint(
                                   config["stealth_payload_min"],
                                   config["saturation_payload_max"]),
                bot_id       = bot_id,
                phase        = 1,
                phase_name   = "rampup",
            ))
            t += dt_now

    return events


def generate_saturation_phase(config: dict, t_offset: int,
                               bot_ips: List[str]) -> List[PacketEvent]:
    """
    Phase 2 — Saturation
    Full flood at max rate
    """
    events: List[PacketEvent] = []
    duration = config["phase_saturation_duration_s"] * 1000
    n_bots   = config["saturation_n_bots"]
    rate     = config["saturation_rate_pps"]

    t = t_offset
    dt = round(1000 / rate)  # inter-packet gap (ms)
    while t < t_offset + duration:
        for bot_id in range(n_bots):
            jitter = random.randint(0, dt)
            events.append(PacketEvent(
                timestamp    = t + jitter,
                src_ip       = bot_ips[bot_id] if bot_id < len(bot_ips)
                                else get_random_ip(),
                dst_ip       = config["target_ip"],
                src_port     = rand_uniform_port(),
                dst_port     = rand_uniform_port(),
                protocol     = "UDP",
                payload_size = random.randint(
                                   config["saturation_payload_min"],
                                   config["saturation_payload_max"]),
                bot_id       = bot_id,
                phase        = 2,
                phase_name   = "saturation",
            ))
            t += dt

    return events


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

class EscalationAttackGenerator:

    def __init__(self, config: dict):
        self.config = config

    def build_bot_pool(self) -> List[str]:
        n = self.config["saturation_n_bots"]
        return [get_random_ip() for _ in range(n)]

    def generate(self) -> List[PacketEvent]:
        bot_ips = self.build_bot_pool()
        events  = []

        t0 = 0
        phase0 = generate_stealth_phase   (self.config, t0,   bot_ips)
        t1     = t0 + self.config["phase_stealth_duration_s"] * 1000
        phase1 = generate_rampup_phase    (self.config, t1,   bot_ips)
        t2     = t1 + self.config["phase_rampup_duration_s"] * 1000
        phase2 = generate_saturation_phase(self.config, t2,   bot_ips)

        events = phase0 + phase1 + phase2

        print(f"[escalation] Phase 0 (stealth):     {len(phase0):>7,} events")
        print(f"[escalation] Phase 1 (ramp-up):     {len(phase1):>7,} events")
        print(f"[escalation] Phase 2 (saturation):  {len(phase2):>7,} events")
        print(f"[escalation] Total:                 {len(events):>7,} events")
        return events

    def create_log(self, events: List[PacketEvent], path: str):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, "w") as f:
            for e in events:
                f.write(json.dumps(asdict(e)) + "\n")
        print(f"[escalation] JSONL → {path}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scenario_config.yaml")
    parser.add_argument("--output", default="logs/escalation_attack_simulation.log")
    args = parser.parse_args()
    
    config = yaml.safe_load(open(args.config))

    random.seed(config.get("seed", 42)) # for reproducibility
    
    gen    = EscalationAttackGenerator(config)
    events = gen.generate()
    
    gen.create_log(events, args.output)