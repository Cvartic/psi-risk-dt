import random
import math
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
    timestamp:    int       # virtual clock in ms
    src_ip:       str
    dst_ip:       str
    src_port:     int
    dst_port:     int
    protocol:     str
    payload_size: int
    
    # Metadata for debugging
    bot_id:       int
    burst_id:     int       # index of the burst that generated this packet (-1 = leakage)
    is_silence:  bool      # True if emitted during a silence


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_random_ip() -> str:
    return f"{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"


def get_port(peaks: List[int], sigma: float = 20) -> int:
    """Sample a port from a gaussian mixture centered on anomalous peaks."""
    mu = random.choice(peaks)
    p  = int(random.gauss(mu, sigma))
    return max(1, min(65535, p))


def get_payload_size(clusters: List[List[int]]) -> int:
    """Sample payload size from a trimodal distribution."""
    mu, sigma = random.choice(clusters)
    size = int(random.gauss(mu, sigma))
    return max(40, min(1400, size))


# ---------------------------------------------------------------------------
# Burst / silence schedule
# ---------------------------------------------------------------------------

def build_burst_schedule(config: dict) -> List[dict]:
    """
    Build a timeline of burst/silence intervals using a non-homogeneous
    Poisson process for silence durations.

    Returns a list of dicts:
      { 'type': 'burst'|'silence', 'start': int (ms), 'duration': int (ms), 'burst_id': int }
    """
    schedule   = []
    t          = 0                              # ms
    burst_id   = 0
    total_ms   = config["duration_s"] * 1000

    while t < total_ms:
        # --- burst ---
        burst_dur = random.randint(
            config["burst_duration_min_ms"],
            config["burst_duration_max_ms"]
        )
        # every ~3 bursts, a longer burst to simulate a wave
        if burst_id % 3 == 2:
            burst_dur = round(burst_dur * random.uniform(1.5, 2.5))

        burst_dur = min(burst_dur, total_ms - t)
        schedule.append({
            "type":     "burst",
            "start":    t,
            "duration": burst_dur,
            "burst_id": burst_id,
        })
        t        += burst_dur
        burst_id += 1

        if t >= total_ms:
            break

        # --- silence ---
        # lambda varies sinusoidally over the attack duration to create a mix of short and long silences
        lambda_base = 1000.0 / config["silence_mean_ms"]
        lambda_now  = lambda_base * (1 + 0.5 * math.sin(2 * math.pi * t / total_ms))
        silence_dur = round(random.expovariate(lambda_now))
        silence_dur = min(
            max(silence_dur, config["silence_min_ms"]),
            config["silence_max_ms"]
        )
        silence_dur = min(silence_dur, total_ms - t)
        schedule.append({
            "type":     "silence",
            "start":    t,
            "duration": silence_dur,
            "burst_id": burst_id,
        })
        t += silence_dur

    return schedule


# ---------------------------------------------------------------------------
# Phase generators
# ---------------------------------------------------------------------------

def generate_burst_packets(interval: dict, config: dict,
                            src_ips: List[str]) -> List[PacketEvent]:
    """Generate packets for a single burst interval."""
    events   = []
    t        = interval["start"]            # ms
    end_t    = t + interval["duration"]     # ms
    burst_id = interval["burst_id"]
    n_srcs   = config["n_sources"]
    rate     = config["burst_rate_pps"]
    dt       = round(1000 / rate)           # inter-packet gap

    while t < end_t:
        # activate a random subset of sources each tick
        active = random.sample(range(n_srcs), k=random.randint(1, n_srcs))
        for bot_id in active:
            pkt_time = random.randint(t, t + dt)
            if pkt_time >= end_t:
                continue
            events.append(PacketEvent(
                timestamp    = pkt_time,
                src_ip       = src_ips[bot_id],
                dst_ip       = config["target_ip"],
                src_port     = random.randint(1024, 65535),
                dst_port     = get_port(
                                   config["sus_ports"],
                                   config["port_sigma"]),
                protocol     = "UDP",
                payload_size = get_payload_size(config["payload_clusters"]),
                bot_id       = bot_id,
                burst_id     = burst_id,
                is_silence  = False,
            ))
            t += dt

    return events


def generate_silence_traffic(interval: dict, config: dict,
                              src_ips: List[str]) -> List[PacketEvent]:
    """
    During silences, emit a few packets.
    Same anomalous port distribution,
    always small payload.
    """
    events    = []
    t         = interval["start"]
    end_t     = t + interval["duration"]
    burst_id  = -1  # not associated with any burst
    leak_rate = config.get("beacon_leak_rate_pps", 10)
    dt        = round(1000 / leak_rate)     # inter-packet gap (ms)

    while t < end_t:
        bot_id = random.randint(0, config["n_sources"] - 1)
        pkt_time = random.randint(t, t + dt)
        if pkt_time < end_t:
            events.append(PacketEvent(
                timestamp    = pkt_time,
                src_ip       = src_ips[bot_id],
                dst_ip       = config["target_ip"],
                src_port     = random.randint(1024, 65535),
                dst_port     = get_port(
                                    config["sus_ports"],
                                    config["port_sigma"]),
                protocol     = "UDP",
                payload_size = 64, # always small packets
                bot_id       = bot_id,
                burst_id     = burst_id,
                is_silence  = True,
            ))
        t += dt

    return events


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

class EntropyAnomalyGenerator:

    def __init__(self, config: dict):
        self.config = config

    def build_source_pool(self) -> List[str]:
        """Stable source IPs — not heavily spoofed."""
        return [get_random_ip() for _ in range(self.config["n_sources"])]

    def generate(self) -> List[PacketEvent]:
        src_ips  = self.build_source_pool()
        schedule = build_burst_schedule(self.config)
        events: List[PacketEvent] = []

        n_bursts   = sum(1 for i in schedule if i["type"] == "burst")
        n_silences = sum(1 for i in schedule if i["type"] == "silence")
        print(f"[entropy_anomaly] Schedule: {n_bursts} bursts, {n_silences} silences")

        for interval in schedule:
            if interval["type"] == "burst":
                events += generate_burst_packets  (interval, self.config, src_ips)
            else:
                events += generate_silence_traffic(interval, self.config, src_ips)

        events.sort(key=lambda e: e.timestamp)

        burst_pkts   = sum(1 for e in events if not e.is_silence)
        silence_pkts = sum(1 for e in events if e.is_silence)
        print(f"[entropy_anomaly] Burst packets:   {burst_pkts:>7,}")
        print(f"[entropy_anomaly] Silence packets: {silence_pkts:>7,}")
        print(f"[entropy_anomaly] Total:            {len(events):>7,}")
        return events

    def create_log(self, events: List[PacketEvent], path: str):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as f:
            for e in events:
                f.write(json.dumps(asdict(e)) + "\n")
        print(f"[entropy_anomaly] JSONL → {path}  ({len(events)} events)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scenario_config.yaml")
    parser.add_argument("--output", default="logs/entropy_anomaly_simulation.log")
    args = parser.parse_args()
    
    config = yaml.safe_load(open(args.config))
    
    random.seed(config.get("seed", 42))  # for reproducibility

    gen    = EntropyAnomalyGenerator(config)
    events = gen.generate()
    
    gen.create_log(events, args.output)