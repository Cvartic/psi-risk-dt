import random
import json
from dataclasses import dataclass, asdict
from typing import List


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PacketEvent:
    timestamp:      int       # virtual clock in ms
    src_ip:         str
    dst_ip:         str
    src_port:       int
    dst_port:       int
    protocol:       str
    payload_size:   int
    device_id:      int
    traffic_class:  int       # 0=heartbeat, 1=dns_ntp, 2=cloud, 3=control
    class_name:     str
    label:          str = "normal"


# ---------------------------------------------------------------------------
# IP helpers
# ---------------------------------------------------------------------------

GATEWAY_IP   = "192.168.1.1"
CLOUD_IPS    = ["203.0.113.10", "203.0.113.20", "198.51.100.5"]
DNS_SERVER   = "8.8.8.8"
NTP_SERVER   = "216.239.35.0"

def device_ip(device_id: int, base: str = "192.168.1") -> str:
    return f"{base}.{10 + device_id}"


# ---------------------------------------------------------------------------
# Traffic class generators
# ---------------------------------------------------------------------------

def generate_heartbeat(config: dict, device_ips: List[str]) -> List[PacketEvent]:
    """
    Class 0 — Heartbeat
    Small, regular packets from each device to the gateway.
    Rate is low and stable; jitter is minimal.
    """
    events   = []
    duration = config["duration_s"] * 1000
    rate     = config["heartbeat_rate_pps"]
    dt       = round(1000 / rate)

    for device_id, src_ip in enumerate(device_ips):
        t = random.randint(0, dt)
        while t < duration:
            pkt_time = random.randint(t, t + dt)
            if pkt_time < duration:
                events.append(PacketEvent(
                    timestamp     = pkt_time,
                    src_ip        = src_ip,
                    dst_ip        = GATEWAY_IP,
                    src_port      = random.randint(49152, 65535),
                    dst_port      = random.choice([8883, 1883, 9000]),
                    protocol      = "UDP",
                    payload_size  = random.randint(
                                        config["heartbeat_payload_min"],
                                        config["heartbeat_payload_max"]),
                    device_id     = device_id,
                    traffic_class = 0,
                    class_name    = "heartbeat",
                ))
            t += dt

    return events


def generate_dns_ntp(config: dict, device_ips: List[str]) -> List[PacketEvent]:
    """
    Class 1 — DNS / NTP
    Rare, small packets to well-known ports 53 and 123.
    """
    events   = []
    duration = config["duration_s"] * 1000
    rate     = config["dns_ntp_rate_pps"]
    dt       = round(1000 / rate)

    for device_id, src_ip in enumerate(device_ips):
        t = random.randint(0, dt)
        while t < duration:
            pkt_time = random.randint(t, t + dt)
            if pkt_time < duration:
                is_dns   = random.random() < 0.6
                events.append(PacketEvent(
                    timestamp     = pkt_time,
                    src_ip        = src_ip,
                    dst_ip        = DNS_SERVER if is_dns else NTP_SERVER,
                    src_port      = random.randint(49152, 65535),
                    dst_port      = 53 if is_dns else 123,
                    protocol      = "UDP",
                    payload_size  = random.randint(28, 64),
                    device_id     = device_id,
                    traffic_class = 1,
                    class_name    = "dns_ntp",
                ))
            t += dt

    return events


def generate_cloud_transfer(config: dict, device_ips: List[str]) -> List[PacketEvent]:
    """
    Class 2 — Cloud transfer
    Moderate rate toward cloud endpoints ({443, 1883}).
    Occasional short bursts simulate data upload events.
    """
    events   = []
    duration = config["duration_s"] * 1000
    rate     = config["cloud_rate_pps"]
    dt       = round(1000 / rate)

    for device_id, src_ip in enumerate(device_ips):
        t = random.randint(0, dt)
        cloud_dst = random.choice(CLOUD_IPS)

        while t < duration:
            pkt_time = random.randint(t, t + dt)
            if pkt_time < duration:
                events.append(PacketEvent(
                    timestamp     = pkt_time,
                    src_ip        = src_ip,
                    dst_ip        = cloud_dst,
                    src_port      = random.randint(49152, 65535),
                    dst_port      = random.choice([443, 1883]),
                    protocol      = "UDP",
                    payload_size  = random.randint(128, 512),
                    device_id     = device_id,
                    traffic_class = 2,
                    class_name    = "cloud_transfer",
                ))
            t += dt

    return events


def generate_control_traffic(config: dict, device_ips: List[str]) -> List[PacketEvent]:
    """
    Class 3 — Control inter-device
    Small packets exchanged between pairs of devices in the local subnet.
    Pattern is quasi-periodic and stable.
    """
    events   = []
    duration = config["duration_s"] * 1000
    rate     = config["control_rate_pps"]
    dt       = round(1000 / rate)
    n        = len(device_ips)

    # build fixed pairs of devices (stable for whole duration)
    pairs = [(i, (i + 1) % n) for i in range(0, n, 2)]

    for src_idx, dst_idx in pairs:
        t = random.randint(0, dt)
        while t < duration:
            pkt_time = random.randint(t, t + dt)
            if pkt_time < duration:
                events.append(PacketEvent(
                    timestamp     = pkt_time,
                    src_ip        = device_ips[src_idx],
                    dst_ip        = device_ips[dst_idx],
                    src_port      = random.randint(49152, 65535),
                    dst_port      = random.choice([5683, 5684, 9000]),
                    protocol      = "UDP",
                    payload_size  = random.randint(32, 128),
                    device_id     = src_idx,
                    traffic_class = 3,
                    class_name    = "control",
                ))
            t += dt

    return events


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

class NormalTrafficGenerator:

    def __init__(self, config: dict):
        self.config = config

    def build_device_pool(self) -> List[str]:
        n    = self.config["n_devices"]
        base = self.config.get("subnet_base", "192.168.1")
        return [device_ip(i, base) for i in range(n)]

    def generate(self) -> List[PacketEvent]:
        device_ips = self.build_device_pool()
        events: List[PacketEvent] = []

        hb  = generate_heartbeat      (self.config, device_ips)
        dns = generate_dns_ntp        (self.config, device_ips)
        cld = generate_cloud_transfer (self.config, device_ips)
        ctl = generate_control_traffic(self.config, device_ips)

        events = hb + dns + cld + ctl
        events.sort(key=lambda e: e.timestamp)

        print(f"[normal] Heartbeat:       {len(hb):>7,} events")
        print(f"[normal] DNS/NTP:         {len(dns):>7,} events")
        print(f"[normal] Cloud transfer:  {len(cld):>7,} events")
        print(f"[normal] Control:         {len(ctl):>7,} events")
        print(f"[normal] Total:           {len(events):>7,} events")
        return events

    def create_log(self, events: List[PacketEvent], path: str):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as f:
            for e in events:
                f.write(json.dumps(asdict(e)) + "\n")
        print(f"[normal] JSONL → {path}  ({len(events)} events)")


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(yaml_path: str) -> dict:
    try:
        import yaml
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("normal_traffic", {})
    except ImportError:
        print("[warning] PyYAML not found — using CLI defaults.")
        return {}
    except FileNotFoundError:
        print(f"[warning] Config file '{yaml_path}' not found — using CLI defaults.")
        return {}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    
    import os
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    os.chdir("..") # scripts root
    
    config = load_config("baseline_generator/baseline_config.yaml")
    gen    = NormalTrafficGenerator(config)
    events = gen.generate()
    gen.create_log(events, "../data/baseline/normal_traffic_simulation.jsonl")