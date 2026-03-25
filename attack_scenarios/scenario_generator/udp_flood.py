import argparse
import yaml
import random
import json
from dataclasses import dataclass, asdict
from typing import List

@dataclass
class PacketEvent:
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    payload_size: int
    
    # Metadata for debugging
    bot_id: int
    step: int


class UDPFloodGenerator:
    """
    Generates a synthetic UDP flood event log following a step-wise
    escalation model (Mirai-like botnet behaviour).
    """

    def __init__(self, config: dict):
        self.n_bots: int         = config.get("n_bots", 100)
        self.base_rate: int      = config.get("base_rate_pps", 200)   # pkt/s per bot
        self.duration: int       = config.get("duration_s", 120) * 1000
        self.step_interval: int  = config.get("escalation_step_s", 10) * 1000
        self.rate_increment: int = config.get("rate_increment_pps", 100)
        self.max_rate: int       = config.get("max_rate_pps", 1000)
        self.payload_min: int    = config.get("payload_min_bytes", 64)
        self.payload_max: int    = config.get("payload_max_bytes", 1400)
        self.dst_ip: str         = config.get("target_ip", "192.168.1.1")

    def get_random_ip(self) -> str:
        return f"{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"

    def get_current_rate(self, cur_step) -> int:
        rate  = self.base_rate + cur_step * self.rate_increment
        return min(rate, self.max_rate)

    def get_current_step(self, elapsed: int) -> int:
        return int(elapsed // self.step_interval)


    def generate(self) -> List[PacketEvent]:
        """
        Returns a list of PacketEvent objects representing the full attack.
        Time is simulated (no real sleep), using a virtual clock.
        """
        events: List[PacketEvent] = []
        t = 0  # virtual clock (ms)

        bot_ips = [self.get_random_ip() for _ in range(self.n_bots)]

        while t < self.duration:
            cur_step  = self.get_current_step(t)
            rate   = self.get_current_rate(cur_step)
            dt = round(1000 / rate)  # inter-packet gap (ms)

            for _ in range(self.n_bots):
                bot_id = random.randint(0, self.n_bots - 1)
                pkt_time = random.randint(t, t + dt)
                if pkt_time >= self.duration:
                    break
                event = PacketEvent(
                    timestamp    = pkt_time,
                    src_ip       = bot_ips[bot_id],
                    dst_ip       = self.dst_ip,
                    src_port     = random.randint(1024, 65535),
                    dst_port     = random.randint(1024, 65535),
                    protocol     = "UDP",
                    payload_size = random.randint(self.payload_min, self.payload_max),
                    bot_id       = bot_id,
                    step        = cur_step,
                )
                events.append(event)
                t += dt  # increment time
                
        return events

    def create_log(self, events: List[PacketEvent], path: str):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, "w") as f:
            for e in events:
                f.write(json.dumps(asdict(e)) + "\n")
        print(f"[udp_flood] JSONL saved → {path}  ({len(events)} events)")



def load_config(yaml_path: str) -> dict:
    try:
        import yaml
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("udp_flood", {})
    except ImportError:
        print("[warning] PyYAML not found — using CLI defaults.")
        return {}
    except FileNotFoundError:
        print(f"[warning] Config file '{yaml_path}' not found — using CLI defaults.")
        return {}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scenario_config.yaml")
    parser.add_argument("--output", default="logs/udp_flood_simulation.log")
    args = parser.parse_args()
    
    config = yaml.safe_load(open(args.config))

    random.seed(config.get("seed", 42)) # for reproducibility

    gen    = UDPFloodGenerator(config)
    events = gen.generate()

    gen.create_log(events, args.output)