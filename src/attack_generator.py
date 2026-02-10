import numpy as np
from typing import List


class AttackGenerator:
    """Generates attacks following Poisson process"""

    def __init__(self, attack_rate: float, packets_per_attack: int):
        """
        Args:
            attack_rate: λ, average attacks per minute
            packets_per_attack: n, packets in each attack
        """
        self.attack_rate = attack_rate
        self.packets_per_attack = packets_per_attack
        self.attacks_generated = 0

    def generate_attack_times(self, start_time: float, duration: float) -> List[float]:
        """
        Generate attack arrival times using Poisson process
        Returns:
            List of attack times in [start_time, start_time + duration]
        """
        attack_times: List[float] = []
        current_time = start_time
        end_time = start_time + duration
        while current_time < end_time:
            # Inter-arrival time follows Exponential(λ)
            inter_arrival = np.random.exponential(1.0 / self.attack_rate)
            current_time += inter_arrival
            if current_time < end_time:
                attack_times.append(current_time)
                self.attacks_generated += 1
        return attack_times

    def create_attack_packets(self, attack_id: int, attack_time: float) -> List[dict]:
        """Create packets for a single attack"""
        packets: List[dict] = []
        for i in range(self.packets_per_attack):
            packet = {
                "id": f"attack{attack_id}_packet{i}",
                "attack_id": attack_id,
                "timestamp": attack_time,
                "size_kb": 1.0,
                "malicious": True,
                "detected": False,
                "dropped": False,
            }
            packets.append(packet)
        return packets

