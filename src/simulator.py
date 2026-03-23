import heapq
from typing import List

import numpy as np
import random

from .network import Network
from .ids import IntrusionDetectionSystem
from .attack_generator import AttackGenerator
from .metrics import MetricsCollector


class Event:
    """Represents a simulation event"""

    def __init__(self, time: float, event_type: str, data: dict = None):
        self.time = time
        self.type = event_type
        self.data = data or {}

    def __lt__(self, other):
        """For heap queue sorting"""
        return self.time < other.time

    def __repr__(self):
        return f"Event(time={self.time:.2f}, type={self.type})"


class SimulationEngine:
    """Discrete-event simulation engine"""

    def __init__(self):
        self.current_time = 0.0
        self.event_queue: List[Event] = []  # Min-heap priority queue
        self.event_handlers = {}

    def schedule_event(self, time: float, event_type: str, data: dict = None):
        """Add an event to the queue"""
        if time < self.current_time:
            raise ValueError(f"Cannot schedule event in the past")
        event = Event(time, event_type, data)
        heapq.heappush(self.event_queue, event)

    def register_handler(self, event_type: str, handler_func):
        """Register a function to handle specific event type"""
        self.event_handlers[event_type] = handler_func

    def run(self, duration: float):
        """Run simulation for specified duration"""
        end_time = self.current_time + duration
        while self.event_queue and self.event_queue[0].time <= end_time:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time
            if event.type in self.event_handlers:
                self.event_handlers[event.type](event)
        self.current_time = end_time


class NetworkAttackSimulation:
    """Complete integrated network attack simulation"""

    def __init__(self, config: dict):
        self.config = config
        # Initialize all components
        self.engine = SimulationEngine()
        self.network = Network(
            bandwidth_mbps=config["network"]["bandwidth"],
            buffer_size=config["network"]["buffer_size"],
            degradation_alpha=config["network"].get("degradation_alpha", 0.0),
        )
        self.ids = IntrusionDetectionSystem(
            detection_probability=config["ids"]["detection_prob"]
        )
        self.attack_gen = AttackGenerator(
            attack_rate=config["attacks"]["rate"],
            packets_per_attack=config["attacks"]["packets_per_attack"],
        )
        self.metrics = MetricsCollector(
            sampling_interval=config["simulation"]["sampling_interval"]
        )
        # Phase 3 tracking state
        self.attack_times_min: List[float] = []
        self.attack_detected = {}
        # Register event handlers
        self.engine.register_handler("attack_start", self._handle_attack_start)
        self.engine.register_handler("packet_arrival", self._handle_packet_arrival)
        self.engine.register_handler("sample_metrics", self._handle_sample_metrics)

    def run(self, duration_minutes: float, seed: int = None):
        """Run complete simulation"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        # Reset run-scoped tracking
        self.attack_times_min = []
        self.attack_detected = {}
        duration_seconds = duration_minutes * 60
        # Generate attack schedule
        attack_times = self.attack_gen.generate_attack_times(0.0, duration_minutes)
        self.attack_times_min = attack_times[:]
        # Schedule all attacks
        for i, attack_time in enumerate(attack_times):
            self.attack_detected[i] = False
            self.engine.schedule_event(
                time=attack_time * 60,
                event_type="attack_start",
                data={"attack_id": i, "attack_time": attack_time * 60},
            )
        # Schedule metrics sampling
        for t in np.arange(
            0, duration_seconds, self.config["simulation"]["sampling_interval"]
        ):
            self.engine.schedule_event(
                time=t,
                event_type="sample_metrics",
                data={},
            )
        # Run simulation
        self.engine.run(duration=duration_seconds)

    def _handle_attack_start(self, event: Event):
        """Handle attack start event"""
        packets = self.attack_gen.create_attack_packets(
            attack_id=event.data["attack_id"],
            attack_time=event.data["attack_time"],
        )
        for i, packet in enumerate(packets):
            self.engine.schedule_event(
                time=event.data["attack_time"] + i * 0.001,
                event_type="packet_arrival",
                data={"packet": packet},
            )

    def _handle_packet_arrival(self, event: Event):
        """Handle packet arrival event"""
        packet = event.data["packet"]
        attack_id = packet.get("attack_id")
        # IDS inspection
        if packet["malicious"]:
            detected = self.ids.inspect_packet(packet)
            if detected:
                if attack_id is not None:
                    self.attack_detected[attack_id] = True
                packet["blocked"] = True
                return
        # Send to network
        accepted = self.network.receive_packet(packet, event.time)
        if accepted:
            self.network.forward_packet(packet, event.time)

    def _handle_sample_metrics(self, event: Event):
        """Handle metrics sampling event"""
        self.metrics.collect(event.time, self.network, self.ids, self.attack_gen)

    def get_phase3_summary(self) -> dict:
        """Return run-level outputs required by the Phase 3 data dictionary."""
        df = self.metrics.get_dataframe()
        if df.empty:
            return {
                "observed_detection_rate": 0.0,
                "observed_packet_detection_rate": 0.0,
                "total_attacks": 0,
                "total_detected": 0,
                "mean_inter_arrival_time": 0.0,
                "attack_counts_per_interval": [],
                "inter_arrival_times": [],
                "final_throughput": 0.0,
                "throughput_timeseries": [],
                "mean_throughput": 0.0,
                "initial_throughput": float(self.config["network"]["bandwidth"]),
            }

        total_attacks = len(self.attack_times_min)
        total_detected = int(sum(1 for v in self.attack_detected.values() if v))
        observed_detection_rate = (
            total_detected / total_attacks if total_attacks > 0 else 0.0
        )

        # Use raw sampled inter-arrivals from the generator to avoid
        # finite-window censoring bias when validating H3.
        inter_arrival_times = list(getattr(self.attack_gen, "last_inter_arrivals", []))
        mean_inter_arrival_time = (
            float(np.mean(inter_arrival_times)) if inter_arrival_times else 0.0
        )

        duration = int(self.config["simulation"].get("duration_minutes", 10))
        attack_counts_per_interval = [0] * max(duration, 1)
        for t in self.attack_times_min:
            idx = int(t)
            if 0 <= idx < len(attack_counts_per_interval):
                attack_counts_per_interval[idx] += 1

        throughput_timeseries = (
            df[["time", "throughput_mbps"]]
            .astype(float)
            .values
            .tolist()
        )

        return {
            "observed_detection_rate": float(observed_detection_rate),
            "observed_packet_detection_rate": float(self.ids.get_detection_rate()),
            "total_attacks": int(total_attacks),
            "total_detected": int(total_detected),
            "mean_inter_arrival_time": float(mean_inter_arrival_time),
            "attack_counts_per_interval": attack_counts_per_interval,
            "inter_arrival_times": inter_arrival_times,
            "final_throughput": float(df["throughput_mbps"].iloc[-1]),
            "throughput_timeseries": throughput_timeseries,
            "mean_throughput": float(df["throughput_mbps"].mean()),
            "initial_throughput": float(self.config["network"]["bandwidth"]),
        }

