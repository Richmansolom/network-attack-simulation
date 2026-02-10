import heapq
from typing import List, Any

import numpy as np

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
        # Register event handlers
        self.engine.register_handler("attack_start", self._handle_attack_start)
        self.engine.register_handler("packet_arrival", self._handle_packet_arrival)
        self.engine.register_handler("sample_metrics", self._handle_sample_metrics)

    def run(self, duration_minutes: float, seed: int = None):
        """Run complete simulation"""
        if seed is not None:
            np.random.seed(seed)
        duration_seconds = duration_minutes * 60
        # Generate attack schedule
        attack_times = self.attack_gen.generate_attack_times(0.0, duration_minutes)
        # Schedule all attacks
        for i, attack_time in enumerate(attack_times):
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
        # IDS inspection
        if packet["malicious"]:
            detected = self.ids.inspect_packet(packet)
            if detected:
                packet["blocked"] = True
                return
        # Send to network
        accepted = self.network.receive_packet(packet, event.time)
        if accepted:
            self.network.forward_packet(packet, event.time)

    def _handle_sample_metrics(self, event: Event):
        """Handle metrics sampling event"""
        self.metrics.collect(event.time, self.network, self.ids, self.attack_gen)

