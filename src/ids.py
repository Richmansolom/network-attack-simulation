import random


class IntrusionDetectionSystem:
    """IDS with probabilistic detection"""

    def __init__(self, detection_probability: float = 0.85):
        self.detection_prob = detection_probability
        # Statistics
        self.total_inspected = 0
        self.malicious_inspected = 0
        self.total_detected = 0
        self.false_negatives = 0

    def inspect_packet(self, packet: dict) -> bool:
        """
        Inspect a packet and determine if detected as malicious
        Uses Bernoulli trial with probability p
        Returns:
            True if detected as malicious, False otherwise
        """
        self.total_inspected += 1
        if packet["malicious"]:
            self.malicious_inspected += 1
            # Bernoulli trial: detect with probability p
            detected = random.random() < self.detection_prob
            if detected:
                self.total_detected += 1
                packet["detected"] = True
                return True
            else:
                self.false_negatives += 1
                packet["detected"] = False
                return False
        else:
            packet["detected"] = False
            return False

    def get_detection_rate(self) -> float:
        """Calculate actual detection rate"""
        if self.malicious_inspected == 0:
            return 0.0
        return self.total_detected / self.malicious_inspected

    def get_stats(self) -> dict:
        """Return IDS statistics"""
        return {
            "total_inspected": self.total_inspected,
            "malicious_inspected": self.malicious_inspected,
            "total_detected": self.total_detected,
            "detection_rate": self.get_detection_rate(),
        }

