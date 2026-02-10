class Network:
    """Simulates network with capacity and buffering"""

    def __init__(self, bandwidth_mbps: float, buffer_size: int, latency_ms: float = 10.0):
        self.bandwidth = bandwidth_mbps
        self.buffer_size = buffer_size
        self.base_latency = latency_ms
        # State variables
        self.buffer_current = 0
        self.total_packets_received = 0
        self.total_packets_dropped = 0
        self.total_packets_forwarded = 0
        # For throughput calculation
        self.bytes_forwarded = 0
        self.last_sample_time = 0

    def receive_packet(self, packet: dict, current_time: float) -> bool:
        """
        Attempt to receive a packet into network buffer
        Returns:
            True if packet accepted, False if dropped
        """
        self.total_packets_received += 1
        if self.buffer_current >= self.buffer_size:
            self.total_packets_dropped += 1
            packet["dropped"] = True
            return False
        self.buffer_current += 1
        packet["dropped"] = False
        return True

    def forward_packet(self, packet: dict, current_time: float) -> None:
        """Process and forward a packet"""
        if self.buffer_current > 0:
            self.buffer_current -= 1
            self.total_packets_forwarded += 1
            self.bytes_forwarded += packet["size_kb"] * 1024

    def get_current_throughput(self, current_time: float) -> float:
        """Calculate current throughput in Mbps"""
        time_diff = current_time - self.last_sample_time
        if time_diff == 0:
            return 0.0
        throughput = (self.bytes_forwarded * 8) / (time_diff * 1e6)
        self.bytes_forwarded = 0
        self.last_sample_time = current_time
        return throughput

    def get_packet_loss_rate(self) -> float:
        """Calculate packet drop rate"""
        if self.total_packets_received == 0:
            return 0.0
        return self.total_packets_dropped / self.total_packets_received

    def get_stats(self) -> dict:
        """Return current network statistics"""
        return {
            "packets_received": self.total_packets_received,
            "packets_dropped": self.total_packets_dropped,
            "drop_rate": self.get_packet_loss_rate(),
            "buffer_utilization": self.buffer_current / self.buffer_size,
        }

