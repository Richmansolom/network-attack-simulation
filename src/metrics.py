import pandas as pd


class MetricsCollector:
    """Collects and exports simulation metrics"""

    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self.metrics_history = []

    def collect(self, current_time: float, network, ids, attack_gen) -> None:
        """Sample current state of all components"""
        net_stats = network.get_stats()
        ids_stats = ids.get_stats()
        metrics = {
            "time": current_time,
            "throughput_mbps": network.get_current_throughput(current_time),
            "buffer_utilization": net_stats["buffer_utilization"],
            "packets_dropped": net_stats["packets_dropped"],
            "drop_rate": net_stats["drop_rate"],
            "total_detections": ids_stats["total_detected"],
            "detection_rate": ids_stats["detection_rate"],
            "attacks_generated": attack_gen.attacks_generated,
        }
        self.metrics_history.append(metrics)

    def export_to_csv(self, filename: str) -> None:
        """Export metrics to CSV file"""
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(filename, index=False)

    def get_dataframe(self) -> pd.DataFrame:
        """Return metrics as pandas DataFrame"""
        return pd.DataFrame(self.metrics_history)

    def get_summary_stats(self) -> dict:
        """Return summary statistics from the last sample (for Phase 3 experiments)."""
        if not self.metrics_history:
            return {}
        last = self.metrics_history[-1]
        return {
            "final_detection_rate": last["detection_rate"],
            "final_drop_rate": last["drop_rate"],
            "total_detections": last["total_detections"],
            "attacks_generated": last["attacks_generated"],
            "final_buffer_utilization": last["buffer_utilization"],
        }

