import unittest

from src.network import Network


class TestNetwork(unittest.TestCase):
    def test_receive_and_forward_packet(self):
        """Network should accept and then forward a packet when buffer is available."""
        net = Network(bandwidth_mbps=100, buffer_size=1)
        packet = {"size_kb": 1.0, "dropped": None}

        accepted = net.receive_packet(packet, current_time=0.0)
        self.assertTrue(accepted)
        self.assertFalse(packet["dropped"])

        net.forward_packet(packet, current_time=1.0)
        self.assertEqual(net.total_packets_forwarded, 1)

    def test_drop_when_buffer_full(self):
        """Network should drop packets when buffer is full."""
        net = Network(bandwidth_mbps=100, buffer_size=1)
        p1 = {"size_kb": 1.0, "dropped": None}
        p2 = {"size_kb": 1.0, "dropped": None}

        self.assertTrue(net.receive_packet(p1, current_time=0.0))
        self.assertFalse(net.receive_packet(p2, current_time=0.0))
        self.assertTrue(p2["dropped"])
        self.assertGreater(net.get_packet_loss_rate(), 0.0)


if __name__ == "__main__":
    unittest.main()

