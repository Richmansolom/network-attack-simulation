import unittest

from src.attack_generator import AttackGenerator


class TestAttackGenerator(unittest.TestCase):
    def test_generate_attack_times_increasing(self):
        """Generated attack times should be strictly increasing."""
        gen = AttackGenerator(attack_rate=1.0, packets_per_attack=10)
        times = gen.generate_attack_times(0.0, 10.0)
        self.assertTrue(all(t2 > t1 for t1, t2 in zip(times, times[1:])))

    def test_create_attack_packets_count(self):
        """create_attack_packets should create the configured number of packets."""
        gen = AttackGenerator(attack_rate=1.0, packets_per_attack=5)
        packets = gen.create_attack_packets(attack_id=1, attack_time=0.0)
        self.assertEqual(len(packets), 5)
        self.assertTrue(all(p["malicious"] for p in packets))


if __name__ == "__main__":
    unittest.main()

