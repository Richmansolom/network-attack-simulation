import unittest

from src.ids import IntrusionDetectionSystem


class TestIntrusionDetectionSystem(unittest.TestCase):
    def test_inspect_packet_sets_detected_flag(self):
        """Inspecting a malicious packet should set the detected field (True/False)."""
        ids = IntrusionDetectionSystem(detection_probability=1.0)
        packet = {"id": 1, "malicious": True, "size_kb": 1.0}

        detected = ids.inspect_packet(packet)
        self.assertTrue(detected)
        self.assertTrue(packet["detected"])
        self.assertEqual(ids.total_inspected, 1)
        self.assertEqual(ids.malicious_inspected, 1)
        self.assertEqual(ids.total_detected, 1)

    def test_non_malicious_packet_not_detected(self):
        """Non-malicious packets should never be flagged as detected."""
        ids = IntrusionDetectionSystem(detection_probability=1.0)
        packet = {"id": 1, "malicious": False, "size_kb": 1.0}

        detected = ids.inspect_packet(packet)
        self.assertFalse(detected)
        self.assertFalse(packet["detected"])
        self.assertEqual(ids.total_inspected, 1)
        self.assertEqual(ids.malicious_inspected, 0)
        self.assertEqual(ids.total_detected, 0)


if __name__ == "__main__":
    unittest.main()

