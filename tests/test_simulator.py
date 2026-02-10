import unittest

from src.simulator import SimulationEngine


class TestSimulationEngine(unittest.TestCase):
    def test_event_scheduling(self):
        """Test that events are processed in time order"""
        sim = SimulationEngine()
        results = []

        def handler(event):
            results.append(event.time)

        sim.register_handler("test", handler)

        # Schedule events out of order
        sim.schedule_event(5.0, "test")
        sim.schedule_event(2.0, "test")
        sim.schedule_event(8.0, "test")
        sim.schedule_event(1.0, "test")

        sim.run(10.0)

        self.assertEqual(results, [1.0, 2.0, 5.0, 8.0])


if __name__ == "__main__":
    unittest.main()

