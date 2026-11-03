"""
Unit test suite verifying vision engine initialization, preset selection, and mode transitions.
"""

import unittest
from vision_engine import SahraEyeEngine, TifawtReIDMatcher


class TestVisionEngineCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Instantiate once to avoid reloading YOLO model weights repeatedly
        cls.engine = SahraEyeEngine()

    def test_engine_initialization(self):
        """Verify default engine attributes and initial mode."""
        self.assertEqual(self.engine.current_mode, 1)
        self.assertIsNotNone(self.engine.reid_matcher)
        self.assertTrue(self.engine.reid_matcher.child_prior)

    def test_reid_matcher_creation(self):
        """Ensure Re-ID matcher instantiates successfully."""
        matcher = TifawtReIDMatcher()
        self.assertIsNotNone(matcher)
        self.assertFalse(matcher.has_target())
        self.assertTrue(matcher.child_prior)

    def test_set_target_preset(self):
        """Verify preset target configuration through set_target_preset."""
        ok, label = self.engine.set_target_preset("preset_1")
        self.assertTrue(ok)
        self.assertIn("Red Cap", label)
        self.assertTrue(self.engine.reid_matcher.has_target())

    def test_mode_switching(self):
        """Test transitioning between Fall Detection (1) and Search (2) modes."""
        self.engine.set_mode(2)
        self.assertEqual(self.engine.current_mode, 2)
        self.engine.set_mode(1)
        self.assertEqual(self.engine.current_mode, 1)

    def test_health_check_endpoint(self):
        """Verify the health check endpoint returns proper schema."""
        from fastapi.testclient import TestClient
        import app

        # Assign test engine instance to app module
        app.engine = self.engine
        client = TestClient(app.app)
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertTrue(data.get("engine_active"))
        self.assertEqual(data.get("service"), "Sahra-Eye Vision Engine")
        self.assertIn("timestamp", data)


if __name__ == "__main__":
    unittest.main()
