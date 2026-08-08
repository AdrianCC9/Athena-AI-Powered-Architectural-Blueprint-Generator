import tempfile
import unittest
from pathlib import Path

from athena.demo import build_demo_page, build_floorplan_svg, export_demo_html, parse_floorplan_prompt


class DemoTests(unittest.TestCase):
    def test_prompt_parser_counts_rooms_and_features(self):
        spec = parse_floorplan_prompt("three bedroom house with two bathrooms, office, garage, and balcony")

        self.assertEqual(spec["bedrooms"], 3)
        self.assertEqual(spec["bathrooms"], 2)
        self.assertTrue(spec["has_office"])
        self.assertTrue(spec["has_garage"])
        self.assertTrue(spec["has_balcony"])

    def test_svg_contains_expected_rooms(self):
        svg = build_floorplan_svg("one bedroom apartment with dining and laundry")

        self.assertIn("<svg", svg)
        self.assertIn("Bedroom 1", svg)
        self.assertIn("Bathroom 1", svg)
        self.assertIn("Dining", svg)
        self.assertIn("Laundry", svg)

    def test_export_writes_standalone_html(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "demo.html"
            exported = export_demo_html("two bedroom apartment", output)

            self.assertEqual(exported, output)
            html = output.read_text(encoding="utf-8")
            self.assertIn("Athena Floorplan Demo", html)
            self.assertIn("<svg", html)

    def test_html_escapes_prompt_text(self):
        html = build_demo_page("<script>alert('x')</script>")

        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("<script>alert", html)


if __name__ == "__main__":
    unittest.main()
