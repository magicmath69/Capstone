"""Image regression checks. Run: python Camera/test_detect_holes.py."""

import json
from pathlib import Path
import unittest

import cv2
import numpy as np

from detect_holes import detect


HERE = Path(__file__).resolve().parent


def nearest_center(result):
    return next((np.array(hole["center"]) for hole in result["holes"]
                 if hole["id"] == result["nearest_hole_id"]), None)


class DetectionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.labels = json.loads((HERE / "validation_labels.json").read_text())
        cls.images = {}
        cls.results = {}
        for name in cls.labels["nearest_centers"]:
            image = cv2.imread(str(HERE / "photos" / name))
            if image is None:
                raise AssertionError(f"Missing validation photo: {name}")
            cls.images[name] = image
            original = image.copy()
            cls.results[name] = detect(image)
            if not np.array_equal(image, original):
                raise AssertionError(f"Detection changed source pixels: {name}")

    def assert_target_near(self, result, expected):
        actual = nearest_center(result)
        self.assertIsNotNone(actual, "No target detected")
        self.assertLess(float(np.linalg.norm(actual - expected)), 20,
                        f"Target {actual} does not match manually marked opening {expected}")

    def test_nearest_opening_in_all_19_photos(self):
        for name, expected in self.labels["nearest_centers"].items():
            with self.subTest(photo=name):
                result = self.results[name]
                self.assert_target_near(result, np.array(expected))
                self.assertIsNotNone(result["frame_outline"])

    def test_known_shadows_are_not_openings(self):
        for name, points in self.labels["non_hole_centers"].items():
            for point in points:
                with self.subTest(photo=name, shadow=point):
                    for hole in self.results[name]["holes"]:
                        self.assertGreater(float(np.linalg.norm(np.array(hole["center"]) - point)), 20)

    def test_rotation(self):
        for name, image in self.images.items():
            with self.subTest(photo=name):
                expected = nearest_center(self.results[name])
                self.assertIsNotNone(expected)
                expected = np.array([image.shape[0] - 1 - expected[1], expected[0]])
                result = detect(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
                self.assert_target_near(result, expected)

    def test_brightness_changes(self):
        for name, image in self.images.items():
            for gain in (0.65, 1.3):
                with self.subTest(photo=name, brightness=gain):
                    altered = np.clip(image.astype(float) * gain, 0, 255).astype(np.uint8)
                    self.assert_target_near(detect(altered), np.array(self.labels["nearest_centers"][name]))

    def test_backgrounds_and_blank_images(self):
        for name, image in self.images.items():
            # These crops are above the racquet and contain no visible openings.
            height = 150 if "_15-" in name else 250
            with self.subTest(background=name):
                self.assertIsNone(nearest_center(detect(image[:height])))
        for level in (0, 127, 255):
            with self.subTest(blank=level):
                self.assertIsNone(nearest_center(detect(np.full((800, 1280, 3), level, np.uint8))))


if __name__ == "__main__":
    unittest.main(verbosity=2)
