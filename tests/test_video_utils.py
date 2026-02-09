import unittest

import numpy as np

from app.media.video import compute_target_size, deflicker_frames


class VideoUtilsTests(unittest.TestCase):
    def test_compute_target_size_even(self) -> None:
        width, height = compute_target_size(1080, 1920, 720)
        self.assertEqual(height, 720)
        self.assertEqual(width % 2, 0)
        self.assertEqual(height % 2, 0)
        self.assertGreater(width, 0)

    def test_deflicker_frames(self) -> None:
        frames = [
            np.full((2, 2, 3), 0, dtype=np.uint8),
            np.full((2, 2, 3), 100, dtype=np.uint8),
            np.full((2, 2, 3), 200, dtype=np.uint8),
        ]
        output = deflicker_frames(frames, window=3)
        self.assertEqual(len(output), len(frames))
        self.assertTrue(np.all(output[1] == 100))


if __name__ == "__main__":
    unittest.main()
