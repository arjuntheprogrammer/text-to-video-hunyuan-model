import unittest

from app.prompt_builder import build_structured_prompt


class PromptBuilderTests(unittest.TestCase):
    def test_build_structured_prompt_includes_fields(self) -> None:
        prompt = "A cool shot."
        result = build_structured_prompt(
            user_prompt=prompt,
            caption="a cat on a chair",
            subject="animal",
            action="walking",
            camera_motion="slow pan",
            shot_type="wide",
            lighting="soft daylight",
            mood="calm",
            default_suffix="keep it real",
        )
        self.assertIn("Image description: a cat on a chair", result)
        self.assertIn("Subject: animal", result)
        self.assertIn("Action: walking", result)
        self.assertIn("Camera: slow pan", result)
        self.assertIn("Shot: wide", result)
        self.assertIn("Lighting: soft daylight", result)
        self.assertIn("Mood: calm", result)
        self.assertIn(prompt, result)
        self.assertTrue(result.endswith("keep it real"))


if __name__ == "__main__":
    unittest.main()
