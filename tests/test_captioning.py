import types
import unittest
from unittest import mock

import torch
from PIL import Image

from app.core.config import settings
from app.services import captioning


class _DummyProcessor:
    def __call__(self, images=None, return_tensors=None):
        return {"pixel_values": torch.zeros((1, 3, 2, 2), dtype=torch.float32)}

    def batch_decode(self, generated, skip_special_tokens=True):
        return ["dummy caption"]


class _DummyTokenizer:
    def batch_decode(self, generated, skip_special_tokens=True):
        return ["dummy caption"]


class _DummyModel:
    def __init__(self):
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        return torch.tensor([[0, 1, 2]])


class CaptioningTests(unittest.TestCase):
    def setUp(self):
        self._orig_enable = settings.enable_captioning
        self._orig_device = settings.caption_device
        self._orig_model_id = settings.caption_model_id
        self._orig_unload = settings.caption_unload_after_use
        captioning._captioner = None

    def tearDown(self):
        settings.enable_captioning = self._orig_enable
        settings.caption_device = self._orig_device
        settings.caption_model_id = self._orig_model_id
        settings.caption_unload_after_use = self._orig_unload
        captioning._captioner = None

    def test_captioner_vision_encoder_decoder_path(self):
        settings.caption_device = "cpu"
        settings.caption_model_id = "dummy/vision-encoder-decoder"

        with mock.patch("transformers.AutoConfig.from_pretrained") as mock_config, \
            mock.patch("transformers.VisionEncoderDecoderModel.from_pretrained", return_value=_DummyModel()), \
            mock.patch("transformers.ViTImageProcessor.from_pretrained", return_value=_DummyProcessor()), \
            mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=_DummyTokenizer()):
            mock_config.return_value = types.SimpleNamespace(model_type="vision-encoder-decoder")

            captioner = captioning.Captioner()
            image = Image.new("RGB", (4, 4), color=(128, 128, 128))
            text = captioner.caption(image)

        self.assertEqual(text, "dummy caption")

    def test_captioner_vision2seq_path(self):
        settings.caption_device = "cpu"
        settings.caption_model_id = "dummy/vision2seq"

        with mock.patch("transformers.AutoConfig.from_pretrained") as mock_config, \
            mock.patch("transformers.AutoModelForVision2Seq.from_pretrained", return_value=_DummyModel()), \
            mock.patch("transformers.AutoProcessor.from_pretrained", return_value=_DummyProcessor()):
            mock_config.return_value = types.SimpleNamespace(model_type="blip-2")

            captioner = captioning.Captioner()
            image = Image.new("RGB", (4, 4), color=(64, 64, 64))
            text = captioner.caption(image)

        self.assertEqual(text, "dummy caption")

    def test_caption_image_on_demand_unloads(self):
        settings.enable_captioning = True
        settings.caption_unload_after_use = True

        created = {}

        class _DummyCaptioner:
            def __init__(self):
                created["instance"] = self
                self.unloaded = False

            def caption(self, image):
                return "ok"

            def unload(self):
                self.unloaded = True

        with mock.patch.object(captioning, "Captioner", _DummyCaptioner):
            image = Image.new("RGB", (2, 2), color=(10, 10, 10))
            text = captioning.caption_image(image)

        self.assertEqual(text, "ok")
        self.assertTrue(created["instance"].unloaded)

    def test_caption_image_cached(self):
        settings.enable_captioning = True
        settings.caption_unload_after_use = False

        created = {"count": 0}

        class _DummyCaptioner:
            def __init__(self):
                created["count"] += 1

            def caption(self, image):
                return "ok"

        with mock.patch.object(captioning, "Captioner", _DummyCaptioner):
            image = Image.new("RGB", (2, 2), color=(10, 10, 10))
            self.assertEqual(captioning.caption_image(image), "ok")
            self.assertEqual(captioning.caption_image(image), "ok")

        self.assertEqual(created["count"], 1)

    def test_caption_image_disabled(self):
        settings.enable_captioning = False
        image = Image.new("RGB", (2, 2), color=(0, 0, 0))
        self.assertIsNone(captioning.caption_image(image))


if __name__ == "__main__":
    unittest.main()
