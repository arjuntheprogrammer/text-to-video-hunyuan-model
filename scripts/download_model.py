import argparse
import os

import torch


def download_model(model_id: str, cache_dir: str, use_cuda: bool) -> None:
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    transformer_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    pipe_dtype = torch.float16 if device == "cuda" else torch.float32
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

    try:
        from diffusers import HunyuanVideoImageToVideoPipeline, HunyuanVideoTransformer3DModel

        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=transformer_dtype,
            cache_dir=cache_dir,
            token=token,
        )
        pipe = HunyuanVideoImageToVideoPipeline.from_pretrained(
            model_id,
            transformer=transformer,
            torch_dtype=pipe_dtype,
            cache_dir=cache_dir,
            token=token,
        )
    except ImportError:
        from diffusers import DiffusionPipeline

        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=pipe_dtype,
            cache_dir=cache_dir,
            trust_remote_code=True,
            token=token,
        )

    pipe.to(device)
    print(f"Model downloaded to {cache_dir} and initialized on {device}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-download HunyuanVideo-I2V model.")
    parser.add_argument(
        "--model-id",
        default="hunyuanvideo-community/HunyuanVideo-I2V",
        help="Model ID from Hugging Face Hub.",
    )
    parser.add_argument("--cache-dir", default="./models", help="Model cache directory.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU initialization.")
    args = parser.parse_args()

    download_model(model_id=args.model_id, cache_dir=args.cache_dir, use_cuda=not args.cpu)


if __name__ == "__main__":
    main()
