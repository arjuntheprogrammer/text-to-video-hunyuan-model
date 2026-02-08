# HunyuanVideo Production Starter Project — Agent Prompt (Docker + Public Gradio)

You are a **Senior Python AI engineer**. Create a **COMPLETE end-to-end Python project** in the **CURRENT DIRECTORY** that generates **10–20 second videos** using the **open-source HunyuanVideo Image-to-Video model**.

**MANDATORY MODEL:**
👉 https://huggingface.co/hunyuanvideo-community/HunyuanVideo-I2V

You MUST base the pipeline on this open-source model (via HuggingFace diffusers or the official implementation). Do not substitute with closed or unrelated models.

---

## OUTPUT RULES (CRITICAL)

- Produce the **full project files and code** when executed by an agent.
- **NO explanations, commentary, or analysis** in the final output.
- Output **files and code only**.
- The project must be runnable with a **single command**:

```bash
docker compose up --build
````

* Target environment:

  * **Ubuntu 22.04 host**
  * **Single NVIDIA L40s GPU**
* The **Gradio UI must be publicly exposed** (listen on `0.0.0.0`).

---

##

## CORE FUNCTIONAL REQUIREMENTS

##

The system must:

1. Accept a **starting image** (Image → Video).
2. Accept a **text prompt** describing motion, animation, or scene changes.
3. Generate a **10–20 second video** using **HunyuanVideo-I2V**.
4. Expose both:

   * **REST API (FastAPI)**
   * **Public Gradio UI**
5. Run efficiently on **NVIDIA L40s** with fp16 inference.
6. Be runnable locally with **minimal setup**.
7. Include **Dockerized production deployment**.

---

##

## USER WORKFLOW

##

User provides:

* starting image
* text prompt
* generation parameters

System outputs:

* generated `.mp4` video
* downloadable via UI and API

---

##

## PYTHON VERSION

##

* **Python 3.10 (default, REQUIRED)**
  Chosen for maximum compatibility with:

  * PyTorch + CUDA wheels
  * diffusers / transformers / accelerate
  * xformers (optional)
  * OpenCV / ffmpeg stack

The project should remain **Python 3.11 compatible**, but Docker and env defaults must use **3.10**.

---

##

## DEPLOYMENT (DOCKER-COMPOSE)

##

### REQUIRED FILES

* `Dockerfile`
* `docker-compose.yml`

### Dockerfile

* Base image:

  * `nvidia/cuda:*runtime-ubuntu22.04` OR equivalent Ubuntu 22.04 CUDA image
* Python 3.10
* System deps:

  * ffmpeg
  * libgl1
  * libglib2.0-0
* Install Python deps via `requirements.txt`
* Entrypoint runs `python run.py`

### docker-compose.yml

* NVIDIA GPU enabled:

  * `deploy.resources.reservations.devices.capabilities: [gpu]`
* Ports:

  * `8000` → FastAPI
  * `7860` → Gradio (public)
* Volumes:

  * `./models:/app/models`
  * `./outputs:/app/outputs`
* Environment:

  * `HF_HOME=/app/models`
  * `HF_HUB_CACHE=/app/models`
  * `TRANSFORMERS_CACHE=/app/models`
  * `TORCH_HOME=/app/models`
* Gradio must bind to `0.0.0.0`

---

##

## PROJECT STRUCTURE (MANDATORY)

##

```
.
├── app/
│   ├── main.py              # FastAPI entrypoint
│   ├── gradio_ui.py         # Public Gradio UI
│   ├── config.py
│   ├── schemas.py
│   ├── pipeline.py          # HunyuanVideo-I2V loading + inference
│   ├── video_utils.py       # mp4 encoding
│   ├── utils.py
│   └── __init__.py
│
├── outputs/                 # generated videos
├── models/                  # HF cache + model weights
│
├── scripts/
│   └── download_model.py    # optional pre-download
│
├── requirements.txt
├── environment.yml
├── .gitignore
├── README.md
├── Dockerfile
├── docker-compose.yml
└── run.py                   # start API + UI
```

Directories must be auto-created if missing.

---

##

## MODEL IMPLEMENTATION

##

### Model

* **HunyuanVideo-I2V**
* Source:
  [https://huggingface.co/hunyuanvideo-community/HunyuanVideo-I2V](https://huggingface.co/hunyuanvideo-community/HunyuanVideo-I2V)

### `app/pipeline.py` MUST:

* Load the model **once globally**
* Use **torch.float16** on GPU
* Automatically detect CUDA
* Fallback to CPU with a **clear warning**
* Enable:

  * `torch.backends.cuda.matmul.allow_tf32 = True`
  * `torch.backends.cudnn.allow_tf32 = True`
  * attention slicing / memory-efficient attention
  * optional xformers if available
* Use:

  * `torch.inference_mode()` or `torch.no_grad()`

### Supported parameters:

* `num_frames`
* `guidance_scale`
* `num_inference_steps`
* `fps`
* `seed`

### Defaults (safe for L40s):

* fps: `16`
* duration: `10s`
* frames: `160`
* max frames allowed: `320` (~20s)

### Inputs:

* PIL Image
* prompt string

### Outputs:

* video frames
* encoded `.mp4`

---

##

## VIDEO ENCODING

##

Use:

* `imageio` + `imageio-ffmpeg` OR OpenCV

### Requirements:

* fps control
* mp4 output
* RGB/BGR correctness
* timestamped filenames
* save to `outputs/`

---

##

## FASTAPI API

##

### Endpoints

#### `GET /health`

Returns:

* status
* device (cpu/cuda)
* model_loaded

#### `POST /generate`

Multipart form:

* image
* prompt
* num_frames
* guidance_scale
* steps
* fps
* seed (optional)

Returns:

* mp4 stream OR
* JSON with video path

Also expose:

* `GET /outputs/{filename}`

---

##

## PUBLIC GRADIO UI

##

Inputs:

* image upload
* prompt textbox
* frame slider
* steps slider
* guidance scale slider
* fps slider
* seed input

Output:

* video player

Rules:

* Calls pipeline **directly**
* Runs on `0.0.0.0`
* Exposed publicly
* GPU-safe queueing / concurrency limit

---

##

## L40s OPTIMIZATION

##

* fp16 inference
* TF32 enabled
* memory-efficient attention
* conservative defaults
* configurable caps in `config.py`

---

##

## DEPENDENCIES

##

### requirements.txt

Must include:

* torch
* torchvision
* diffusers
* transformers
* accelerate
* fastapi
* uvicorn
* gradio
* pillow
* opencv-python
* imageio
* imageio-ffmpeg
* numpy
* safetensors
* xformers (optional)

---

##

## README.md MUST INCLUDE

##

* Overview
* GPU + NVIDIA Container Toolkit setup
* `docker compose up --build`
* API usage example (curl)
* UI URL
* Output persistence
* Performance notes for L40s

---

## DOCS

Create docs folder in parent directory and have swagger API docs in there

---

##

## RUN SCRIPT

##

`run.py` must:

* Start FastAPI
* Start Gradio
* Print URLs
* Handle SIGTERM cleanly
* Work inside Docker

---

##

## CODE QUALITY

##

* Modular
* Documented
* Validated inputs
* Robust error handling
* Auto-create dirs
* No TODOs
* Production-grade defaults

---

##

## FINAL RULE

##

When executed by an agent:

* Output **only files and code**
* No explanations
* Fully runnable
* Uses **HunyuanVideo-I2V open-source model**
* Optimized for **NVIDIA L40s**