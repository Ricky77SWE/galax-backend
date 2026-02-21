from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import requests
import itertools
import os
import random
import time
import base64

# =====================================================
# APP SETUP
# =====================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# REQUEST MODEL (MATCHAR draw.js)
# =====================================================
print("POSTING TO:", f"{base}/prompt")

class GenerateRequest(BaseModel):
    styleKey: str
    seed: Optional[int] = None
    mode: Optional[str] = None
    clientId: Optional[str] = None

# =====================================================
# GPU CONFIG
# =====================================================

GPU_ENDPOINTS = [
    "https://w3gfk2krqc76x4-8188.proxy.runpod.net",
    "https://iygoioc7o5hytl-8188.proxy.runpod.net"
]

GPU_TIMEOUT = 30
POLL_INTERVAL = 1
MAX_POLL_SECONDS = 90

gpu_cycle = itertools.cycle(GPU_ENDPOINTS)

# =====================================================
# WORKFLOW BUILDER
# =====================================================

def build_workflow(style_key: str, seed: Optional[int]):

    seed = seed or random.randint(1, 999999999)

    positive_prompt = f"""
    Cute, friendly, high quality 3D CGI superhero creature.
    Clean silhouette. Bright cinematic lighting.
    Style inspired by {style_key}.
    Child friendly. Disney/Pixar quality.
    """

    negative_prompt = """
    blurry, low quality, distorted, creepy, horror,
    extra limbs, floating head, watermark, text
    """

    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"}
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": positive_prompt, "clip": ["1", 1]}
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["1", 1]}
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 896, "height": 896, "batch_size": 1}
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "seed": seed,
                "steps": 28,
                "cfg": 4.5,
                "sampler_name": "dpmpp_2m_sde",
                "scheduler": "karras",
                "denoise": 1.0,
                "latent_image": ["4", 0]
            }
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]}
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {"images": ["6", 0], "filename_prefix": "galax"}
        }
    }

# =====================================================
# GPU GENERATION
# =====================================================

def wait_for_gpu(base_url, max_wait=60):
    print("⏳ Waiting for GPU to wake...")
    start = time.time()

    while time.time() - start < max_wait:
        try:
            r = requests.get(f"{base_url}/system_stats", timeout=5)
            if r.status_code == 200:
                print("🟢 GPU awake")
                return True
        except:
            pass

        time.sleep(3)

    print("🔴 GPU failed to wake")
    return False
    
def generate_via_gpu(style_key: str, seed: Optional[int]):

    workflow = build_workflow(style_key, seed)

    for _ in range(len(GPU_ENDPOINTS)):
        endpoint = next(gpu_cycle)
        base = endpoint.rstrip("/")

        print(f"🔄 Trying GPU: {base}")

        try:
            # 🟡 AUTO WAKE
            if not wait_for_gpu(base):
                continue

            # 1️⃣ SEND PROMPT
            r = requests.post(
                f"{base}/prompt",
                json={
                    "prompt": workflow,
                    "client_id": "galax-backend"
                },
                timeout=GPU_TIMEOUT
            )

            r.raise_for_status()
            prompt_id = r.json().get("prompt_id")

            if not prompt_id:
                continue

            # 2️⃣ POLL HISTORY
            start_time = time.time()

            while time.time() - start_time < MAX_POLL_SECONDS:
                time.sleep(POLL_INTERVAL)

                h = requests.get(
                    f"{base}/history/{prompt_id}",
                    timeout=10
                )

                if h.status_code != 200:
                    continue

                data = h.json()
                outputs = data.get(prompt_id, {}).get("outputs", {})

                for node in outputs.values():
                    if "images" in node:
                        image_meta = node["images"][0]

                        view_url = (
                            f"{base}/view?"
                            f"filename={image_meta['filename']}&"
                            f"subfolder={image_meta.get('subfolder','')}&"
                            f"type={image_meta.get('type','output')}"
                        )

                        img_resp = requests.get(view_url, timeout=20)
                        img_resp.raise_for_status()

                        image_base64 = base64.b64encode(img_resp.content).decode()

                        print("🟢 GPU SUCCESS")
                        return image_base64

        except Exception as e:
            print(f"❌ GPU failed {base}: {e}")

    print("🚨 All GPUs failed")
    return None
    
# =====================================================
# FALLBACK IMAGES
# =====================================================

FALLBACK_IMAGES = [
    f"https://www-static.wemmstudios.se/wp-content/uploads/2026/02/hero_{i:02d}.png"
    for i in range(1, 15)
]

fallback_cycle = itertools.cycle(FALLBACK_IMAGES)

# =====================================================
# API ENDPOINT
# =====================================================

@app.post("/generate")
def generate(request: GenerateRequest):

    print("📩 Generate request:", request.styleKey)

    gpu_image = generate_via_gpu(
        style_key=request.styleKey,
        seed=request.seed
    )

    if gpu_image:
        return {
            "status": "READY",
            "source": "gpu",
            "image": f"data:image/png;base64,{gpu_image}"
        }

    return {
        "status": "READY",
        "source": "fallback",
        "image": next(fallback_cycle)
    }

@app.get("/health")
def health():
    return {"status": "ok"}
