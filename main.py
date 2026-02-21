from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import requests
import base64
from itertools import cycle
import itertools
import os
import random
from keep_alive import start_keep_alive

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Request model
# -------------------------

class GenerateRequest(BaseModel):
    prompt: Optional[str] = None
    image_base64: Optional[str] = None


# -------------------------
# GPU config
# -------------------------

GPU_ENDPOINTS = [
    "https://w3gfk2krqc76x4-8188.proxy.runpod.net/",
    "https://iygoioc7o5hytl-8188.proxy.runpod.net/"
]

GPU_TIMEOUT = 20
GPU_API_KEY = os.getenv("GPU_API_KEY")

gpu_cycle = itertools.cycle(GPU_ENDPOINTS)


def generate_via_gpu(payload):

    for _ in range(len(GPU_ENDPOINTS)):
        endpoint = next(gpu_cycle)
        print(f"🔄 Trying GPU: {endpoint}")

        try:
            response = requests.post(
                endpoint,
                json=payload,
                timeout=GPU_TIMEOUT
            )

            response.raise_for_status()
            data = response.json()

            image = data.get("image")

            if image:
                print(f"🟢 GPU SUCCESS from {endpoint}")
                return image

        except Exception as e:
            print(f"❌ GPU failed {endpoint}: {e}")

    print("🚨 All GPUs failed")
    return None


# -------------------------
# Fallback images
# -------------------------

FALLBACK_IMAGES = [
    f"https://www-static.wemmstudios.se/wp-content/uploads/2026/02/hero_{i:02d}.png"
    for i in range(1, 15)
]

fallback_cycle = cycle(FALLBACK_IMAGES)


# -------------------------
# Startup
# -------------------------

@app.on_event("startup")
def startup_event():
    start_keep_alive()


# -------------------------
# Generate endpoint
# -------------------------

@app.post("/generate")
def generate(request: GenerateRequest):

    payload = {
        "prompt": request.prompt,
        "image": request.image_base64
    }

    # 🔥 Try GPU first
    gpu_image = generate_via_gpu(payload)

    if gpu_image:
        return {
            "status": "READY",
            "source": "gpu",
            "image": gpu_image
        }

    # 🔁 Fallback
    img_url = next(fallback_cycle)

    return {
        "status": "READY",
        "source": "fallback",
        "image": img_url
    }


# -------------------------
# Health
# -------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}
