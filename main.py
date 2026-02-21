from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import base64
from itertools import cycle
from keep_alive import start_keep_alive
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔥 GPU LISTA
GPU_ENDPOINTS = [
    "https://w3gfk2krqc76x4-8188.proxy.runpod.net/",
    "https://iygoioc7o5hytl-8188.proxy.runpod.net/"
]

GPU_TIMEOUT = 20
GPU_API_KEY = os.getenv("GPU_API_KEY")
GPU_TIMEOUT = 20  # sekunder

# 🔁 WordPress fallback-bilder
FALLBACK_IMAGES = [
    f"https://www-static.wemmstudios.se/wp-content/uploads/2026/02/hero_{i:02d}.png"
    for i in range(1, 15)
]

fallback_cycle = cycle(FALLBACK_IMAGES)

@app.on_event("startup")
def startup_event():
    start_keep_alive()

import itertools

# Round-robin index
gpu_cycle = itertools.cycle(GPU_ENDPOINTS)

def generate_via_gpu(payload):

    # Vi testar båda GPU:erna max en gång per request
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
        
@app.post("/generate")
def generate(request: GenerateRequest):

    payload = {
        "prompt": request.prompt,
        "image": request.image_base64
    }

    # 🔥 Försök GPU först
    gpu_image = generate_via_gpu(payload)

    if gpu_image:
        print("🟢 GPU SUCCESS")
        return {
            "status": "READY",
            "source": "gpu",
            "image": gpu_image
        }

    # 🔁 Fallback om GPU failar
    print("🟡 GPU FAILED – using fallback")

    return {
        "status": "READY",
        "source": "fallback",
        "image": random.choice(FALLBACK_IMAGES)
    }

async def generate(payload: dict):
    """
    CPU-backend:
    - Tar emot data från appen
    - Returnerar ALLTID en bild (fallback nu, GPU senare)
    """
    try:
        img_url = next(fallback_cycle)
        r = requests.get(img_url, timeout=10)
        r.raise_for_status()

        b64 = base64.b64encode(r.content).decode("utf-8")
        data_url = f"data:image/png;base64,{b64}"

        return {
            "status": "READY",
            "source": "fallback",
            "image": data_url
        }

    except Exception as e:
        return {
            "status": "READY",
            "source": "fallback-error",
            "image": None,
            "error": str(e)
        }

@app.get("/health")
async def health():
    return {"status": "ok"}
