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

app = FastAPI()

# =====================================================
# KEEP CPU ALIVE
# =====================================================

from keep_alive import start_keep_alive
@app.on_event("startup")
def startup_event():
    start_keep_alive()

# =====================================================
# APP SETUP
# =====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# REQUEST MODEL (MATCHAR draw.js)
# =====================================================

class GenerateRequest(BaseModel):
    styleKey: str
    seed: Optional[int] = None
    image_base64: str
    mode: Optional[str] = None
    clientId: Optional[str] = None

# =====================================================
# GPU CONFIG
# =====================================================

GPU_ENDPOINTS = [
    "https://w3gfk2krqc76x4-8188.proxy.runpod.net",
    "https://iygoioc7o5hytl-8188.proxy.runpod.net",
    "https://n8ghjjv9kdbhez-8188.proxy.runpod.net"
]

GPU_TIMEOUT = 30
POLL_INTERVAL = 1
MAX_POLL_SECONDS = 90

# =====================================================
# GPU STATE CACHE
# =====================================================

ACTIVE_GPU = None
DEAD_GPUS = set()

# =====================================================
# GALAX STYLE DESCRIPTIONS (från draw.js)
# =====================================================

GALAX_DESCRIPTIONS = {
    "blobbis":  "Small animal with short arms and legs. Smooth soft surface with gentle edges, not round or spherical. Friendly face with large eye or eyes, clear silhouette.",
    "kramis":   "Very large hug monster with distinct head, short legs, and long arms. Thick fluffy fur with visible volume. Broad shoulders, big feet, gentle expressive eyes.",
    "plupp":    "Small slim animal-like character with thin arms and legs, bright glowing body. Large eyes. Translucent wings attached directly to the back, clearly visible in depth.",
    "snurroga": "Medium-sized creature dressed who looks like a clown with big head, arms and one short leg. Happy patterns on clothing. Balanced proportions, joyful facial expression.",
    "sticky":   "Agile athletic superhero character with slim torso and clearly defined limbs. Alert climbing-ready posture, sharp silhouette, mischievous friendly face.",
    "wille":    "Tiny extremely cute baby character with very small body and oversized head. Short arms and legs, rounded cheeks, big sparkling eyes. Happy smile with tongue slightly out, stable standing pose."
}

# =====================================================
# IMAGE UPLOAD (flyttad från draw.js)
# =====================================================

def upload_to_comfy(base_url, image_base64):

    # Om prefix finns, ta bort det
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]

    image_bytes = base64.b64decode(image_base64)

    files = {
        "image": ("upload.png", image_bytes, "image/png")
    }

    response = requests.post(
        f"{base_url}/upload/image",
        files=files,
        timeout=GPU_TIMEOUT
    )

    response.raise_for_status()
    data = response.json()

    if "name" not in data:
        raise Exception("Upload failed")

    return data["name"]


# =====================================================
# WORKFLOW (1:1 från draw.js)
# =====================================================

def build_workflow(style_key: str, seed: Optional[int], uploaded_image_name: str):

    style_text = GALAX_DESCRIPTIONS.get(style_key, "")
    seed = seed or random.randint(1, 999999999)

    positive_text = (
        "Cute, friendly, 3D CGI children movie single superhero-creature "
        "that matches the drawing's shape and colors. "
        "Face and body with at least one leg or/and one arm. "
        "Soft studio lighting, subtle rim light, volumetric soft shadows, "
        "shallow depth of field, pristine smooth materials. "
        "Playful, heartwarming, fun for kids. "
        + style_text
    )

    negative_text = (
        "outline, wires, strings, scribbles, tangled lines, photoreal, film grain, "
        "watermark, signature, busy background, creepy, nude, human, adult human, "
        "young human kids, genitals, floating head, spherical body, ball-shaped character, "
        "blob creature, abstract form, chibi proportions, floating limbs"
    )

    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"}
        },
        "2": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "sdxl_vae.safetensors"}
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": positive_text, "clip": ["1", 1]}
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_text, "clip": ["1", 1]}
        },
        "6": {
            "class_type": "ControlNetLoader",
            "inputs": {"control_net_name": "controlnet-depth-sdxl-1.0.safetensors"}
        },
        "7": {
            "class_type": "LoadImage",
            "inputs": {"image": uploaded_name}
        },
        "9": {
            "class_type": "ControlNetApply",
            "inputs": {
                "conditioning": ["3", 0],
                "control_net": ["6", 0],
                "image": ["7", 0],
                "strength": 0.55,
                "guidance_start": 0.00,
                "guidance_end": 0.95
            }
        },
        "10": {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["1", 0],
                "clip": ["1", 1],
                "lora_name": "realcartoon3d_v17.safetensors",
                "strength_model": 0.40,
                "strength_clip": 0.40
            }
        },
        "11": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 896, "height": 896, "batch_size": 1}
        },
        "12": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["10", 0],
                "positive": ["9", 0],
                "negative": ["4", 0],
                "seed": seed,
                "steps": 32,
                "cfg": 3.3,
                "sampler_name": "dpmpp_2m_sde",
                "scheduler": "karras",
                "denoise": 1.0,
                "latent_image": ["11", 0]
            }
        },
        "13": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["12", 0], "vae": ["2", 0]}
        },
        "14": {
            "class_type": "SaveImage",
            "inputs": {"images": ["13", 0], "filename_prefix": "galax_depth_only"}
        }
    }

# =====================================================
# GPU GENERATION
# =====================================================

def wait_for_gpu(base_url, max_wait=10):
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

        time.sleep(2)

    print("🔴 GPU failed to wake")
    return False
    
@app.post("/generate")
def generate(request: GenerateRequest):

    global ACTIVE_GPU
    global DEAD_GPUS

    # 🔥 prioritera fungerande GPU
    if ACTIVE_GPU:
        endpoints = [ACTIVE_GPU] + [
            e for e in GPU_ENDPOINTS
            if e != ACTIVE_GPU and e not in DEAD_GPUS
        ]
    else:
        endpoints = [e for e in GPU_ENDPOINTS if e not in DEAD_GPUS]

    for endpoint in endpoints:

        base = endpoint.rstrip("/")
        print("🔄 Trying:", base)

        if base in DEAD_GPUS:
            continue

        try:
            if not wait_for_gpu(base):
                DEAD_GPUS.add(base)
                continue

            # 1️⃣ upload bild
            uploaded_name = upload_to_comfy(base, request.image_base64)

            # 2️⃣ bygg workflow
            workflow = build_workflow(
                request.styleKey,
                request.seed,
                uploaded_name
            )

            # 3️⃣ skicka prompt
            r = requests.post(
                f"{base}/prompt",
                json={
                    "prompt": workflow,
                    "client_id": "galax-backend"
                },
                timeout=GPU_TIMEOUT
            )

            r.raise_for_status()
            prompt_id = r.json()["prompt_id"]

            # 4️⃣ poll
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

                        image_base64 = base64.b64encode(
                            img_resp.content
                        ).decode()

                        ACTIVE_GPU = base
                        DEAD_GPUS.discard(base)

                        print("🟢 GPU SUCCESS")

                        return {
                            "status": "READY",
                            "source": base,
                            "image": f"data:image/png;base64,{image_base64}"
                        }

            print("⚠️ GPU timeout")
            DEAD_GPUS.add(base)

        except Exception as e:
            print("❌ GPU error:", e)
            DEAD_GPUS.add(base)

    ACTIVE_GPU = None

    return {
        "status": "READY",
        "source": "fallback",
        "image": next(fallback_cycle)
    }
    
# =====================================================
# FALLBACK IMAGES
# =====================================================

FALLBACK_IMAGES = [
    f"https://www-static.wemmstudios.se/wp-content/uploads/2026/02/hero_{i:02d}.png"
    for i in range(1, 15)
]

fallback_cycle = itertools.cycle(FALLBACK_IMAGES)

# =====================================================
# STATUS CHECK
# =====================================================

@app.get("/health")
def health():
    return {"status": "ok"}
