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
# GALAX STYLE DESCRIPTIONS
# =====================================================

import random
from typing import Optional

PROPORTION_VARIATIONS = [
    "slightly rounder body proportions",
    "slightly taller creature proportions",
    "slightly bigger head compared to body",
    "slightly shorter legs",
    "slightly longer arms"
]

BODY_VARIATIONS = [
    "slightly taller proportions",
    "slightly rounder proportions",
    "slightly shorter legs",
    "slightly bigger head",
    "slightly longer arms"
]

TEXTURE_VARIATIONS = [
    "soft plush surface",
    "soft fluffy fur",
    "smooth animated texture",
    "velvet-like creature skin"
]

POSE_VARIATIONS = [
    "natural relaxed pose",
    "friendly open stance",
    "subtle playful posture"
]

BACKGROUND_VARIATIONS = [
    "soft magical forest background",
    "colorful sky background",
    "simple gradient background",
    "soft glowing fantasy environment",
    "subtle playful background"
]

GALAX_DESCRIPTIONS = {

    "blobbis": (
        "Small plush creature with compact rounded body. "
        "Clear readable silhouette. Cute friendly expression."
    ),

    "kramis": (
        "Large friendly creature with big body and strong soft presence. "
        "Rounded silhouette. Gentle smiling eyes."
    ),

    "plupp": (
        "Small magical glowing creature with light delicate body. "
        "Clear fantasy silhouette. Cheerful personality."
    ),

    "snurroga": (
        "Colorful playful fantasy creature with expressive face. "
        "Distinct silhouette and joyful presence."
    ),

    "sticky": (
        "Agile fantasy creature superhero with dynamic pose. "
        "Flexible cartoon creature anatomy."
    ),

    "wille": (
        "Tiny baby fantasy creature with oversized head and small body. "
        "Innocent joyful expression."
    )
}

# =====================================================
# IMAGE UPLOAD (flyttad från draw.js)
# =====================================================

def upload_to_comfy(base_url, image_base64):

    image_bytes = base64.b64decode(image_base64.split(",")[1])

    files = {
        "image": ("upload.png", image_bytes, "image/png")
    }

    r = requests.post(
        f"{base_url}/upload/image",
        files=files,
        timeout=GPU_TIMEOUT
    )

    r.raise_for_status()
    data = r.json()

    if "name" not in data:
        raise Exception("Upload failed")

    return data["name"]


# =====================================================
# WORKFLOW (1:1 från draw.js)
# =====================================================

def build_workflow(style_key: str, seed: Optional[int], uploaded_name: str):

    style_text = GALAX_DESCRIPTIONS.get(style_key, "")

    prop = random.choice(PROPORTION_VARIATIONS)
    body = random.choice(BODY_VARIATIONS)
    texture = random.choice(TEXTURE_VARIATIONS)
    pose = random.choice(POSE_VARIATIONS)
    bg = random.choice(BACKGROUND_VARIATIONS)

    variation_text = (
        f"{prop}, {body}, {texture}, {pose}, {bg}. "
    )
    
    seed = seed or random.randint(1, 999999999)

    positive_text = (
        "High-quality stylized 3D animated fantasy creature. "
        "Inspired by the original drawing shape and colors. "
        "Keep overall silhouette similar but allow creative refinement. "
        "Preserve dominant colors from the drawing. "
        "Clearly NON-HUMAN creature. "
        "Fully covered body with fur, fabric or soft creature surface. "
        "Friendly children animation style. "
        "Soft cinematic lighting, depth of field. "
        f"{variation_text}"
        f"{style_text}"
    )

    negative_text = (
        "realistic human, realistic anatomy, nude, naked, nipples, genitalia, "
        "photorealistic, horror, creepy, glitch, distorted body, "
        "broken anatomy, extra limbs, melted body, outlines, wires"
    )

    P = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"
            }
        },
        "2": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "sdxl_vae.safetensors"
            }
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": positive_text,
                "clip": ["1", 1]
            }
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_text,
                "clip": ["1", 1]
            }
        },
        "6": {
            "class_type": "ControlNetLoader",
            "inputs": {
                "control_net_name": "controlnet-depth-sdxl-1.0.safetensors"
            }
        },
        "7": {
            "class_type": "LoadImage",
            "inputs": {
                "image": uploaded_name
            }
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
            "inputs": {
                "width": 896,
                "height": 896,
                "batch_size": 1
            }
        },
        "12": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["10", 0],
                "positive": ["9", 0],
                "negative": ["4", 0],
                "seed": int(seed or 123456789),
                "steps": 32,
                "cfg": 3.4,
                "sampler_name": "dpmpp_2m_sde",
                "scheduler": "karras",
                "denoise": 1.0,
                "latent_image": ["11", 0]
            }
        },
        "13": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["12", 0],
                "vae": ["2", 0]
            }
        },
        "15": {
            "class_type": "PreviewImage",
            "inputs": {
                "images": ["13", 0],
                "every_n_steps": 3,
                "filename_prefix": "galax_preview"
            }
        },
        "14": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["13", 0],
                "filename_prefix": "galax_depth_only"
            }
        }
    }
    
    return P

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

from fastapi import Request
from pydantic import ValidationError

@app.post("/generate")
async def generate(request: GenerateRequest):

    global ACTIVE_GPU
    global DEAD_GPUS

    print("========== /generate ==========")
    print("style:", request.styleKey)

    for endpoint in GPU_ENDPOINTS:

        base = endpoint.rstrip("/")
        print("🔄 Trying:", base)

        try:
            if not wait_for_gpu(base):
                continue

            uploaded_name = upload_to_comfy(base, request.image_base64)

            workflow = build_workflow(
                request.styleKey,
                request.seed,
                uploaded_name
            )

            r = requests.post(
                f"{base}/prompt",
                json={"prompt": workflow, "client_id": "galax-backend"},
                timeout=GPU_TIMEOUT
            )

            r.raise_for_status()
            prompt_id = r.json()["prompt_id"]

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
                history_data = h.json()

                if prompt_id not in history_data:
                    continue

                prompt_outputs = history_data[prompt_id].get("outputs", {})

                for node_id, node_data in prompt_outputs.items():
                    if "images" in node_data and len(node_data["images"]) > 0:

                        image_meta = node_data["images"][0]

                        view_url = (
                            f"{base}/view?"
                            f"filename={image_meta['filename']}&"
                            f"subfolder={image_meta.get('subfolder','')}&"
                            f"type={image_meta.get('type','output')}"
                        )

                        img_resp = requests.get(view_url, timeout=20)
                        img_resp.raise_for_status()

                        image_base64 = base64.b64encode(img_resp.content).decode()

                        print("🟢 IMAGE FOUND in node:", node_id)

                        return {
                            "status": "READY",
                            "source": base,
                            "image": f"data:image/png;base64,{image_base64}"
                        }
                        image_meta = node["images"][0]

                        view_url = (
                            f"{base}/view?"
                            f"filename={image_meta['filename']}&"
                            f"subfolder={image_meta.get('subfolder','')}&"
                            f"type={image_meta.get('type','output')}"
                        )

                        img = requests.get(view_url, timeout=20).content
                        img_b64 = base64.b64encode(img).decode()

                        print("🟢 GPU SUCCESS")

                        return {
                            "status": "READY",
                            "source": base,
                            "image": f"data:image/png;base64,{img_b64}"
                        }

        except Exception as e:
            print("❌ GPU error:", e)

    print("⚠ All GPUs failed → fallback")

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
