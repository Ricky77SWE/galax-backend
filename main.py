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
import signal

# =====================================================
# TIMEOUT SAFETY
# =====================================================

def timeout_handler(signum, frame):
    raise TimeoutError("Request timed out")

signal.signal(signal.SIGALRM, timeout_handler)

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
MAX_POLL_SECONDS = 40

# =====================================================
# FALLBACK IMAGES
# =====================================================

FALLBACK_IMAGES = [
    f"https://www-static.wemmstudios.se/wp-content/uploads/2026/02/hero_{i:02d}.png"
    for i in range(1, 15)
]

fallback_cycle = itertools.cycle(FALLBACK_IMAGES)

# =====================================================
# UTIL
# =====================================================

def to_data_url(image_bytes: bytes) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"
    
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
    "soft magical pink background",
    "colorful sky background",
    "simple gradient background",
    "soft glowing fantasy rainbow environment",
    "subtle playful light gren background"
]

GALAX_DESCRIPTIONS = {

    "blobbis": (
        "Small cute plush animal with compact rounded body. "
        "Clear readable silhouette. Cute friendly expression."
    ),

    "kramis": (
        "Large friendly creature with big soft body and Gentle smiling eyes."
    ),

    "plupp": (
        "Small magical glowing creature. Cheerful personality."
    ),

    "snurroga": (
        "Colorful playful fantasy creature with expressive face. "
    ),

    "sticky": (
        "Agile fantasy creature with dynamic pose. Flexible cartoon anatomy."
    ),

    "wille": (
        "Tiny cure, happy baby fantasy animal creature with oversized head and small body. "
    )
}

# =====================================================
# IMAGE UPLOAD (flyttad från draw.js)
# =====================================================

def upload_to_comfy(base_url, image_base64):

    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]

    image_bytes = base64.b64decode(image_base64)

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
        "High-quality stylized 3D animated cute fantasy creature. "
        "STRICTLY preserve original silhouette, body proportions and limb placement. Do NOT modify the overall shape."
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
                "strength": 0.50,
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
                "strength_model": 0.35,
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
                "steps": 28,
                "cfg": 3.4,
                "sampler_name": "dpmpp_2m_sde",
                "scheduler": "karras",
                "denoise": 0.85,
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
    
import requests
import time
import base64
import random
import itertools
import concurrent.futures

# ==========================================
# PERFORMANCE CONFIG
# ==========================================

POLL_INTERVAL = 0.25          # 🔥 snabb polling
MAX_TOTAL_TIME = 18           # 🔥 hård timeout (sekunder)
GPU_CONNECT_TIMEOUT = 5
GPU_READ_TIMEOUT = 15

# ==========================================
# UTIL
# ==========================================

def to_data_url(image_bytes: bytes) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"

# ==========================================
# FAST GPU EXECUTION (SINGLE GPU)
# ==========================================

def run_gpu_job(endpoint, request):

    base = endpoint.rstrip("/")

    try:
        # 1️⃣ Wake test
        r = requests.get(
            f"{base}/system_stats",
            timeout=(GPU_CONNECT_TIMEOUT, GPU_READ_TIMEOUT)
        )
        if r.status_code != 200:
            return None

        # 2️⃣ Upload
        image_base64 = request.image_base64
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        image_bytes = base64.b64decode(image_base64)

        files = {
            "image": ("upload.png", image_bytes, "image/png")
        }

        r = requests.post(
            f"{base}/upload/image",
            files=files,
            timeout=(GPU_CONNECT_TIMEOUT, GPU_READ_TIMEOUT)
        )
        r.raise_for_status()
        uploaded_name = r.json()["name"]

        # 3️⃣ Workflow
        workflow = build_workflow(
            request.styleKey,
            request.seed,
            uploaded_name
        )

        r = requests.post(
            f"{base}/prompt",
            json={"prompt": workflow, "client_id": "galax"},
            timeout=(GPU_CONNECT_TIMEOUT, GPU_READ_TIMEOUT)
        )
        r.raise_for_status()
        prompt_id = r.json()["prompt_id"]

        # 4️⃣ FAST POLLING LOOP
        start = time.time()

        while True:

            if time.time() - start > MAX_TOTAL_TIME:
                return None

            time.sleep(POLL_INTERVAL)

            try:
                h = requests.get(
                    f"{base}/history/{prompt_id}",
                    timeout=(GPU_CONNECT_TIMEOUT, GPU_READ_TIMEOUT)
                )

                if h.status_code != 200:
                    continue

                data = h.json()
                if prompt_id not in data:
                    continue

                outputs = data[prompt_id].get("outputs", {})
                save_node = outputs.get("14")

                if save_node and save_node.get("images"):
                    image_meta = save_node["images"][0]

                    view_url = (
                        f"{base}/view?"
                        f"filename={image_meta['filename']}&"
                        f"subfolder={image_meta.get('subfolder','')}&"
                        f"type={image_meta.get('type','output')}"
                    )

                    img_resp = requests.get(
                        view_url,
                        timeout=(GPU_CONNECT_TIMEOUT, GPU_READ_TIMEOUT)
                    )
                    img_resp.raise_for_status()

                    print("🟢 GPU SUCCESS:", base)

                    return to_data_url(img_resp.content)

            except:
                continue

    except:
        return None

    return None

# ==========================================
# PARALLEL GPU EXECUTION
# ==========================================

def try_all_gpus_parallel(request):

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(GPU_ENDPOINTS)) as executor:

        futures = [
            executor.submit(try_single_gpu, endpoint, request)
            for endpoint in GPU_ENDPOINTS
        ]

        for future in concurrent.futures.as_completed(futures):

            try:
                result = future.result()

                if result:
                    print("🟢 First GPU returned result")
                    return result

            except Exception as e:
                print("❌ GPU future failed:", e)

    # Om ingen GPU gav bild
    print("⚠ No GPU returned image")
    return None

# ==========================================
# MAIN ENDPOINT
# ==========================================

@app.post("/generate")
async def generate(request: GenerateRequest):

    print("========== /generate ==========")
    print("Style:", request.styleKey)

    image_data_url = try_all_gpus_parallel(request)

    if image_data_url:
        return {
            "ok": True,
            "source": "gpu",
            "image": image_data_url
        }

    # ---------- FALLBACK ----------
    print("⚠ Using fallback")

    try:
        img_url = next(fallback_cycle)

        r = requests.get(img_url, timeout=5)
        r.raise_for_status()

        b64 = base64.b64encode(r.content).decode()

        return {
            "ok": True,
            "source": "fallback",
            "image": f"data:image/png;base64,{b64}"
        }

    except Exception as e:
        print("💀 FALLBACK FAILED:", e)

        return {
            "ok": True,
            "source": "emergency",
            "image": None
        }
    
    # -------------------------------------------------
    # FALLBACK (ALWAYS SAFE)
    # -------------------------------------------------

    print("All GPUs failed → fallback")

    try:
        img_url = next(fallback_cycle)

        r = requests.get(img_url, timeout=10)
        r.raise_for_status()

        return {
            "status": "READY",
            "image": to_data_url(r.content)
        }

    except Exception as e:
        print("Fallback failed:", e)

        # Emergency blank image
        blank = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAACXBIWXMAAAsTAAALEwEAmpwYAAA"
            "AAXNSR0IArs4c6QAAABxpRE9UAAAAAgAAAAAAAAABAAAAKAAAACgAAAABAAAABAAAACgAAAAB"
        )

        return {
            "status": "READY",
            "image": to_data_url(blank)
        }

# =====================================================
# STATUS CHECK
# =====================================================

@app.get("/health")
def health():
    return {"status": "ok"}

