from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import request
import requests
import itertools
import random
import time
import base64
import hashlib
import threading

REQUEST_CACHE = {}
IN_FLIGHT = set()
CACHE_TTL = 60  # sekunder
LOCK = threading.Lock()

# =====================================================
# APP INIT
# =====================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# KEEP CPU ALIVE
# =====================================================

from keep_alive import start_keep_alive

@app.on_event("startup")
def startup_event():
    start_keep_alive()


# =====================================================
# GPU EXECUTION
# =====================================================

def run_gpu_job(endpoint, request):

    base = endpoint.rstrip("/")

    try:

        # 1️⃣ Wake check
        r = requests.get(
            f"{base}/system_stats",
            timeout=(GPU_CONNECT_TIMEOUT, GPU_READ_TIMEOUT)
        )
        if r.status_code != 200:
            return None

        # 2️⃣ Upload image
        color_ratio = detect_color_ratio_from_base64(request.image_base64)
        
        img_b64 = request.image_base64.split(",")[-1]
        image_bytes = base64.b64decode(img_b64)

        files = {"image": (f"upload_{random.randint(1,9999999)}.png", image_bytes, "image/png")}

        r = requests.post(
            f"{base}/upload/image",
            files=files,
            timeout=(GPU_CONNECT_TIMEOUT, GPU_READ_TIMEOUT)
        )
        r.raise_for_status()

        uploaded_name = r.json()["name"]

        # 🔒 IMPORTANT: If upload worked, do NOT allow fallback to other GPU
        # Everything below MUST stay on this endpoint only

        workflow = build_workflow(
            request.styleKey,
            request.seed,
            uploaded_name,
            color_ratio
        )

        r = requests.post(
            f"{base}/prompt",
            json={"prompt": workflow, "client_id": "galax"},
            timeout=(GPU_CONNECT_TIMEOUT, GPU_READ_TIMEOUT)
        )
        r.raise_for_status()

        prompt_id = r.json()["prompt_id"]
        print("Prompt ID:", prompt_id)

        # Poll same GPU only
        start = time.time()

        while time.time() - start < MAX_TOTAL_TIME:

            time.sleep(POLL_INTERVAL)

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

            for node in outputs.values():
                images = node.get("images")
                if images:
                    image_meta = images[0]

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
                    return to_data_url(img_resp.content)

        return None

    except Exception as e:
        print("GPU ERROR:", e)
        return None
        

# =====================================================
# REQUEST MODEL
# =====================================================

class GenerateRequest(BaseModel):
    styleKey: str
    seed: Optional[int] = None
    image_base64: str
    mode: Optional[str] = None
    clientId: Optional[str] = None

def build_fingerprint(request: GenerateRequest) -> str:

    img_hash = hashlib.sha1(
        request.image_base64.encode()
    ).hexdigest()[:12]

    raw = f"{request.styleKey}-{request.seed}-{img_hash}-{request.clientId}"
    return hashlib.sha1(raw.encode()).hexdigest()
    
# =====================================================
# GPU CONFIG
# =====================================================

GPU_ENDPOINTS = [
    "https://w3gfk2krqc76x4-8188.proxy.runpod.net",
    "https://iygoioc7o5hytl-8188.proxy.runpod.net",
    "https://n8ghjjv9kdbhez-8188.proxy.runpod.net",
    "https://6yo3hior7k7lf8-8188.proxy.runpod.net"
]

LAST_WORKING_GPU = None


# PERFORMANCE
MAX_TOTAL_TIME = 45
POLL_INTERVAL = 0.25
GPU_CONNECT_TIMEOUT = 15
GPU_READ_TIMEOUT = 30

# =====================================================
# FALLBACK
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
# STYLE SYSTEM (SIMPLIFIED – DRAWING FIRST)
# =====================================================

BACKGROUND_GLOW_VARIATIONS = [
    "soft magical rim light",
    "subtle glowing aura around the character",
    "gentle cinematic studio lighting",
    "soft ambient glow from behind",
]

BACKGROUND_VARIATIONS = [
    "simple soft gradient background",
    "dark magical background with subtle light haze",
    "soft pastel sky background",
    "clean minimal background with light bloom",
]

GALAX_DESCRIPTIONS = {
    "blobbis": "Small compact rounded creature.",
    "kramis": "Large soft creature with big body mass.",
    "plupp": "Small magical creature with gentle glow.",
    "snurroga": "Playful fantasy creature with expressive features.",
    "sticky": "Agile creature with dynamic posture.",
    "wille": "Tiny happy creature with small body and large head."
}

# =====================================================
# MODE TUNING VARIABLES
# =====================================================
from typing import Optional
from PIL import Image
import numpy as np

# 🔹 Hur mycket färg som krävs för COLOR_MODE
COLOR_PIXEL_THRESHOLD = 0.08      # 8% färgpixlar
SCRIBBLE_PIXEL_THRESHOLD = 0.02   # under 2% = EXTREME_SCRIBBLE

# ===============================
# EXTREME SCRIBBLE MODE
# ===============================
SCRIBBLE_DENOISE = 1.0
SCRIBBLE_CFG = 3.5
SCRIBBLE_CONTROLNET_STRENGTH = 0.95
SCRIBBLE_LORA_STRENGTH = 0.75

# ===============================
# LINE MODE
# ===============================
LINE_DENOISE = 0.90
LINE_CFG = 3.0
LINE_CONTROLNET_STRENGTH = 0.85
LINE_LORA_STRENGTH = 0.65

# ===============================
# COLOR MODE
# ===============================
COLOR_DENOISE = 0.40
COLOR_CFG = 2.5
COLOR_CONTROLNET_STRENGTH = 0.75
COLOR_LORA_STRENGTH = 0.55

# ===============================
# Sampler
# ===============================
SAMPLER_STEPS = 30
SAMPLER_NAME = "dpmpp_2m_sde"
SAMPLER_SCHEDULER = "karras"

# ===============================
# Models
# ===============================
CHECKPOINT_NAME = "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"
VAE_NAME = "sdxl_vae.safetensors"
CONTROLNET_MODEL = "controlnet-canny-sdxl-1.0.safetensors"
LORA_NAME = "realcartoon3d_v17.safetensors"


# =====================================================
# COLOUR DETECTION
# =====================================================

import io
from PIL import Image
import numpy as np

def detect_color_ratio_from_base64(image_b64):

    image_bytes = base64.b64decode(image_b64.split(",")[-1])
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    arr = np.array(img)

    non_black = np.sum(
        (arr[:,:,0] > 20) |
        (arr[:,:,1] > 20) |
        (arr[:,:,2] > 20)
    )

    total = arr.shape[0] * arr.shape[1]

    return non_black / total


# =====================================================
# WORKFLOW BUILDER
# =====================================================

def build_workflow(style_key: str, seed: Optional[int], uploaded_name: str, color_ratio: float):

    seed = seed or random.randint(1, 999999999)

    # -------------------------------------------------
    # Mode detection
    # -------------------------------------------------

    if color_ratio < SCRIBBLE_PIXEL_THRESHOLD:
        mode = "EXTREME_SCRIBBLE"
        denoise = SCRIBBLE_DENOISE
        cfg = SCRIBBLE_CFG
        control_strength = SCRIBBLE_CONTROLNET_STRENGTH
        lora_strength = SCRIBBLE_LORA_STRENGTH
        latent_source = ["11", 0]

    elif color_ratio < COLOR_PIXEL_THRESHOLD:
        mode = "LINE"
        denoise = LINE_DENOISE
        cfg = LINE_CFG
        control_strength = LINE_CONTROLNET_STRENGTH
        lora_strength = LINE_LORA_STRENGTH
        latent_source = ["11", 0]

    else:
        mode = "COLOR"
        denoise = COLOR_DENOISE
        cfg = COLOR_CFG
        control_strength = COLOR_CONTROLNET_STRENGTH
        lora_strength = COLOR_LORA_STRENGTH
        latent_source = ["8", 0]

    print(f"MODE: {mode} | color_ratio={color_ratio:.3f}")

    # -------------------------------------------------
    # Prompts
    # -------------------------------------------------

    positive_text = (
        "High quality stylized 3D animated cartoon creature. "
        "Interpret the drawing as contour guide only. "
        "Build a thick solid rounded 3D character body inside the lines. "
        "Convert neon lines into soft glowing edges on a full volumetric character. "
        "Strong depth, soft shadows, cinematic rim light. "
        "Keep overall silhouette and pose from the drawing. "
        "Preserve main color idea from the lines. "
        "Clearly non-human fantasy creature."
    )

    negative_text = (
        "realistic human, photorealistic, horror, glitch, distorted body, "
        "thin line art, flat drawing, 2D sketch, wireframe, transparent body"
    )

    # -------------------------------------------------
    # Workflow
    # -------------------------------------------------

    return {

        # Checkpoint
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": CHECKPOINT_NAME
            }
        },

        # VAE
        "2": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": VAE_NAME
            }
        },

        # Positive
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": positive_text,
                "clip": ["10", 1]
            }
        },

        # Negative
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_text,
                "clip": ["10", 1]
            }
        },

        # Load drawing
        "7": {
            "class_type": "LoadImage",
            "inputs": {
                "image": uploaded_name
            }
        },

        # VAE Encode (for COLOR mode)
        "8": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["7", 0],
                "vae": ["2", 0]
            }
        },

        # ControlNet
        "6": {
            "class_type": "ControlNetLoader",
            "inputs": {
                "control_net_name": CONTROLNET_MODEL
            }
        },

        "9": {
            "class_type": "ControlNetApply",
            "inputs": {
                "conditioning": ["3", 0],
                "control_net": ["6", 0],
                "image": ["7", 0],
                "strength": control_strength,
                "guidance_start": 0.0,
                "guidance_end": 0.9
            }
        },

        # LoRA
        "10": {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["1", 0],
                "clip": ["1", 1],
                "lora_name": LORA_NAME,
                "strength_model": lora_strength,
                "strength_clip": lora_strength
            }
        },

        # Empty latent (for LINE + SCRIBBLE)
        "11": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": 896,
                "height": 896,
                "batch_size": 1
            }
        },

        # Sampler
        "12": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["10", 0],
                "positive": ["9", 0],
                "negative": ["4", 0],
                "seed": seed,
                "steps": SAMPLER_STEPS,
                "cfg": cfg,
                "sampler_name": SAMPLER_NAME,
                "scheduler": SAMPLER_SCHEDULER,
                "denoise": denoise,
                "latent_image": latent_source
            }
        },

        # Decode
        "13": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["12", 0],
                "vae": ["2", 0]
            }
        },

        # Save
        "14": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["13", 0],
                "filename_prefix": "galax"
            }
        }
    }

# =====================================================
# MAIN ENDPOINT
# =====================================================

@app.post("/generate")
async def generate(request: GenerateRequest):

    global LAST_WORKING_GPU

    fingerprint = build_fingerprint(request)
    now = time.time()

    print("========== /generate ==========")
    print("Fingerprint:", fingerprint)

    # ---------------------------------
    # CACHE CHECK
    # ---------------------------------

    with LOCK:

        # Om redan körs → blockera dubbel
        if fingerprint in IN_FLIGHT:
            print("⚠ Duplicate in-flight blocked")
            return {
                "ok": True,
                "source": "duplicate_blocked",
                "image": None
            }

        # Om nyligen färdig → returnera cached
        if fingerprint in REQUEST_CACHE:
            cached = REQUEST_CACHE[fingerprint]

            if now - cached["time"] < CACHE_TTL:
                print("⚡ Returning cached result")
                return {
                    "ok": True,
                    "source": "cache",
                    "image": cached["image"]
                }

        # Markera som aktiv
        IN_FLIGHT.add(fingerprint)


    # ---------------------------------
    # GPU EXECUTION (ONE GPU PER REQUEST)
    # ---------------------------------
    
    endpoints = GPU_ENDPOINTS.copy()
    random.shuffle(endpoints)
    
    result_image = None
    
    for endpoint in endpoints:
    
        print("Trying GPU:", endpoint)
    
        result_image = run_gpu_job(endpoint, request)
    
        if result_image:
            LAST_WORKING_GPU = endpoint
            break
    
    # ---------------------------------
    # FALLBACK
    # ---------------------------------

    if not result_image:
        print("Using fallback")

        try:
            img_url = next(fallback_cycle)
            r = requests.get(img_url, timeout=5)
            r.raise_for_status()
            result_image = to_data_url(r.content)
        except:
            result_image = None

    # ---------------------------------
    # STORE RESULT + CLEANUP
    # ---------------------------------

    with LOCK:

        if result_image:
            REQUEST_CACHE[fingerprint] = {
                "image": result_image,
                "time": time.time()
            }

        if fingerprint in IN_FLIGHT:
            IN_FLIGHT.remove(fingerprint)

    return {
        "ok": True,
        "source": "gpu" if LAST_WORKING_GPU else "fallback",
        "image": result_image
    }

# =====================================================
# HEALTH
# =====================================================

@app.get("/health")
def health():
    return {"status": "ok"}
