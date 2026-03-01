from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
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
    "https://n8ghjjv9kdbhez-8188.proxy.runpod.net"
]

LAST_WORKING_GPU = None

# PERFORMANCE
MAX_TOTAL_TIME = 25
POLL_INTERVAL = 0.25
GPU_CONNECT_TIMEOUT = 5
GPU_READ_TIMEOUT = 15

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
# WORKFLOW BUILDER
# =====================================================

def build_workflow(style_key: str, seed: Optional[int], uploaded_name: str):

    seed = seed or random.randint(1, 999999999)

    variation_text = (
        f"{random.choice(BACKGROUND_VARIATIONS)}, "
        f"{random.choice(BACKGROUND_GLOW_VARIATIONS)}."
    )

    positive_text = (
        "High quality stylized 3D animated cartoon creature. "
        "Convert neon line drawing into a solid 3D character body. "
        "Thick rounded volumetric body instead of thin lines. "
        "Build full character volume from the sketch. "
        "Soft cinematic lighting and colorful rim glow. "
        "Keep original pose and overall silhouette from the drawing. "
        "Preserve main color idea from the lines. "
        "Clearly non-human fantasy creature."
        f"{GALAX_DESCRIPTIONS.get(style_key, '')} "
        f"{variation_text}"
    )

    negative_text = (
        "realistic human, photorealistic, horror, glitch, distorted body, "
        "extra limbs, thin line art, flat drawing, 2D sketch"
    )

    return {

        # Load checkpoint
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"
            }
        },

        # Load VAE
        "2": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "sdxl_vae.safetensors"
            }
        },

        # Positive prompt
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": positive_text,
                "clip": ["1", 1]
            }
        },

        # Negative prompt
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_text,
                "clip": ["1", 1]
            }
        },

        # Load drawing
        "7": {
            "class_type": "LoadImage",
            "inputs": {
                "image": uploaded_name
            }
        },

        # Depth ControlNet
        "6": {
            "class_type": "ControlNetLoader",
            "inputs": {
                "control_net_name": "controlnet-depth-sdxl-1.0.safetensors"
            }
        },

        # Apply ControlNet
        "9": {
            "class_type": "ControlNetApply",
            "inputs": {
                "conditioning": ["3", 0],
                "control_net": ["6", 0],
                "image": ["7", 0],
                "strength": 0.4,
                "guidance_start": 0.0,
                "guidance_end": 0.8
            }
        },

        # Encode drawing for img2img
        "8": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["7", 0],
                "vae": ["2", 0]
            }
        },

        # LoRA loader
        "10": {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["1", 0],
                "clip": ["1", 1],
                "lora_name": "realcartoon3d_v17.safetensors",
                "strength_model": 0.75,
                "strength_clip": 0.75
            }
        },

        # Sampler
        "12": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["10", 0],
                "positive": ["9", 0],   # ControlNet-conditioned prompt
                "negative": ["4", 0],
                "seed": seed,
                "steps": 34,
                "cfg": 5.0,
                "sampler_name": "dpmpp_2m_sde",
                "scheduler": "karras",
                "denoise": 0.88,
                "latent_image": ["8", 0]
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
        img_b64 = request.image_base64.split(",")[-1]
        image_bytes = base64.b64decode(img_b64)

        files = {"image": ("upload.png", image_bytes, "image/png")}

        r = requests.post(
            f"{base}/upload/image",
            files=files,
            timeout=(GPU_CONNECT_TIMEOUT, GPU_READ_TIMEOUT)
        )
        r.raise_for_status()

        uploaded_name = r.json()["name"]

        # 3️⃣ Build workflow
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

        # 4️⃣ Poll for result
        start = time.time()

        while time.time() - start < MAX_TOTAL_TIME:

            time.sleep(POLL_INTERVAL)

            try:
                h = requests.get(
                    f"{base}/history/{prompt_id}",
                    timeout=(GPU_CONNECT_TIMEOUT, GPU_READ_TIMEOUT)
                )
            except Exception as e:
                print("History fetch error:", e)
                continue

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

                    print("🟢 GPU SUCCESS:", base)
                    return to_data_url(img_resp.content)

        return None

    except Exception as e:
        print("GPU ERROR:", e)
        return None
        
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
    # GPU EXECUTION
    # ---------------------------------

    endpoints = GPU_ENDPOINTS.copy()

    if LAST_WORKING_GPU in endpoints:
        endpoints.remove(LAST_WORKING_GPU)
        endpoints.insert(0, LAST_WORKING_GPU)

    result_image = None

    for endpoint in endpoints:

        print("Trying GPU:", endpoint)

        result = run_gpu_job(endpoint, request)

        if result:
            LAST_WORKING_GPU = endpoint
            result_image = result
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
