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
FALLBACK_LOCK = threading.Lock()
GPU_INDEX = 0

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
        metrics = analyze_drawing(request.image_base64)

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
            metrics
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
            
            # history kan ibland vara tom första sekunderna
            if not data:
                continue
            
            if prompt_id not in data:
                continue
            
            outputs = data[prompt_id].get("outputs", {})
            
            for node_id, node in outputs.items():
            
                if "images" in node and len(node["images"]) > 0:
            
                    image_meta = node["images"][0]
                    
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
                    print("Image received from GPU")
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
    "https://bzlfpe77lovz54-8188.proxy.runpod.net",
    "https://w3gfk2krqc76x4-8188.proxy.runpod.net",
    "https://n8ghjjv9kdbhez-8188.proxy.runpod.net",
    "https://6yo3hior7k7lf8-8188.proxy.runpod.net"    
]

LAST_WORKING_GPU = None
MAX_GPU_ATTEMPTS = 4

# PERFORMANCE
MAX_TOTAL_TIME = 45
POLL_INTERVAL = 0.5
GPU_CONNECT_TIMEOUT = 6
GPU_READ_TIMEOUT = 20

# =====================================================
# FALLBACK
# =====================================================

FALLBACK_IMAGES = [
    f"https://www-static.wemmstudios.se/wp-content/uploads/2026/02/hero_{i:02d}.png"
    for i in range(1, 15)
]

fallback_cycle = itertools.cycle(FALLBACK_IMAGES)
LAST_FALLBACK_IMAGE = None

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
# MODE TUNING VARIABLES (GALAX AI ENGINE)
# =====================================================

from typing import Optional
import io
import base64
import random
from PIL import Image
import numpy as np

# =====================================================
# Sampler (snabb men stabil)
# =====================================================

SAMPLER_STEPS = 24
SAMPLER_NAME = "dpmpp_2m_sde"
SAMPLER_SCHEDULER = "karras"

CFG_SCALE = 3.5

# =====================================================
# Models
# =====================================================

CHECKPOINT_NAME = "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"
VAE_NAME = "sdxl_vae.safetensors"

# =====================================================
# DRAWING ANALYSIS (GALAX ENGINE)
# =====================================================

def analyze_drawing(image_b64):

    image_bytes = base64.b64decode(image_b64.split(",")[-1])
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    arr = np.array(img)

    h, w, _ = arr.shape
    total_pixels = h * w

    # ---------------------------------
    # how much drawing exists
    # ---------------------------------

    non_black = np.sum(
        (arr[:,:,0] > 20) |
        (arr[:,:,1] > 20) |
        (arr[:,:,2] > 20)
    )

    coverage = non_black / total_pixels

    # ---------------------------------
    # colorfulness metric
    # ---------------------------------

    r = arr[:,:,0].astype(float)
    g = arr[:,:,1].astype(float)
    b = arr[:,:,2].astype(float)

    rg = np.abs(r - g)
    yb = np.abs(0.5*(r+g) - b)

    colorfulness = np.mean(rg) + np.mean(yb)

    # ---------------------------------
    # brightness
    # ---------------------------------

    brightness = np.mean(arr)

    return {
        "coverage": coverage,
        "colorfulness": colorfulness,
        "brightness": brightness
    }


# =====================================================
# CHARACTER STYLE GENERATION
# =====================================================

def build_character_style(metrics):

    coverage = metrics["coverage"]
    colorfulness = metrics["colorfulness"]
    brightness = metrics["brightness"]

    # BODY SIZE

    if coverage < 0.02:
        body = "tiny skinny creature"
    elif coverage < 0.06:
        body = "small creature"
    elif coverage < 0.15:
        body = "medium creature"
    else:
        body = "large round fluffy creature"

    # COLOR STYLE

    if colorfulness < 15:
        colors = "soft pastel colors"
    elif colorfulness < 40:
        colors = "bright cartoon colors"
    else:
        colors = "very colorful rainbow fantasy colors"

    # PERSONALITY

    if brightness < 60:
        personality = "mysterious magical creature"
    elif brightness < 120:
        personality = "cute friendly creature"
    else:
        personality = "happy energetic creature"

    return body, colors, personality


# =====================================================
# RANDOM CREATURE FEATURES (MAGIC SYSTEM)
# =====================================================

def generate_creature_features():

    eyes = random.choice([
        "two big eyes",
        "three glowing eyes",
        "one giant eye",
        "many tiny eyes"
    ])

    ears = random.choice([
        "small ears",
        "long bunny ears",
        "tiny horns",
        "round ears",
        "no ears"
    ])

    tail = random.choice([
        "short tail",
        "long fluffy tail",
        "tiny tail",
        "no tail"
    ])

    texture = random.choice([
        "soft fluffy fur",
        "smooth cartoon skin",
        "slimy texture",
        "sparkling magical skin"
    ])

    extra = random.choice([
        "tiny wings",
        "small horns",
        "antenna",
        "glowing spots",
        "none"
    ])

    return f"{eyes}, {ears}, {tail}, {texture}, {extra}"


# =====================================================
# WORKFLOW BUILDER (NO CONTROLNET)
# =====================================================

def build_workflow(style_key: str, seed: Optional[int], uploaded_name: str, metrics):

    seed = seed or random.randint(1, 999999999)

    body, colors, personality = build_character_style(metrics)
    features = generate_creature_features()
    
    print("Drawing metrics:", metrics)
    print("Generated character:", body, colors, personality)

    positive_text = f"""
    Cute 3D cartoon fantasy creature.

    Body type: {body}
    Color palette: {colors}
    Personality: {personality}

    Creature features: {features}

    Pixar style character.
    Large expressive eyes.
    Rounded shapes.
    Soft materials.

    Magical creature from the GALAX universe.

    High quality 3D render.
    Studio lighting.
    Soft shadows.

    Inspired by a child's drawing but NOT following the lines.
    """

    negative_text = """
    realistic human
    photorealistic
    horror
    ugly
    distorted anatomy
    thin stick figure
    line drawing
    scribble
    sketch
    anime
    2D illustration
    """
    
    return {

        # MODEL
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

        # POSITIVE PROMPT
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": positive_text,
                "clip": ["1",1]
            }
        },

        # NEGATIVE PROMPT
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_text,
                "clip": ["1",1]
            }
        },

        # LATENT IMAGE
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": 768,
                "height": 768,
                "batch_size": 1
            }
        },

        # SAMPLER
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1",0],
                "positive": ["3",0],
                "negative": ["4",0],
                "seed": seed,
                "steps": SAMPLER_STEPS,
                "cfg": CFG_SCALE,
                "sampler_name": SAMPLER_NAME,
                "scheduler": SAMPLER_SCHEDULER,
                "denoise": 1.0,
                "latent_image": ["5",0]
            }
        },

        # DECODE
        "7": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["6",0],
                "vae": ["2",0]
            }
        },

        # SAVE
        "8": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["7",0],
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
    global LAST_FALLBACK_IMAGE
    global GPU_INDEX

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
    
    global GPU_INDEX
    
    result_image = None
    
    # skapa endpoint-lista
    endpoints = GPU_ENDPOINTS.copy()
    
    # prioritera senaste fungerande GPU
    if LAST_WORKING_GPU in endpoints:
        endpoints.remove(LAST_WORKING_GPU)
        endpoints.insert(0, LAST_WORKING_GPU)
    
    # round-robin startposition
    with LOCK:
        start_index = GPU_INDEX % len(endpoints)
        GPU_INDEX += 1
    
    # rotera listan
    endpoints = endpoints[start_index:] + endpoints[:start_index]
    
    # försök några GPUs
    for endpoint in endpoints[:MAX_GPU_ATTEMPTS]:
    
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
            with FALLBACK_LOCK:
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
