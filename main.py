import os
import random
import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FALLBACK_DIR = os.path.join(BASE_DIR, "fallback")
COMFYUI_URL = os.getenv("COMFYUI_URL")


def get_fallback_image():
    files = [f for f in os.listdir(FALLBACK_DIR) if f.endswith(".png")]
    if not files:
        raise RuntimeError("No fallback images found")
    return os.path.join(FALLBACK_DIR, random.choice(files))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
def generate():
    try:
        if not COMFYUI_URL:
            raise RuntimeError("COMFYUI_URL not set")

        r = requests.post(
            COMFYUI_URL,
            json={"prompt": "test"},
            timeout=25
        )
        r.raise_for_status()

        # Här hade du normalt returnerat GPU-bilden
        raise RuntimeError("GPU response handling not implemented")

    except Exception as e:
        print("⚠️ Using fallback:", e)
        fallback_path = get_fallback_image()
        return FileResponse(
            fallback_path,
            media_type="image/png",
            headers={"X-Fallback": "true"}
        )
