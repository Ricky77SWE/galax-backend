from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import base64
from itertools import cycle

from keep_alive import start_keep_alive

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔁 WordPress fallback-bilder
FALLBACK_IMAGES = [
    f"https://www-static.wemmstudios.se/wp-content/uploads/2026/02/hero_{i:02d}.png"
    for i in range(1, 15)
]

fallback_cycle = cycle(FALLBACK_IMAGES)

@app.on_event("startup")
def startup_event():
    start_keep_alive()

@app.post("/generate")
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
