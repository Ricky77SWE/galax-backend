from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import random
import requests
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔁 WordPress fallback-bilder
FALLBACK_IMAGES = [ 
    f"https://www-static.wemmstudios.se/wp-content/uploads/2026/01/hero_{i:02d}.png"
    for i in range(1, 15)
]

@app.post("/generate")
async def generate(payload: dict):
    """
    CPU-backend:
    - Tar emot data från appen
    - Returnerar ALLTID en bild (fallback nu, GPU senare)
    """
    try:
        img_url = random.choice(FALLBACK_IMAGES)
        r = requests.get(img_url, timeout=10)
        r.raise_for_status()

        b64 = base64.b64encode(r.content).decode("utf-8")
        data_url = f"data:image/png;base64,{b64}"

        return {
            "ok": True,
            "source": "fallback",
            "image": data_url
        }

    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={
                "ok": True,
                "source": "fallback-error",
                "image": None,
                "error": str(e)
            }
        )

@app.get("/health")
async def health():
    return {"status": "ok"}
