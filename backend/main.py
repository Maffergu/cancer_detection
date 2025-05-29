# backend/main.py

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from model_utils import super_resolve_and_classify

app = FastAPI()

# Permitir CORS para conectar con frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, pon solo tu dominio frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    # Guardar temporalmente
    temp_dir = "static/uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Ruta al modelo
    model_path = "SR/experiments/pretrained_models/ESRGAN/ESRGAN_PSNR_SRx4_DF2K_official-150ff491.pth"
    
    try:
        result = super_resolve_and_classify(temp_path, model_path)
        return JSONResponse({
            "class": result["class"],
            "confidence": result["confidence"],
            "image_url": f"/results/{os.path.basename(result['image_path'])}"
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Para servir imágenes procesadas
@app.get("/results/{filename}")
async def get_image(filename: str):
    file_path = f"static/results/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/png")
    return JSONResponse(status_code=404, content={"error": "Image not found"})
