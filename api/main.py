"""
API REST — Sistema de Detección de Fatiga en Conductores
FastAPI + TensorFlow/Keras + OpenCV
Modelo: MobileNetV2 — Test Accuracy: 99.02%

Estrategia de doble predicción:
  - Predicción A: región de ojos recortada → mejor para ojo_abierto / ojo_cerrado
  - Predicción B: cara completa            → mejor para bostezo / no_bostezo
  - Se elige la predicción con mayor confianza en su clase ganadora

Autores: Katia Paola Victoria Escalera, Isamar Chávez Díaz, Angel de Jesús Canela Figueroa
Materia: Visión Por Computadora Y Machine Learning — Mayo 2026
"""

import io
import json
import time
import logging
import urllib.request
import numpy as np
import cv2

from pathlib import Path
from PIL import Image

import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

#  Logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#  Constantes 
MODEL_PATH  = Path("model/modelo_fatiga.keras")
META_PATH   = Path("model/modelo_fatiga_metadata.json")
TARGET_SIZE = (224, 224)

CASCADE_DIR       = Path("model/cascades")
CASCADE_FACE_PATH = CASCADE_DIR / "haarcascade_frontalface_default.xml"
CASCADE_EYE_PATH  = CASCADE_DIR / "haarcascade_eye.xml"

CASCADE_FACE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
CASCADE_EYE_URL  = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"

NIVELES_ALERTA = {
    "ojo_abierto": {"nivel": "NORMAL",   "color": "#4CAF50", "mensaje": "Conductor alerta"},
    "ojo_cerrado": {"nivel": "PELIGRO",  "color": "#F44336", "mensaje": "  Ojos cerrados — posible fatiga"},
    "bostezo":     {"nivel": "ATENCION", "color": "#FF9800", "mensaje": "  Bostezo detectado — tomar descanso"},
    "no_bostezo":  {"nivel": "NORMAL",   "color": "#4CAF50", "mensaje": "Conductor alerta"},
}

# Índices de cada clase en el output del modelo
IDX = {"ojo_abierto": 0, "ojo_cerrado": 1, "bostezo": 2, "no_bostezo": 3}

#  Descarga de cascades 
def descargar_cascades():
    CASCADE_DIR.mkdir(exist_ok=True)
    for path, url in [(CASCADE_FACE_PATH, CASCADE_FACE_URL),
                      (CASCADE_EYE_PATH,  CASCADE_EYE_URL)]:
        if not path.exists():
            logger.info(f"Descargando {path.name}...")
            urllib.request.urlretrieve(url, str(path))
            logger.info(f" {path.name} listo.")

descargar_cascades()

#  Carga del modelo 
logger.info("Cargando modelo de fatiga...")
modelo    = None
metadatos = None

try:
    modelo = tf.keras.models.load_model(str(MODEL_PATH))
    logger.info(f" Modelo cargado: {MODEL_PATH}")
except Exception as e:
    logger.error(f" Error cargando modelo: {e}")

try:
    with open(META_PATH, "r") as f:
        metadatos = json.load(f)
    logger.info(f" Metadatos: {metadatos['clases']}")
except Exception as e:
    logger.warning(f"  Metadatos no encontrados: {e}")
    metadatos = {
        "clases": ["ojo_abierto", "ojo_cerrado", "bostezo", "no_bostezo"],
        "test_accuracy": 0.9902,
        "arquitectura": "MobileNetV2 + Transfer Learning",
    }

CLASES = metadatos["clases"]

#  Detectores OpenCV 
detector_cara = cv2.CascadeClassifier(str(CASCADE_FACE_PATH))
detector_ojos = cv2.CascadeClassifier(str(CASCADE_EYE_PATH))

#  App FastAPI 
app = FastAPI(
    title="API Detección de Fatiga en Conductores",
    description="MobileNetV2 + Transfer Learning. Doble predicción: ojos recortados + cara completa.",
    version="1.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  Schema 
class PrediccionResponse(BaseModel):
    clase: str
    probabilidad: float
    nivel_alerta: str
    color_alerta: str
    mensaje_alerta: str
    todas_las_clases: dict
    tiempo_inferencia_ms: float
    modo_preprocesamiento: str
    cara_detectada: bool

#  Funciones de preprocesamiento 
def aplicar_clahe(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_eq = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

def normalizar(img_bgr: np.ndarray) -> np.ndarray:
    """Resize + CLAHE + normalización → array (1, 224, 224, 3)."""
    h, w = img_bgr.shape[:2]
    interp = cv2.INTER_AREA if (w > TARGET_SIZE[0] or h > TARGET_SIZE[1]) else cv2.INTER_CUBIC
    img = cv2.resize(img_bgr, TARGET_SIZE, interpolation=interp)
    img = aplicar_clahe(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def decodificar(imagen_bytes: bytes):
    """Bytes → img_bgr + img_gray."""
    pil_img = Image.open(io.BytesIO(imagen_bytes)).convert("RGB")
    img_rgb  = np.array(pil_img, dtype=np.uint8)
    img_bgr  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_bgr, img_gray

def recortar_ojos(img_bgr, img_gray, cara):
    """Recorta la región de ojos de una cara detectada."""
    x, y, w, h = cara
    # Mitad superior = zona de ojos
    mitad_y  = y + h // 2
    roi_gray = img_gray[y:mitad_y, x:x + w]
    roi_bgr  = img_bgr[y:mitad_y, x:x + w]

    if roi_bgr.size == 0:
        return None

    # Intentar recortar ojo individual
    ojos = detector_ojos.detectMultiScale(
        roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
    )
    if len(ojos) >= 1:
        ojos_sorted = sorted(ojos, key=lambda o: o[2] * o[3], reverse=True)
        ex, ey, ew, eh = ojos_sorted[0]
        ojo_crop = roi_bgr[ey:ey + eh, ex:ex + ew]
        if ojo_crop.size > 0:
            return ojo_crop

    return roi_bgr  # fallback: mitad superior de la cara

def recortar_cara(img_bgr, cara):
    """Recorta la cara completa."""
    x, y, w, h = cara
    crop = img_bgr[y:y + h, x:x + w]
    return crop if crop.size > 0 else None

#  Lógica de doble predicción 
def predecir_doble(imagen_bytes: bytes):
    """
    Estrategia de doble predicción:

    1. Detectar cara con Haar Cascade
    2. Predicción A — región de ojos:
       Especializada en ojo_abierto / ojo_cerrado
    3. Predicción B — cara completa:
       Especializada en bostezo / no_bostezo
    4. Decisión final:
       - Tomar la confianza máxima de A en clases de ojo
       - Tomar la confianza máxima de B en clases de boca
       - Elegir la predicción con mayor confianza global

    Si no se detecta cara → una sola predicción con imagen completa.
    """
    img_bgr, img_gray = decodificar(imagen_bytes)

    # Detectar cara
    caras = detector_cara.detectMultiScale(
        img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(caras) == 0:
        # Sin cara: predicción directa con imagen completa
        pred = modelo.predict(normalizar(img_bgr), verbose=0)[0]
        idx  = int(np.argmax(pred))
        return CLASES[idx], float(pred[idx]), pred, "imagen_directa", False

    # Cara más grande
    cara = max(caras, key=lambda c: c[2] * c[3])

    #  Predicción A: región de ojos 
    region_ojos = recortar_ojos(img_bgr, img_gray, cara)
    pred_a = None
    if region_ojos is not None:
        pred_a = modelo.predict(normalizar(region_ojos), verbose=0)[0]

    #  Predicción B: cara completa 
    region_cara = recortar_cara(img_bgr, cara)
    pred_b = None
    if region_cara is not None:
        pred_b = modelo.predict(normalizar(region_cara), verbose=0)[0]

    #  Decisión final 
    # Confianza de A en clases de ojo (lo que A sabe bien)
    conf_ojos = max(
        pred_a[IDX["ojo_abierto"]],
        pred_a[IDX["ojo_cerrado"]]
    ) if pred_a is not None else 0.0

    # Confianza de B en clases de boca (lo que B sabe bien)
    conf_boca = max(
        pred_b[IDX["bostezo"]],
        pred_b[IDX["no_bostezo"]]
    ) if pred_b is not None else 0.0

    if conf_ojos >= conf_boca:
        # Gana la predicción de ojos
        pred_final = pred_a
        modo = "ojos_recortados"
    else:
        # Gana la predicción de boca/cara
        pred_final = pred_b
        modo = "cara_completa"

    idx_final = int(np.argmax(pred_final))
    return CLASES[idx_final], float(pred_final[idx_final]), pred_final, modo, True

#  Endpoints 
@app.get("/", response_class=FileResponse)
def frontend():
    return FileResponse("frontend/index.html", media_type="text/html")


@app.get("/health")
def health_check():
    return {
        "mensaje": "Sistema de Detección de Fatiga activo",
        "version": "1.2.0",
        "modelo": "MobileNetV2 + Transfer Learning",
        "test_accuracy": metadatos.get("test_accuracy"),
        "clases": CLASES,
        "modelo_cargado": modelo is not None,
        "estrategia": "Doble predicción: ojos recortados + cara completa"
    }

@app.post("/predict", response_model=PrediccionResponse)
async def predecir(file: UploadFile = File(...)):
    """
    Recibe imagen, aplica doble predicción (ojos + cara completa)
    y devuelve el estado de fatiga con mayor confianza.
    """
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Se esperaba una imagen")

    try:
        imagen_bytes = await file.read()

        t0 = time.perf_counter()
        clase, prob, pred_arr, modo, cara_detectada = predecir_doble(imagen_bytes)
        tiempo_ms = (time.perf_counter() - t0) * 1000

        alerta = NIVELES_ALERTA.get(clase, NIVELES_ALERTA["ojo_abierto"])
        todas  = {CLASES[i]: round(float(pred_arr[i]), 4) for i in range(len(CLASES))}

        logger.info(
            f"[{modo}] {'cara' if cara_detectada else 'sin cara'} -> "
            f"{clase} ({prob:.2%}) | {tiempo_ms:.1f}ms"
        )

        return PrediccionResponse(
            clase=clase,
            probabilidad=round(prob, 4),
            nivel_alerta=alerta["nivel"],
            color_alerta=alerta["color"],
            mensaje_alerta=alerta["mensaje"],
            todas_las_clases=todas,
            tiempo_inferencia_ms=round(tiempo_ms, 2),
            modo_preprocesamiento=modo,
            cara_detectada=cara_detectada
        )

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/modelo/info")
def info_modelo():
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    return {
        "arquitectura":    metadatos.get("arquitectura"),
        "clases":          CLASES,
        "input_shape":     list(modelo.input_shape[1:]),
        "parametros":      modelo.count_params(),
        "test_accuracy":   metadatos.get("test_accuracy"),
        "estrategia":      "Doble predicción: ojos recortados (ojo_abierto/ojo_cerrado) + cara completa (bostezo/no_bostezo)",
        "preprocesamiento":"Haar Cascade → CLAHE → resize 224×224 → normalización [0,1]"
    }