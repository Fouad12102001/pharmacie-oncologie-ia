from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import io
import logging

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from transformers import CLIPProcessor, CLIPModel

from barcode_scanner import scan_code_barres, CodeBarresResult
from forecasting import previsions_consommation, PrevisionRequest, PrevisionResult
from anomaly_detection import detecter_anomalies, AnomalieRequest, AnomalieResult

# pillow-heif permet de lire les photos iPhone (.heic/.heif)
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:
    HEIC_SUPPORTED = False

# ================= LOGGING =================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-scanner")

# ================= APP =================
app = FastAPI(title="AI Medicament Scanner", version="2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= CONFIG =================
MAX_FILE_SIZE_MB = 10
MAX_IMAGE_DIMENSION = 1600  # redimensionnement pour perf, sans perdre en qualité utile
BLUR_THRESHOLD = 60.0       # en dessous -> image jugée trop floue
CONFIDENCE_THRESHOLD = 0.35 # en dessous -> "aucun match fiable"
TOP_K = 3

ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/webp",
    "image/heic", "image/heif", "image/bmp", "image/tiff",
}

# ================= DEVICE =================
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device utilisé: {device} | Support HEIC: {HEIC_SUPPORTED}")

# ================= MODEL IA =================
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# ================= BASE MEDICAMENTS =================
# Idéalement à terme : charger dynamiquement depuis l'API Laravel (/api/medicaments/liste-ia)
# plutôt que hardcodé, pour rester synchronisé avec le vrai stock.
MEDICAMENTS = [
    "Cisplatine",
    "Paracétamol",
    "Amoxicilline",
    "Morphine",
    "Insuline",
    "Aspirine",
    "Omeprazole",
    "Salbutamol",
    "Furosemide",
    "Ibuprofen",
    "Tramadol",
    "Prednisolone",
]


# ================= SCHEMAS =================
class Candidat(BaseModel):
    nom: str
    confidence: float


class ScanResult(BaseModel):
    nom_detecte: str | None
    confidence: float
    candidats: List[Candidat]
    status: str          # success | low_confidence | blurry | error
    message: str | None = None


# ================= VALIDATION / OUVERTURE IMAGE =================
async def load_image_from_upload(file: UploadFile) -> Image.Image:
    """
    Lit et valide n'importe quel type d'image envoyé (upload classique
    ou capture caméra en blob), sans jamais écrire sur le disque.
    """
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        # On ne bloque pas forcément (certains navigateurs envoient
        # 'application/octet-stream' pour un blob canvas) mais on log.
        logger.warning(f"Content-Type inattendu: {file.content_type}")

    raw = await file.read()

    if not raw:
        raise HTTPException(status_code=400, detail="Fichier vide.")

    size_mb = len(raw) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"Image trop volumineuse ({size_mb:.1f} Mo, max {MAX_FILE_SIZE_MB} Mo)."
        )

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()  # force la lecture complète -> détecte les fichiers corrompus ici
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Format d'image non reconnu ou fichier corrompu.")
    except Exception as e:
        logger.error(f"Erreur ouverture image: {e}")
        raise HTTPException(status_code=400, detail="Impossible de lire l'image.")

    # Corrige l'orientation selon les métadonnées EXIF (photos prises en mode portrait)
    image = ImageOps.exif_transpose(image)

    # Convertit tout vers RGB (gère CMYK, palette, RGBA, niveaux de gris...)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Redimensionne si trop grande (accélère l'inférence, CLIP n'a pas besoin de plus)
    if max(image.size) > MAX_IMAGE_DIMENSION:
        image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.LANCZOS)

    return image


def is_blurry(image: Image.Image) -> float:
    """
    Détecte le flou via variance du Laplacien (sans dépendance OpenCV,
    implémenté avec numpy pour rester léger).
    Retourne un score : plus bas = plus flou.
    """
    gray = np.array(image.convert("L"), dtype=np.float64)
    # Noyau Laplacien 3x3 appliqué manuellement via convolution simple
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    from scipy.signal import convolve2d
    lap = convolve2d(gray, kernel, mode="valid")
    return float(lap.var())


# ================= CORE IA =================
def predict_medicament(image: Image.Image) -> List[Candidat]:
    inputs = processor(
        text=MEDICAMENTS,
        images=image,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = outputs.logits_per_image.softmax(dim=1)[0]

    top_k = min(TOP_K, len(MEDICAMENTS))
    top_probs, top_indices = torch.topk(probs, top_k)

    return [
        Candidat(nom=MEDICAMENTS[idx.item()], confidence=round(float(p), 3))
        for p, idx in zip(top_probs, top_indices)
    ]


def run_scan(image: Image.Image) -> ScanResult:
    blur_score = is_blurry(image)
    if blur_score < BLUR_THRESHOLD:
        return ScanResult(
            nom_detecte=None,
            confidence=0.0,
            candidats=[],
            status="blurry",
            message="Image trop floue, veuillez reprendre la photo en stabilisant l'appareil.",
        )

    candidats = predict_medicament(image)
    best = candidats[0]

    if best.confidence < CONFIDENCE_THRESHOLD:
        return ScanResult(
            nom_detecte=None,
            confidence=best.confidence,
            candidats=candidats,
            status="low_confidence",
            message="Aucun médicament identifié avec certitude. Voici les meilleures suggestions.",
        )

    return ScanResult(
        nom_detecte=best.nom,
        confidence=best.confidence,
        candidats=candidats,
        status="success",
    )


# ================= ROUTES =================
@app.post("/scan", response_model=ScanResult)
async def scan_medicament(file: UploadFile = File(...)):
    """Scan d'une image uploadée (galerie, fichier, etc.)"""
    image = await load_image_from_upload(file)
    return run_scan(image)


@app.post("/scan-frame", response_model=ScanResult)
async def scan_frame(file: UploadFile = File(...)):
    """Scan d'une frame de caméra live (capture directe)."""
    image = await load_image_from_upload(file)
    return run_scan(image)


@app.post("/scan-code-barres", response_model=CodeBarresResult)
async def scan_barcode(file: UploadFile = File(...)):
    """Lecture de code-barres/DataMatrix GS1 (lot + date d'expiration fiables)."""
    image = await load_image_from_upload(file)
    return scan_code_barres(image)


@app.post("/prevision-stock", response_model=PrevisionResult)
async def prevision_stock(req: PrevisionRequest):
    """Prévision de consommation + date de rupture estimée (Holt-Winters)."""
    return previsions_consommation(req)


@app.post("/detecter-anomalies", response_model=AnomalieResult)
async def anomalies_stock(req: AnomalieRequest):
    """Détection de sorties de stock anormales (z-score)."""
    return detecter_anomalies(req)


@app.get("/")
def home():
    return {
        "status": "AI Medicament Scanner Running",
        "device": device,
        "heic_support": HEIC_SUPPORTED,
        "medicaments_count": len(MEDICAMENTS),
    }