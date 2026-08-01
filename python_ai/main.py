import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from transformers import CLIPProcessor, CLIPModel

from barcode_scanner import scan_code_barres, CodeBarresResult
from ocr_reader import lire_texte_ocr
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
app = FastAPI(title="AI Medicament Scanner", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= CONFIG =================
MAX_FILE_SIZE_MB = 10
MAX_IMAGE_DIMENSION = 1600
BLUR_THRESHOLD = 60.0

# Avec une liste de ~90 médicaments (contre 12 avant), le softmax répartit
# la probabilité sur beaucoup plus de candidats -> le score du bon candidat
# baisse mécaniquement même quand la détection est correcte. On abaisse donc
# le seuil par rapport à la version à 12 candidats (0.58), tout en restant
# nettement au-dessus du seuil d'origine (0.35) qui laissait passer un QR code.
# A AJUSTER avec de vrais tests sur tes propres boîtes (voir conversation).
CONFIDENCE_THRESHOLD = float(os.getenv("CLIP_CONFIDENCE_THRESHOLD", "0.42"))
TOP_K = 5  # on remonte à 5 suggestions vu le nombre de candidats plus élevé

# URL Laravel pour charger dynamiquement la liste réelle des médicaments en base
# (route déjà existante : GET /oncologie/medicaments/liste-ia -> [{id, nom}, ...])
LARAVEL_LISTE_IA_URL = os.getenv("LARAVEL_LISTE_IA_URL", "")  # ex: http://127.0.0.1/oncologie/medicaments/liste-ia
LARAVEL_FETCH_TIMEOUT = float(os.getenv("LARAVEL_FETCH_TIMEOUT", "5"))

ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/webp",
    "image/heic", "image/heif", "image/bmp", "image/tiff",
}

# ================= DEVICE =================
device = "cuda" if torch.cuda.is_available() else "cpu"

# ================= MODEL IA =================
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# ================= BASE MÉDICAMENTS D'ONCOLOGIE =================
# Liste de secours (fallback) utilisée si LARAVEL_LISTE_IA_URL n'est pas
# configurée ou injoignable. Couvre les grandes classes utilisées en
# oncologie médicale, tous types de cancer confondus :
# chimiothérapies cytotoxiques classiques, thérapies ciblées, immunothérapies,
# hormonothérapies, et médicaments de support couramment gérés par la
# pharmacie oncologique (anti-émétiques, facteurs de croissance, protecteurs).
MEDICAMENTS_ONCOLOGIE_DEFAUT = [
    # ── Agents alkylants ──
    "Cyclophosphamide", "Ifosfamide", "Cisplatine", "Carboplatine",
    "Oxaliplatine", "Chlorambucil", "Melphalan", "Busulfan",
    "Dacarbazine", "Temozolomide", "Procarbazine", "Bendamustine",

    # ── Antimétabolites ──
    "Methotrexate", "Fluorouracile", "Capecitabine", "Gemcitabine",
    "Cytarabine", "Pemetrexed", "Fludarabine", "Mercaptopurine",
    "Thioguanine", "Hydroxyuree", "Cladribine", "Azacitidine",
    "Decitabine", "Nelarabine",

    # ── Antibiotiques antitumoraux ──
    "Doxorubicine", "Epirubicine", "Bleomycine", "Mitomycine",
    "Daunorubicine", "Idarubicine", "Mitoxantrone", "Doxorubicine liposomale",
    "Dactinomycine",

    # ── Inhibiteurs de topoisomérase ──
    "Etoposide", "Irinotecan", "Topotecan", "Teniposide",

    # ── Poisons du fuseau (taxanes / vinca-alcaloïdes) ──
    "Paclitaxel", "Docetaxel", "Vincristine", "Vinblastine",
    "Vinorelbine", "Cabazitaxel", "Nab-paclitaxel", "Eribuline",

    # ── Anticorps monoclonaux / thérapies ciblées ──
    "Trastuzumab", "Rituximab", "Bevacizumab", "Cetuximab",
    "Panitumumab", "Pertuzumab", "Trastuzumab emtansine",
    "Obinutuzumab", "Daratumumab", "Brentuximab vedotin",
    "Ramucirumab", "Blinatumomab",

    # ── Inhibiteurs de tyrosine kinase / thérapies orales ciblées ──
    "Imatinib", "Erlotinib", "Gefitinib", "Sunitinib",
    "Sorafenib", "Lapatinib", "Dasatinib", "Nilotinib",
    "Pazopanib", "Regorafenib", "Axitinib", "Ibrutinib",
    "Palbociclib", "Ribociclib", "Osimertinib", "Vemurafenib",
    "Dabrafenib", "Trametinib", "Crizotinib", "Olaparib",

    # ── Immunothérapies (inhibiteurs de checkpoint) ──
    "Pembrolizumab", "Nivolumab", "Ipilimumab", "Atezolizumab",
    "Durvalumab", "Avelumab",

    # ── Hormonothérapies (sein / prostate) ──
    "Tamoxifene", "Letrozole", "Anastrozole", "Exemestane",
    "Fulvestrant", "Bicalutamide", "Flutamide", "Leuproreline",
    "Goserelline", "Triptoreline", "Abiraterone", "Enzalutamide",

    # ── Autres agents oncologiques ──
    "Asparaginase", "Bortezomib", "Lenalidomide", "Thalidomide",
    "Pomalidomide", "Interferon alfa", "Interleukine 2", "Arsenic trioxide",
    "Tretinoine", "Hydroxycarbamide",

    # ── Médicaments de support en oncologie ──
    "Ondansetron", "Granisetron", "Palonosetron", "Aprepitant",
    "Filgrastim", "Pegfilgrastim", "Erythropoietine", "Folinate de calcium",
    "Mesna", "Dexamethasone", "Amifostine", "Darbepoetine alfa",
]

# ================= CHARGEMENT DYNAMIQUE (Laravel -> fallback statique) =================
def charger_liste_medicaments() -> List[str]:
    """
    Tente de charger la liste réelle des médicaments depuis Laravel
    (route GET /oncologie/medicaments/liste-ia, déjà existante).
    Retombe sur la liste statique d'oncologie en cas d'échec ou si
    LARAVEL_LISTE_IA_URL n'est pas configurée.
    """
    if not LARAVEL_LISTE_IA_URL:
        logger.info("LARAVEL_LISTE_IA_URL non configurée -> utilisation de la liste statique d'oncologie.")
        return list(MEDICAMENTS_ONCOLOGIE_DEFAUT)

    try:
        import requests
        resp = requests.get(LARAVEL_LISTE_IA_URL, timeout=LARAVEL_FETCH_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        noms = [item["nom"] for item in data if item.get("nom")]

        if not noms:
            logger.warning("Liste Laravel vide -> fallback sur la liste statique d'oncologie.")
            return list(MEDICAMENTS_ONCOLOGIE_DEFAUT)

        logger.info(f"{len(noms)} médicaments chargés depuis Laravel ({LARAVEL_LISTE_IA_URL}).")
        return noms
    except Exception as e:
        logger.warning(f"Impossible de charger la liste depuis Laravel ({e}) -> fallback statique.")
        return list(MEDICAMENTS_ONCOLOGIE_DEFAUT)


MEDICAMENTS = charger_liste_medicaments()
logger.info(
    f"Device: {device} | HEIC: {HEIC_SUPPORTED} | Seuil confiance: {CONFIDENCE_THRESHOLD} "
    f"| Médicaments chargés: {len(MEDICAMENTS)}"
)


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


class ReloadResult(BaseModel):
    status: str
    medicaments_count: int
    source: str


# ================= VALIDATION / OUVERTURE IMAGE =================
async def load_image_from_upload(file: UploadFile) -> Image.Image:
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
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
        image.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Format d'image non reconnu ou fichier corrompu.")
    except Exception as e:
        logger.error(f"Erreur ouverture image: {e}")
        raise HTTPException(status_code=400, detail="Impossible de lire l'image.")

    image = ImageOps.exif_transpose(image)

    if image.mode != "RGB":
        image = image.convert("RGB")

    if max(image.size) > MAX_IMAGE_DIMENSION:
        image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.LANCZOS)

    return image


def is_blurry(image: Image.Image) -> float:
    gray = np.array(image.convert("L"), dtype=np.float64)
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
    image = await load_image_from_upload(file)
    return run_scan(image)


@app.post("/scan-frame", response_model=ScanResult)
async def scan_frame(file: UploadFile = File(...)):
    image = await load_image_from_upload(file)
    return run_scan(image)


@app.post("/scan-code-barres", response_model=CodeBarresResult)
async def scan_barcode(file: UploadFile = File(...)):
    """
    Lecture de code-barres/QR/DataMatrix (lot + dates + GTIN).
    Si AUCUN code n'est détecté sur l'image (boîte sans code, code
    endommagé/flou), bascule automatiquement sur l'OCR en dernier recours
    pour tenter de lire lot/dates directement dans le texte imprimé.
    """
    image = await load_image_from_upload(file)
    resultat = scan_code_barres(image)

    if resultat.trouve:
        return resultat

    # Aucun code détecté -> tentative OCR de secours
    ocr = lire_texte_ocr(image)

    if not ocr.disponible:
        # Tesseract non installé sur le serveur -> on renvoie le résultat
        # "non trouvé" tel quel, sans faire échouer la requête.
        return resultat

    if ocr.numero_lot or ocr.date_expiration or ocr.date_fabrication:
        return CodeBarresResult(
            trouve=True,
            source="ocr",
            numero_lot=ocr.numero_lot,
            date_expiration=ocr.date_expiration,
            date_fabrication=ocr.date_fabrication,
            raw=ocr.texte_brut,
        )

    # OCR a tourné mais n'a rien reconnu de fiable -> on reste honnête
    return resultat


@app.post("/prevision-stock", response_model=PrevisionResult)
async def prevision_stock(req: PrevisionRequest):
    return previsions_consommation(req)


@app.post("/detecter-anomalies", response_model=AnomalieResult)
async def anomalies_stock(req: AnomalieRequest):
    return detecter_anomalies(req)


@app.post("/reload-medicaments", response_model=ReloadResult)
def reload_medicaments():
    """
    Recharge la liste des médicaments à chaud (sans redémarrer uvicorn) —
    utile après avoir ajouté de nouveaux médicaments dans Laravel.
    A appeler manuellement (ex: bouton admin, ou cron) si LARAVEL_LISTE_IA_URL
    est configurée.
    """
    global MEDICAMENTS
    MEDICAMENTS = charger_liste_medicaments()
    return ReloadResult(
        status="ok",
        medicaments_count=len(MEDICAMENTS),
        source="laravel" if LARAVEL_LISTE_IA_URL else "liste_statique_oncologie",
    )


@app.get("/")
def home():
    return {
        "status": "AI Medicament Scanner Running",
        "device": device,
        "heic_support": HEIC_SUPPORTED,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "top_k": TOP_K,
        "medicaments_count": len(MEDICAMENTS),
        "medicaments_source": "laravel" if LARAVEL_LISTE_IA_URL else "liste_statique_oncologie",
    }


@app.get("/medicaments", response_model=List[str])
def liste_medicaments():
    """Debug : voir la liste actuellement chargée en mémoire."""
    return MEDICAMENTS