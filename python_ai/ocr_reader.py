"""
Lecture OCR de secours, utilisée UNIQUEMENT quand aucun code-barres/QR/
DataMatrix n'a pu être détecté sur l'image (boîte sans code, code
endommagé, photo qui ne cadre pas le code). Extrait le texte imprimé sur
l'emballage via Tesseract OCR, puis repère le numéro de lot et les dates
grâce à des expressions régulières tolérantes (FR/EN), car le texte imprimé
sur les boîtes pharmaceutiques suit rarement une mise en page fixe.

Dépendance SYSTÈME requise (pas seulement pip) :
    sudo apt install tesseract-ocr tesseract-ocr-fra
    pip install pytesseract

Sans le binaire Tesseract installé, ce module se désactive proprement
(disponible=False) plutôt que de faire planter le service.
"""

import re
import logging
from typing import Optional
from PIL import Image, ImageOps, ImageFilter
from pydantic import BaseModel

logger = logging.getLogger("ocr-reader")

try:
    import pytesseract
    TESSERACT_OK = True
except ImportError:
    TESSERACT_OK = False
    logger.warning("pytesseract non installé -> le fallback OCR sera désactivé.")


class OcrResult(BaseModel):
    disponible: bool                      # False si Tesseract n'est pas installé
    texte_brut: Optional[str] = None
    numero_lot: Optional[str] = None
    date_expiration: Optional[str] = None
    date_fabrication: Optional[str] = None


# ── Expressions régulières génériques (FR/EN), tolérantes à la casse et aux séparateurs ──
RE_LOT = re.compile(
    r"(?:LOT|BATCH|N[°o]?\s*LOT)[\s:\-\.]{0,3}([A-Z0-9\-]{3,15})",
    re.IGNORECASE,
)
RE_EXP = re.compile(
    r"(?:EXP(?:IRE)?\.?|PEREMPTION|P[ÉE]REMPTION|BBE|A\s+UTILISER\s+AVANT)"
    r"[\s:\-\.]{0,3}(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4}|\d{1,2}[\/\.\-]\d{4})",
    re.IGNORECASE,
)
RE_FAB = re.compile(
    r"(?:FAB(?:RICATION)?\.?|MFG|MFD|FABRIQUE\s+LE)"
    r"[\s:\-\.]{0,3}(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4}|\d{1,2}[\/\.\-]\d{4})",
    re.IGNORECASE,
)


def _normaliser_date(texte_date: str) -> Optional[str]:
    """Convertit une date DD/MM/YYYY, DD/MM/YY ou MM/YYYY (séparateurs / . -) vers ISO 8601."""
    texte_date = texte_date.replace(".", "/").replace("-", "/")
    parties = texte_date.split("/")

    try:
        if len(parties) == 3:
            jj, mm, aaaa = parties
            if len(aaaa) == 2:
                aaaa = "20" + aaaa
            return f"{int(aaaa):04d}-{int(mm):02d}-{int(jj):02d}"
        elif len(parties) == 2:
            mm, aaaa = parties
            if len(aaaa) == 2:
                aaaa = "20" + aaaa
            return f"{int(aaaa):04d}-{int(mm):02d}-01"
    except (ValueError, IndexError):
        return None
    return None


def extraire_champs_texte(texte: str) -> dict:
    """
    Applique les regex de reconnaissance sur un texte brut (sortie OCR, ou
    contenu d'un QR/code-barres qui n'est PAS au format GS1 structuré) pour
    en extraire lot/dates si présents. Réutilisée aussi par barcode_scanner.py
    pour les codes non-GS1.
    """
    resultat = {"numero_lot": None, "date_expiration": None, "date_fabrication": None}

    m_lot = RE_LOT.search(texte)
    if m_lot:
        resultat["numero_lot"] = m_lot.group(1).strip()

    m_exp = RE_EXP.search(texte)
    if m_exp:
        resultat["date_expiration"] = _normaliser_date(m_exp.group(1))

    m_fab = RE_FAB.search(texte)
    if m_fab:
        resultat["date_fabrication"] = _normaliser_date(m_fab.group(1))

    return resultat


def _pretraiter_image_ocr(image: Image.Image) -> Image.Image:
    """Améliore les chances de lecture OCR : niveaux de gris, contraste, netteté, taille."""
    img = image.convert("L")
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)

    # Tesseract lit mieux du texte suffisamment grand
    if max(img.size) < 1200:
        ratio = 1200 / max(img.size)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)

    return img


def lire_texte_ocr(image: Image.Image) -> OcrResult:
    """Point d'entrée principal : lit le texte de l'image et en extrait lot/dates."""
    if not TESSERACT_OK:
        return OcrResult(disponible=False)

    try:
        img_pretraitee = _pretraiter_image_ocr(image)
        # "fra+eng" : les boîtes du CLCC sont majoritairement en français,
        # mais certaines mentions (EXP, LOT, MFG) sont en anglais.
        texte = pytesseract.image_to_string(img_pretraitee, lang="fra+eng")
    except Exception as e:
        logger.error(f"Erreur OCR: {e}")
        return OcrResult(disponible=False)

    if not texte or not texte.strip():
        return OcrResult(disponible=True, texte_brut="")

    champs = extraire_champs_texte(texte)

    return OcrResult(
        disponible=True,
        texte_brut=texte.strip(),
        numero_lot=champs["numero_lot"],
        date_expiration=champs["date_expiration"],
        date_fabrication=champs["date_fabrication"],
    )