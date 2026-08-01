"""
Lecture de codes-barres, QR codes et DataMatrix présents sur les emballages
pharmaceutiques. pyzbar décode nativement de nombreuses symbologies
(DATAMATRIX, QRCODE, EAN13, EAN8, UPCA, CODE128, CODE39, PDF417...) —
aucune restriction de type n'est appliquée ici, TOUT code lisible est traité.

Trois niveaux de reconnaissance, du plus fiable au moins fiable :
  1. GS1 structuré (DataMatrix pharma norme GS1) -> lot + dates + GTIN fiables
  2. GTIN simple (EAN13/UPC classique)           -> gtin seul
  3. QR/texte libre non-GS1                       -> extraction par regex
     (réutilise les mêmes règles que le fallback OCR, cf. ocr_reader.py)

Si le code est mal cadré, incliné ou peu contrasté, plusieurs prétraitements
(rotation, contraste) sont tentés avant d'abandonner.
"""

from pyzbar import pyzbar
from PIL import Image, ImageOps
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel

from ocr_reader import extraire_champs_texte  # règles regex partagées avec l'OCR


class CodeBarresResult(BaseModel):
    trouve: bool
    source: str = "aucun"          # gs1_datamatrix | code_simple | qr_texte_libre | aucun
    gtin: Optional[str] = None
    numero_lot: Optional[str] = None
    date_expiration: Optional[str] = None    # format ISO (AI 17, ou regex générique)
    date_fabrication: Optional[str] = None   # format ISO (AI 11, ou regex générique)
    type_code: Optional[str] = None          # DATAMATRIX, QRCODE, EAN13, CODE128...
    raw: Optional[str] = None


# Identifiants d'application GS1 (AI) les plus utilisés en pharma.
# Volontairement élargi par rapport à la version précédente pour couvrir
# davantage de variantes réelles rencontrées sur les emballages.
GS1_AI_PATTERNS = {
    "01": 14,   # GTIN - longueur fixe
    "17": 6,    # Date expiration YYMMDD - longueur fixe
    "11": 6,    # Date de fabrication/production YYMMDD - longueur fixe
    "13": 6,    # Date d'emballage YYMMDD - longueur fixe
    "15": 6,    # Date "à consommer de préférence avant" YYMMDD - longueur fixe
    "10": None, # Numéro de lot - longueur variable
    "21": None, # Numéro de série - longueur variable
}

GS1_SEPARATOR = "\x1d"  # FNC1, séparateur standard entre champs variables


def parse_gs1(raw: str) -> dict:
    """
    Parse une chaîne GS1 brute en dictionnaire de champs.
    Contrairement à la version précédente, un AI inconnu ne stoppe plus
    tout le parsing : on avance prudemment pour tenter de récupérer les
    champs suivants malgré tout (les emballages réels ne respectent pas
    toujours l'ordre AI "canonique").
    """
    result = {}
    i = 0
    raw = raw.replace(GS1_SEPARATOR, "|")

    while i < len(raw) - 1:
        ai = raw[i:i + 2]

        if ai not in GS1_AI_PATTERNS:
            i += 1  # avance d'1 caractère (et non 2) pour ne pas rater un AI décalé
            continue

        length = GS1_AI_PATTERNS[ai]
        i += 2

        if length:
            value = raw[i:i + length]
            i += length
        else:
            end = raw.find("|", i)
            if end == -1:
                value = raw[i:]
                i = len(raw)
            else:
                value = raw[i:end]
                i = end + 1

        result[ai] = value

    return result


def format_date_gs1(yymmdd: str) -> Optional[str]:
    """Convertit une date GS1 (YYMMDD) en ISO 8601."""
    if not yymmdd or len(yymmdd) != 6 or not yymmdd.isdigit():
        return None
    try:
        yy, mm, dd = yymmdd[:2], yymmdd[2:4], yymmdd[4:6]
        year = 2000 + int(yy)
        day = int(dd) if dd != "00" else 1
        return datetime(year, int(mm), day).strftime("%Y-%m-%d")
    except (ValueError, IndexError):
        return None


def _pretraitements_image(image: Image.Image) -> List[Image.Image]:
    """
    Génère plusieurs variantes de l'image pour maximiser les chances de
    détection : l'originale, et une version niveaux de gris + contraste
    renforcé (utile sur les photos peu nettes ou mal éclairées).
    """
    variantes = [image]
    try:
        gris_contraste = ImageOps.autocontrast(image.convert("L"))
        variantes.append(gris_contraste)
    except Exception:
        pass
    return variantes


def _decoder_tous_codes(image: Image.Image) -> list:
    """
    Essaie de décoder l'image sous plusieurs rotations (0°/90°/180°/270°)
    et plusieurs prétraitements, car une photo prise à main levée n'est
    presque jamais parfaitement droite. S'arrête dès qu'au moins un code
    est trouvé, pour ne pas multiplier inutilement les calculs.
    """
    vus = set()
    resultats = []

    for variante in _pretraitements_image(image):
        for angle in (0, 90, 180, 270):
            img_test = variante.rotate(angle, expand=True) if angle else variante
            try:
                codes = pyzbar.decode(img_test)
            except Exception:
                continue

            for c in codes:
                cle = (c.type, c.data)
                if cle not in vus:
                    vus.add(cle)
                    resultats.append(c)

        if resultats:
            break  # une variante a fonctionné, inutile de tester la suivante

    return resultats


def scan_code_barres(image: Image.Image) -> CodeBarresResult:
    """Détecte et décode le meilleur code-barres/QR/DataMatrix présent dans l'image."""
    codes = _decoder_tous_codes(image)

    if not codes:
        return CodeBarresResult(trouve=False, source="aucun")

    # ── Priorité 1 : chercher un code GS1 structuré parmi TOUS les codes détectés ──
    for code in codes:
        raw = code.data.decode("utf-8", errors="ignore")
        fields = parse_gs1(raw)

        if fields.get("10") or fields.get("17"):
            return CodeBarresResult(
                trouve=True,
                source="gs1_datamatrix",
                gtin=fields.get("01"),
                numero_lot=fields.get("10"),
                date_expiration=format_date_gs1(fields.get("17")) if fields.get("17") else None,
                date_fabrication=format_date_gs1(fields.get("11")) if fields.get("11") else None,
                type_code=code.type,
                raw=raw,
            )

    # ── Priorité 2 : pas de GS1 structuré -> on prend le premier code lisible ──
    code = codes[0]
    raw = code.data.decode("utf-8", errors="ignore")

    # GTIN simple (EAN13/EAN8/UPC) : uniquement des chiffres, longueur standard
    if raw.isdigit() and len(raw) in (8, 12, 13, 14):
        return CodeBarresResult(
            trouve=True,
            source="code_simple",
            gtin=raw,
            type_code=code.type,
            raw=raw,
        )

    # QR / texte libre non-GS1 : on tente d'y repérer lot/dates par regex,
    # avec les mêmes règles que le fallback OCR (cas fréquent pour des
    # codes internes/génériques qui ne suivent pas la norme GS1)
    champs = extraire_champs_texte(raw)

    return CodeBarresResult(
        trouve=True,
        source="qr_texte_libre",
        numero_lot=champs["numero_lot"],
        date_expiration=champs["date_expiration"],
        date_fabrication=champs["date_fabrication"],
        type_code=code.type,
        raw=raw,
    )