"""
Lecture de codes-barres GS1 DataMatrix présents sur les emballages pharmaceutiques.
Beaucoup plus fiable que l'OCR pour le numéro de lot et la date d'expiration,
car ces données sont encodées numériquement (norme GS1), pas juste imprimées en texte.
 
Format GS1 typique décodé : (01)GTIN(17)DateExpiration(10)NumeroLot
Exemple brut : 010360012345678917260731109A1B2C3
"""
 
from pyzbar import pyzbar
from PIL import Image
from datetime import datetime
import re
from typing import Optional
from pydantic import BaseModel
 
 
class CodeBarresResult(BaseModel):
    trouve: bool
    gtin: Optional[str] = None
    numero_lot: Optional[str] = None
    date_expiration: Optional[str] = None  # format ISO
    type_code: Optional[str] = None
    raw: Optional[str] = None
 
 
# Identifiants d'application GS1 (AI) les plus utilisés en pharma
GS1_AI_PATTERNS = {
    "01": 14,   # GTIN - longueur fixe 14
    "17": 6,    # Date expiration YYMMDD - longueur fixe 6
    "10": None, # Numéro de lot - longueur variable (jusqu'au séparateur)
    "21": None, # Numéro de série - longueur variable
}
 
GS1_SEPARATOR = "\x1d"  # FNC1, séparateur standard entre champs variables
 
 
def parse_gs1(raw: str) -> dict:
    """Parse une chaîne GS1 brute en dictionnaire de champs."""
    result = {}
    i = 0
    raw = raw.replace(GS1_SEPARATOR, "|")  # normalisation du séparateur
 
    while i < len(raw):
        ai = raw[i:i + 2]
        if ai not in GS1_AI_PATTERNS:
            break
 
        length = GS1_AI_PATTERNS[ai]
        i += 2
 
        if length:
            value = raw[i:i + length]
            i += length
        else:
            # Longueur variable : on lit jusqu'au séparateur ou à la fin
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
    if not yymmdd or len(yymmdd) != 6:
        return None
    try:
        yy, mm, dd = yymmdd[:2], yymmdd[2:4], yymmdd[4:6]
        # Règle GS1 : si dd == 00, dernier jour du mois -> ici on simplifie au 1er
        year = 2000 + int(yy)
        day = int(dd) if dd != "00" else 1
        return datetime(year, int(mm), day).strftime("%Y-%m-%d")
    except (ValueError, IndexError):
        return None
 
 
def scan_code_barres(image: Image.Image) -> CodeBarresResult:
    """Détecte et décode tout code-barres/DataMatrix présent dans l'image."""
    codes = pyzbar.decode(image)
 
    if not codes:
        return CodeBarresResult(trouve=False)
 
    code = codes[0]  # on privilégie le premier détecté
    raw = code.data.decode("utf-8", errors="ignore")
    type_code = code.type  # ex: "DATAMATRIX", "CODE128", "EAN13"
 
    fields = parse_gs1(raw)
 
    return CodeBarresResult(
        trouve=True,
        gtin=fields.get("01"),
        numero_lot=fields.get("10"),
        date_expiration=format_date_gs1(fields.get("17")) if fields.get("17") else None,
        type_code=type_code,
        raw=raw,
    )