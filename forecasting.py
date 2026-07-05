"""
Prévision de consommation de médicaments (donc de date probable de rupture)
via lissage exponentiel de Holt-Winters, plus fin qu'une simple moyenne mobile
car il capture la TENDANCE (consommation croissante/décroissante), pas juste
une moyenne plate sur les 30 derniers jours.
 
Reçoit depuis Laravel l'historique des sorties de stock (mouvements_stock),
renvoie une prévision sur N jours + la date de rupture estimée.
"""
 
from pydantic import BaseModel
from typing import List, Optional
from datetime import date, timedelta
import numpy as np
 
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_OK = True
except ImportError:
    STATSMODELS_OK = False
 
 
class HistoriquePoint(BaseModel):
    date: str        # "2026-06-01"
    quantite: float  # quantité sortie ce jour-là
 
 
class PrevisionRequest(BaseModel):
    historique: List[HistoriquePoint]  # idéalement >= 14 jours de données
    stock_actuel: float
    horizon_jours: int = 30
 
 
class PrevisionResult(BaseModel):
    methode: str
    consommation_prevue_par_jour: List[float]
    jours_avant_rupture: Optional[int]
    date_rupture_estimee: Optional[str]
    fiabilite: str  # "faible" | "moyenne" | "bonne"
 
 
def previsions_consommation(req: PrevisionRequest) -> PrevisionResult:
    valeurs = [p.quantite for p in req.historique]
    n = len(valeurs)
 
    # Pas assez de données -> fallback moyenne simple, on est transparent sur la fiabilité
    if n < 14 or not STATSMODELS_OK:
        moyenne = float(np.mean(valeurs)) if valeurs else 0.0
        prevision = [moyenne] * req.horizon_jours
        fiabilite = "faible"
        methode = "moyenne_simple"
    else:
        try:
            serie = np.array(valeurs, dtype=float)
            # trend="add" capte une tendance linéaire, seasonal désactivé
            # (peu de médicaments ont une saisonnalité hebdo claire en oncologie,
            # à activer plus tard si on a un historique >90 jours)
            modele = ExponentialSmoothing(
                serie, trend="add", seasonal=None, initialization_method="estimated"
            ).fit()
            prevision = list(modele.forecast(req.horizon_jours))
            prevision = [max(0.0, float(v)) for v in prevision]  # pas de conso négative
            fiabilite = "bonne" if n >= 30 else "moyenne"
            methode = "holt_winters"
        except Exception:
            moyenne = float(np.mean(valeurs))
            prevision = [moyenne] * req.horizon_jours
            fiabilite = "faible"
            methode = "moyenne_simple_fallback"
 
    # Calcul du jour de rupture : on soustrait les prévisions cumulées au stock actuel
    stock_restant = req.stock_actuel
    jours_avant_rupture = None
    for i, conso_jour in enumerate(prevision):
        stock_restant -= conso_jour
        if stock_restant <= 0:
            jours_avant_rupture = i + 1
            break
 
    date_rupture = (
        (date.today() + timedelta(days=jours_avant_rupture)).isoformat()
        if jours_avant_rupture else None
    )
 
    return PrevisionResult(
        methode=methode,
        consommation_prevue_par_jour=[round(v, 2) for v in prevision],
        jours_avant_rupture=jours_avant_rupture,
        date_rupture_estimee=date_rupture,
        fiabilite=fiabilite,
    )