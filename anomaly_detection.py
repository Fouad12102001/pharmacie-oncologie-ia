"""
Détection d'anomalies sur les mouvements de stock via score Z (z-score).
Utile pour repérer : erreurs de saisie (quantité x10 par erreur de frappe),
ou sorties suspectes de molécules sensibles (opioïdes, stupéfiants).

Principe : si une sortie s'écarte de plus de N écarts-types (par défaut 3)
de la moyenne habituelle pour ce médicament, elle est signalée.
"""

from pydantic import BaseModel
from typing import List
import numpy as np


class MouvementPoint(BaseModel):
    id: int
    quantite: float
    date: str


class AnomalieRequest(BaseModel):
    mouvements: List[MouvementPoint]  # historique des SORTIES pour un médicament
    seuil_zscore: float = 3.0


class AnomalieDetectee(BaseModel):
    mouvement_id: int
    quantite: float
    zscore: float
    date: str


class AnomalieResult(BaseModel):
    moyenne: float
    ecart_type: float
    anomalies: List[AnomalieDetectee]


def detecter_anomalies(req: AnomalieRequest) -> AnomalieResult:
    quantites = np.array([m.quantite for m in req.mouvements], dtype=float)

    if len(quantites) < 5:
        # Pas assez d'historique pour un z-score fiable
        return AnomalieResult(moyenne=0.0, ecart_type=0.0, anomalies=[])

    moyenne = float(np.mean(quantites))
    ecart_type = float(np.std(quantites))

    anomalies = []
    if ecart_type > 0:  # éviter division par zéro si toutes les valeurs sont identiques
        for m, q in zip(req.mouvements, quantites):
            z = (q - moyenne) / ecart_type
            if abs(z) >= req.seuil_zscore:
                anomalies.append(AnomalieDetectee(
                    mouvement_id=m.id,
                    quantite=m.quantite,
                    zscore=round(float(z), 2),
                    date=m.date,
                ))

    return AnomalieResult(
        moyenne=round(moyenne, 2),
        ecart_type=round(ecart_type, 2),
        anomalies=anomalies,
    )