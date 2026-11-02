"""
Module de remplacement de l'Agent LLM dans le workflow Abafusion.
Prend en entree le JSON produit par Detection_anomaly, renvoie
exactement le meme contrat de sortie que l'ancien Agent LLM :
{"severity": int, "analyse": str}
"""
import pickle
import pandas as pd
import numpy as np

with open('fleetguard_severity_model.pkl', 'rb') as f:
    _bundle = pickle.load(f)

_MODEL = _bundle['model']
_FEATURE_COLUMNS = _bundle['feature_columns']
_CAT_COLUMNS = _bundle['categorical_columns']

SEVERITY_LABELS = {1: 'NORMAL', 2: 'SURVEILLANCE', 3: 'CRITIQUE'}


def _extract_features(payload: dict) -> dict:
    """Extrait les 19 features attendues a partir du JSON complet (meme structure
    que l'ancienne entree de l'Agent LLM)."""
    ca = payload['consumption_analysis']
    ad = payload['anomaly_detection']
    vr = payload.get('vehicle_reference', {})
    metrics = ad.get('metrics', {})

    anomaly_types = {a['type'] for a in ad.get('anomalies', [])}
    overconsumption_detected = int('OVERCONSUMPTION' in anomaly_types)
    excessive_idle_detected = int('EXCESSIVE_IDLE' in anomaly_types)
    impossible_refuel_detected = int('IMPOSSIBLE_REFUEL' in anomaly_types)

    fuel_added_l = payload.get('fuel_added_l', 0)
    tank_capacity = vr.get('fuel_capacity_liters', payload.get('fuel_tank_capacity_l', 1)) or 1
    fuel_start_l = payload.get('fuel_level_start_l', 0)
    fuel_end_l = payload.get('fuel_level_l', 0)
    idle_seconds = payload.get('idle_duration_seconds', 0)
    trip_seconds = payload.get('trip_duration_seconds', 1) or 1

    return {
        'trip_distance_km': payload.get('trip_distance_km', ca.get('trip_distance_km')),
        'trip_duration_hours': trip_seconds / 3600.0,
        'reference_consumption_l_100km': ca.get('reference_consumption_l_100km'),
        'calculated_actual_consumption_l_100km': ca.get('calculated_actual_consumption_l_100km'),
        'deviation_percent': ca.get('deviation_percent'),
        'fuel_difference_l': ca.get('fuel_difference_l'),
        'overconsumption_percent': ca.get('overconsumption_percent', max(ca.get('deviation_percent', 0), 0)),
        'idle_ratio_percent': (idle_seconds / trip_seconds * 100) if trip_seconds else 0,
        'fuel_added_l': fuel_added_l,
        'refuel_ratio_percent': (fuel_added_l / tank_capacity * 100) if tank_capacity else 0,
        'fuel_start_percent': (fuel_start_l / tank_capacity * 100) if tank_capacity else 0,
        'fuel_end_percent': (fuel_end_l / tank_capacity * 100) if tank_capacity else 0,
        'vehicle_type': vr.get('vehicle_type', 'Van'),
        'fuel_type': vr.get('fuel_type', 'Diesel'),
        'mission_status': payload.get('mission_status', 'COMPLETED'),
        'overconsumption_detected': overconsumption_detected,
        'excessive_idle_detected': excessive_idle_detected,
        'impossible_refuel_detected': impossible_refuel_detected,
        'anomaly_count': ad.get('anomaly_count', overconsumption_detected + excessive_idle_detected + impossible_refuel_detected),
    }


def _vectorize(features: dict) -> pd.DataFrame:
    row = pd.DataFrame([features])
    row = pd.get_dummies(row, columns=_CAT_COLUMNS, drop_first=False)
    # aligner exactement sur les colonnes vues a l'entrainement
    row = row.reindex(columns=_FEATURE_COLUMNS, fill_value=0)
    return row


def _generate_analyse(payload: dict, features: dict, severity: int) -> str:
    """Genere un texte d'analyse dans le meme style que l'ancien Agent LLM,
    par template (pas de LLM), a partir des memes chiffres."""
    ca = payload['consumption_analysis']
    ad = payload['anomaly_detection']
    distance = features['trip_distance_km']
    actual = ca.get('calculated_actual_consumption_l_100km')
    ref = ca.get('reference_consumption_l_100km')
    dev = ca.get('deviation_percent')
    anomaly_types = [a['type'] for a in ad.get('anomalies', [])]

    parts = []
    if features['overconsumption_detected']:
        mot = 'critique' if severity == 3 else ('notable' if severity == 2 else 'legere')
        parts.append(
            f"Une surconsommation {mot} de {dev:.2f} % a ete detectee, avec une consommation "
            f"reelle calculee a {actual:.2f} L/100 km pour une reference de {ref:.2f} L/100 km "
            f"sur un trajet de {distance:.0f} km."
        )
    if features['excessive_idle_detected']:
        parts.append(
            f"Un ratio de ralenti eleve ({features['idle_ratio_percent']:.1f} % du temps de trajet) "
            f"a egalement ete releve."
        )
    if features['impossible_refuel_detected']:
        parts.append(
            f"Un plein incoherent a ete signale (volume ajoute representant "
            f"{features['refuel_ratio_percent']:.1f} % de la capacite du reservoir)."
        )
    if not parts:
        parts.append(
            f"Le trajet de {distance:.0f} km presente une consommation conforme aux attentes "
            f"({actual:.2f} L/100 km pour une reference de {ref:.2f} L/100 km)."
        )

    if severity == 3:
        parts.append("Cet ecart important necessite une verification des donnees de consommation et du fonctionnement du vehicule.")
    elif severity == 2:
        parts.append("Une surveillance de ce vehicule est recommandee lors des prochains trajets.")
    else:
        parts.append("Aucune action n'est requise pour ce trajet.")

    return " ".join(parts)


def predict_severity(payload: dict) -> dict:
    """Point d'entree principal : remplace l'appel a l'Agent LLM.
    payload = le meme JSON complet qui etait envoye a l'ancien Agent LLM.
    Retourne : {"severity": int, "analyse": str}
    """
    features = _extract_features(payload)
    X = _vectorize(features)
    severity = int(_MODEL.predict(X)[0])
    analyse = _generate_analyse(payload, features, severity)
    return {"severity": severity, "analyse": analyse}


if __name__ == '__main__':
    import json
    example = {
        "vehicle_id": "VH-001", "vehicle_model": "Dacia Dokker", "driver_name": "Said Youlakou",
        "trip_route": "Casablanca -> Rabat", "trip_distance_km": 87, "trip_duration_seconds": 14400,
        "fuel_tank_capacity_l": 50, "fuel_level_start_l": 41.9, "fuel_level_l": 34.8,
        "fuel_added_l": 0, "idle_duration_seconds": 0,
        "vehicle_reference": {
            "vehicle_id": "VH-001", "vehicle_name": "Dacia Dokker", "vehicle_type": "Van",
            "fuel_type": "Diesel", "fuel_capacity_liters": 50, "reference_consumption_l_100km": 5.5,
            "status": "active"
        },
        "consumption_analysis": {
            "trip_distance_km": 87, "fuel_start_l": 41.9, "fuel_added_l": 0, "fuel_end_l": 34.8,
            "calculated_fuel_used_l": 7.1, "reference_consumption_l_100km": 5.5,
            "expected_fuel_for_trip_l": 4.79, "calculated_actual_consumption_l_100km": 8.16,
            "fuel_difference_l": 2.32, "deviation_l_100km": 2.66, "deviation_percent": 48.38,
            "overconsumption_percent": 48.38, "consumption_direction": "ABOVE_REFERENCE"
        },
        "anomaly_detection": {
            "is_anomaly": True, "anomaly_count": 1,
            "anomalies": [{"type": "OVERCONSUMPTION", "detected": True, "value": 48.38, "unit": "%"}],
            "metrics": {"deviation_percent": 48.38, "idle_ratio_percent": 0}
        }
    }
    result = predict_severity(example)
    print(json.dumps(result, indent=2, ensure_ascii=False))
