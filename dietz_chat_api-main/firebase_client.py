import os
import json

import firebase_admin
from firebase_admin import credentials, firestore

_db = None  # cache af firestore-klienten


def get_db() -> firestore.Client:
    """
    Giver dig en global Firestore client.
    Bruger service account JSON fra env var på Render.
    """
    global _db
    if _db is not None:
        return _db

    # Prøv begge mulige env-vars, så vi kan genbruge den du allerede har
    json_str = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON") or os.getenv(
        "FIREBASE_CREDENTIALS_JSON"
    )
    if not json_str:
        raise RuntimeError(
            "Mangler FIREBASE_SERVICE_ACCOUNT_JSON eller FIREBASE_CREDENTIALS_JSON i environment"
        )

    # Env-variablen indeholder hele JSON-filen som tekst
    service_account_info = json.loads(json_str)

    cred = credentials.Certificate(service_account_info)

    # Initialiser kun én gang pr. proces
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)

    _db = firestore.client()
    return _db
