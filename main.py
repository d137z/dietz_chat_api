from datetime import datetime, timezone
from typing import List, Dict, Optional, Set
import os
import json
import logging
from pathlib import Path

from fastapi import FastAPI, Depends, Header, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from firebase_client import get_db

# Try import Firestore query helpers (til counters og collection_group)
try:
    from google.cloud import firestore as gc_firestore
except ImportError:
    gc_firestore = None  # type: ignore

# --- Firebase / FCM setup ---
try:
    import firebase_admin
    from firebase_admin import credentials, messaging

    FCM_AVAILABLE = True
except ImportError:
    firebase_admin = None  # type: ignore
    credentials = None  # type: ignore
    messaging = None  # type: ignore
    FCM_AVAILABLE = False

logger = logging.getLogger("dietz_chat_backend")
logging.basicConfig(level=logging.INFO)


def init_firebase_app():
    if not FCM_AVAILABLE:
        logger.warning("firebase_admin ikke installeret; push-notifikationer er slået fra.")
        return None

    if firebase_admin._apps:
        return firebase_admin.get_app()

    cred = None
    try:
        cred_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
        if cred_json:
            cred_dict = json.loads(cred_json)
            cred = credentials.Certificate(cred_dict)
        else:
            cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if cred_path:
                cred = credentials.Certificate(cred_path)

        if cred is None:
            logger.warning(
                "Ingen Firebase credentials fundet (FIREBASE_CREDENTIALS_JSON eller "
                "GOOGLE_APPLICATION_CREDENTIALS). Push-notifikationer er slået fra."
            )
            return None

        app = firebase_admin.initialize_app(cred)
        logger.info("Firebase app initialiseret til FCM.")
        return app
    except Exception as e:
        logger.exception("Kunne ikke initialisere Firebase app: %s", e)
        return None


FIREBASE_APP = init_firebase_app()

ADMIN_DEVICE_TOKENS: Set[str] = set()
ADMIN_TOKENS_INITIALIZED = False
ADMIN_TOKENS_COLLECTION = "admin_devices"


def save_admin_device_token(token: str) -> None:
    try:
        db = get_db()
    except Exception:
        logger.exception("Kunne ikke gemme admin-device token i Firestore")
        return

    doc_ref = db.collection(ADMIN_TOKENS_COLLECTION).document(token)
    try:
        doc_ref.set({"token": token, "updated_at": datetime.utcnow()}, merge=True)
    except Exception:
        logger.exception("Fejl ved skriv til admin_devices i Firestore")


def load_admin_device_tokens() -> None:
    global ADMIN_DEVICE_TOKENS
    try:
        db = get_db()
    except Exception:
        logger.exception("Kan ikke læse admin_devices i Firestore")
        return

    tokens: Set[str] = set()
    try:
        docs = db.collection(ADMIN_TOKENS_COLLECTION).stream()
    except Exception:
        logger.exception("Fejl ved stream af admin_devices collection")
        return

    for doc in docs:
        data = doc.to_dict() or {}
        token = data.get("token") or doc.id
        if token:
            tokens.add(token)

    ADMIN_DEVICE_TOKENS = tokens
    logger.info("Indlæste %d admin tokens fra Firestore", len(ADMIN_DEVICE_TOKENS))


def ensure_admin_tokens_loaded() -> None:
    global ADMIN_TOKENS_INITIALIZED
    if ADMIN_TOKENS_INITIALIZED:
        return
    ADMIN_TOKENS_INITIALIZED = True
    load_admin_device_tokens()


def send_push_to_admins(title: str, body: str, data: Optional[Dict[str, str]] = None) -> None:
    if not FCM_AVAILABLE or FIREBASE_APP is None:
        logger.info("FCM ikke konfigureret; ville have sendt push: %r / %r", title, body)
        return

    ensure_admin_tokens_loaded()
    tokens = list(ADMIN_DEVICE_TOKENS)
    if not tokens:
        logger.info("Ingen ADMIN_DEVICE_TOKENS registreret; ingen push sendt.")
        return

    for t in tokens:
        try:
            message = messaging.Message(
                notification=messaging.Notification(title=title, body=body),
                data=data or {},
                token=t,
            )
            messaging.send(message)
        except Exception:
            logger.exception("Fejl ved send af FCM-notifikation til token %s", t)


# --- FastAPI app / CORS ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.dietzcc.dk",
        "https://dietzcc.dk",
        "http://localhost:5173",
        "https://prismatic-marzipan-6f51be.netlify.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ping")
def ping():
    return {"message": "pong"}


# --- Modeller ---
class MessageIn(BaseModel):
    text: str = Field(..., min_length=1)
    conversation_id: Optional[int] = None
    name: Optional[str] = None
    email: Optional[str] = None
    client_conversation_id: Optional[str] = None
    client_message_id: Optional[str] = None

class MessageOut(BaseModel):
    id: int
    conversation_id: int
    text: str
    created_at: datetime
    sender: str
    name: Optional[str] = None
    email: Optional[str] = None
    client_message_id: Optional[str] = None

class ConversationSummary(BaseModel):
    id: int
    created_at: datetime
    last_message_at: datetime
    last_message_preview: str
    is_read: bool
    status: str

class RegisterAdminDevice(BaseModel):
    token: str


# --- In-memory cache ---
MESSAGES: List[MessageOut] = []
CONVERSATIONS: Dict[int, ConversationSummary] = {}
NEXT_MESSAGE_ID = 1
NEXT_CONVERSATION_ID = 1
COUNTERS_INITIALIZED = False

CLIENT_CONVERSATION_MAP: Dict[str, int] = {}
SEEN_CLIENT_MESSAGE_IDS: Set[str] = set()
MESSAGE_BY_CLIENT_KEY: Dict[str, MessageOut] = {}

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")

def require_admin(x_admin_token: Optional[str] = Header(default=None)):
    if ADMIN_TOKEN is None:
        return
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ----------------------------
# BilkaToGo job-kø (Firestore)
# ----------------------------
BILKA_ADMIN_PASSWORD = os.getenv("BILKA_ADMIN_PASSWORD", "")
BILKA_WORKER_TOKEN = os.getenv("BILKA_WORKER_TOKEN", "")
BILKA_JOBS_COLLECTION = "bilka_jobs"

# Render-sikkert: /tmp er altid skrivbart (ephemeral men OK til downloads)
UPLOAD_DIR = Path("/tmp/uploads/mealplans")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def require_bilka_admin(authorization: Optional[str] = Header(default=None)):
    if not BILKA_ADMIN_PASSWORD:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token != BILKA_ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized")

def require_bilka_worker(authorization: Optional[str] = Header(default=None)):
    if not BILKA_WORKER_TOKEN:
        raise HTTPException(status_code=500, detail="Worker token not configured")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token != BILKA_WORKER_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

class BilkaLoginIn(BaseModel):
    password: str

class BilkaJobCreateIn(BaseModel):
    meals: int = Field(..., ge=1, le=30)
    clear_cart: bool = False
    min_meal_dkk: float = 0
    max_total_dkk: float = 0

@app.post("/bilka/login")
def bilka_login(payload: BilkaLoginIn):
    if not BILKA_ADMIN_PASSWORD:
        raise HTTPException(status_code=500, detail="BILKA_ADMIN_PASSWORD not set on server")
    if payload.password != BILKA_ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Bad password")
    return {"token": BILKA_ADMIN_PASSWORD}

@app.post("/bilka/jobs")
def bilka_create_job(payload: BilkaJobCreateIn, _: None = Depends(require_bilka_admin)):
    db = get_db()
    job = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
        "meals": int(payload.meals),
        "clear_cart": bool(payload.clear_cart),
        "min_meal_dkk": float(payload.min_meal_dkk or 0),
        "max_total_dkk": float(payload.max_total_dkk or 0),
        "log": "",
        "pdf_url": "",
    }
    ref = db.collection(BILKA_JOBS_COLLECTION).document()
    ref.set(job)
    return {"id": ref.id}

@app.get("/bilka/jobs")
def bilka_list_jobs(limit: int = 20, _: None = Depends(require_bilka_admin)):
    limit = max(1, min(50, int(limit)))
    db = get_db()

    # Firestore sort direction: brug Query.DESCENDING hvis muligt
    direction = None
    if gc_firestore is not None:
        direction = gc_firestore.Query.DESCENDING

    q = db.collection(BILKA_JOBS_COLLECTION)
    if direction is not None:
        q = q.order_by("created_at", direction=direction)
    else:
        q = q.order_by("created_at")  # fallback

    docs = q.limit(limit).stream()
    jobs = [{"id": d.id, **(d.to_dict() or {})} for d in docs]
    return {"jobs": jobs}

@app.post("/bilka/worker/poll")
def bilka_worker_poll(_: None = Depends(require_bilka_worker)):
    db = get_db()

    q = db.collection(BILKA_JOBS_COLLECTION).where("status", "==", "pending")

    # order_by kræver ofte et index; keep it simple:
    q = q.order_by("created_at").limit(1)

    docs = list(q.stream())
    if not docs:
        return {"job": None}

    d = docs[0]
    db.collection(BILKA_JOBS_COLLECTION).document(d.id).update({"status": "running"})
    return {"job": {"id": d.id, **(d.to_dict() or {})}}

@app.post("/bilka/worker/update")
def bilka_worker_update(body: Dict, _: None = Depends(require_bilka_worker)):
    job_id = body.get("id")
    if not job_id:
        raise HTTPException(status_code=400, detail="Missing id")

    patch = {}
    for k in ("status", "log", "pdf_url"):
        if k in body:
            patch[k] = body[k]

    db = get_db()
    db.collection(BILKA_JOBS_COLLECTION).document(job_id).update(patch)
    return {"ok": True}

@app.post("/bilka/worker/upload_pdf")
def bilka_worker_upload_pdf(
    id: str = Form(...),
    file: UploadFile = File(...),
    _: None = Depends(require_bilka_worker),
):
    if not id.strip():
        raise HTTPException(status_code=400, detail="Missing id")

    out = UPLOAD_DIR / f"{id}.pdf"
    data = file.file.read()
    out.write_bytes(data)

    pdf_url = f"/bilka/jobs/{id}/pdf"
    try:
        db = get_db()
        db.collection(BILKA_JOBS_COLLECTION).document(id).update({"pdf_url": pdf_url})
    except Exception:
        logger.exception("Kunne ikke opdatere pdf_url i bilka_jobs/%s", id)

    return {"pdf_url": pdf_url}

@app.get("/bilka/jobs/{job_id}/pdf")
def bilka_job_pdf(job_id: str, _: None = Depends(require_bilka_admin)):
    path = UPLOAD_DIR / f"{job_id}.pdf"
    if not path.exists():
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(str(path), media_type="application/pdf", filename="mealplan.pdf")
