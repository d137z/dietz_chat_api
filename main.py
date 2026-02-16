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
# Chat (Firestore-baseret, med RAM fallback)
# ----------------------------

def _parse_dt(x) -> datetime:
    # Firestore kan give datetime direkte, ellers str; vi falder tilbage pænt.
    if isinstance(x, datetime):
        return x
    return datetime.utcnow()


def init_counters_from_firestore() -> None:
    """
    Sync NEXT_CONVERSATION_ID og NEXT_MESSAGE_ID med det, der allerede ligger i Firestore.
    Kører kun én gang pr. proces.
    """
    global COUNTERS_INITIALIZED, NEXT_CONVERSATION_ID, NEXT_MESSAGE_ID

    if COUNTERS_INITIALIZED:
        return

    if gc_firestore is None:
        logger.warning("google.cloud.firestore ikke tilgængelig; springer counter-init over")
        COUNTERS_INITIALIZED = True
        return

    try:
        db = get_db()
    except Exception:
        logger.exception("Kunne ikke få Firestore-klient til at læse counters")
        COUNTERS_INITIALIZED = True
        return

    max_conv_id = 0
    max_msg_id = 0

    try:
        conv_query = (
            db.collection("conversations")
            .order_by("id", direction=gc_firestore.Query.DESCENDING)
            .limit(1)
        )
        conv_docs = list(conv_query.stream())
        if conv_docs:
            data = conv_docs[0].to_dict() or {}
            if "id" in data:
                max_conv_id = int(data["id"])
    except Exception:
        logger.exception("Fejl ved læsning af max conversation id fra Firestore")

    try:
        msg_query = (
            db.collection_group("messages")
            .order_by("id", direction=gc_firestore.Query.DESCENDING)
            .limit(1)
        )
        msg_docs = list(msg_query.stream())
        if msg_docs:
            data = msg_docs[0].to_dict() or {}
            if "id" in data:
                max_msg_id = int(data["id"])
    except Exception:
        logger.exception("Fejl ved læsning af max message id fra Firestore")

    if max_conv_id >= NEXT_CONVERSATION_ID:
        NEXT_CONVERSATION_ID = max_conv_id + 1
    if max_msg_id >= NEXT_MESSAGE_ID:
        NEXT_MESSAGE_ID = max_msg_id + 1

    COUNTERS_INITIALIZED = True
    logger.info(
        "Counters initialiseret fra Firestore: NEXT_CONVERSATION_ID=%d, NEXT_MESSAGE_ID=%d",
        NEXT_CONVERSATION_ID,
        NEXT_MESSAGE_ID,
    )


def load_conversation_from_firestore(conv_id: int) -> Optional[ConversationSummary]:
    try:
        db = get_db()
    except Exception:
        logger.exception("Kunne ikke få Firestore-klient til at hente conversation")
        return None

    doc = db.collection("conversations").document(str(conv_id)).get()
    if not doc.exists:
        return None

    data = doc.to_dict() or {}
    return ConversationSummary(
        id=int(data.get("id", conv_id)),
        created_at=_parse_dt(data.get("created_at")),
        last_message_at=_parse_dt(data.get("last_message_at")),
        last_message_preview=str(data.get("last_message_preview", "")),
        is_read=bool(data.get("is_read", False)),
        status=str(data.get("status", "open")),
    )


def save_conversation_to_firestore(conv: ConversationSummary, client_conversation_id: Optional[str] = None) -> None:
    try:
        db = get_db()
    except Exception:
        logger.exception("Kunne ikke få Firestore-klient til at gemme conversation")
        return

    doc_ref = db.collection("conversations").document(str(conv.id))
    payload = {
        "id": conv.id,
        "created_at": conv.created_at,
        "last_message_at": conv.last_message_at,
        "last_message_preview": conv.last_message_preview,
        "is_read": conv.is_read,
        "status": conv.status,
    }
    if client_conversation_id:
        payload["client_conversation_id"] = client_conversation_id

    try:
        doc_ref.set(payload, merge=True)
    except Exception:
        logger.exception("Fejl ved gem conversation %s", conv.id)


def save_message_to_firestore(msg: MessageOut) -> None:
    try:
        db = get_db()
    except Exception:
        logger.exception("Kunne ikke få Firestore-klient til at gemme message")
        return

    conv_ref = db.collection("conversations").document(str(msg.conversation_id))
    msg_ref = conv_ref.collection("messages").document(str(msg.id))

    try:
        msg_ref.set(
            {
                "id": msg.id,
                "conversation_id": msg.conversation_id,
                "text": msg.text,
                "created_at": msg.created_at,
                "sender": msg.sender,
                "name": msg.name,
                "email": msg.email,
                "client_message_id": msg.client_message_id,
            }
        )
    except Exception:
        logger.exception("Fejl ved gem message %s", msg.id)


def create_conversation(initial_text: str, client_conversation_id: Optional[str] = None) -> ConversationSummary:
    global NEXT_CONVERSATION_ID
    init_counters_from_firestore()

    now = datetime.utcnow()
    conv = ConversationSummary(
        id=NEXT_CONVERSATION_ID,
        created_at=now,
        last_message_at=now,
        last_message_preview=initial_text[:120],
        is_read=False,
        status="open",
    )
    CONVERSATIONS[conv.id] = conv
    NEXT_CONVERSATION_ID += 1

    save_conversation_to_firestore(conv, client_conversation_id=client_conversation_id)
    if client_conversation_id:
        CLIENT_CONVERSATION_MAP[client_conversation_id] = conv.id

    return conv


def touch_conversation(conv_id: int, new_text: str, from_visitor: bool) -> ConversationSummary:
    now = datetime.utcnow()
    conv = CONVERSATIONS.get(conv_id)

    if conv is None:
        existing = load_conversation_from_firestore(conv_id)
        if existing is not None:
            conv = existing
        else:
            conv = ConversationSummary(
                id=conv_id,
                created_at=now,
                last_message_at=now,
                last_message_preview=new_text[:120],
                is_read=not from_visitor,
                status="open",
            )

    conv.last_message_at = now
    conv.last_message_preview = new_text[:120]
    conv.is_read = not from_visitor

    CONVERSATIONS[conv.id] = conv
    save_conversation_to_firestore(conv)

    return conv


@app.post("/messages", response_model=MessageOut, response_model_exclude_none=True)
def create_message(msg: MessageIn):
    """
    Modtag en ny besked fra websitet (besøgende).
    """
    global NEXT_MESSAGE_ID
    init_counters_from_firestore()

    # 1) Find/ opret conversation
    if msg.conversation_id is None:
        if msg.client_conversation_id:
            mapped = CLIENT_CONVERSATION_MAP.get(msg.client_conversation_id)
            if mapped:
                conv_id = mapped
                _ = touch_conversation(conv_id, msg.text, from_visitor=True)
            else:
                # Prøv Firestore lookup på client_conversation_id
                conv_id = None
                try:
                    db = get_db()
                    docs = (
                        db.collection("conversations")
                        .where("client_conversation_id", "==", msg.client_conversation_id)
                        .limit(1)
                        .stream()
                    )
                    doc = next(docs, None)
                    if doc is not None and doc.exists:
                        data = doc.to_dict() or {}
                        conv_id = int(data.get("id"))
                        CLIENT_CONVERSATION_MAP[msg.client_conversation_id] = conv_id
                        _ = touch_conversation(conv_id, msg.text, from_visitor=True)
                except Exception:
                    conv_id = None

                if conv_id is None:
                    conv = create_conversation(msg.text, client_conversation_id=msg.client_conversation_id)
                    conv_id = conv.id
        else:
            conv = create_conversation(msg.text)
            conv_id = conv.id
    else:
        conv_id = int(msg.conversation_id)
        _ = touch_conversation(conv_id, msg.text, from_visitor=True)

    # 2) Idempotency på client_message_id
    if msg.client_message_id:
        client_key = f"{conv_id}:{msg.client_message_id}"
        existing = MESSAGE_BY_CLIENT_KEY.get(client_key)
        if existing is None:
            # Firestore lookup best effort
            try:
                db = get_db()
                conv_ref = db.collection("conversations").document(str(conv_id))
                docs = (
                    conv_ref.collection("messages")
                    .where("client_message_id", "==", msg.client_message_id)
                    .limit(1)
                    .stream()
                )
                doc = next(docs, None)
                if doc is not None and doc.exists:
                    data = doc.to_dict() or {}
                    existing = MessageOut(
                        id=int(data.get("id")),
                        conversation_id=int(data.get("conversation_id")),
                        text=str(data.get("text", "")),
                        created_at=_parse_dt(data.get("created_at")),
                        sender=str(data.get("sender", "visitor")),
                        name=data.get("name"),
                        email=data.get("email"),
                        client_message_id=data.get("client_message_id"),
                    )
                    MESSAGE_BY_CLIENT_KEY[client_key] = existing
            except Exception:
                pass

        if existing is not None:
            return existing

    # 3) Opret message
    new_msg = MessageOut(
        id=NEXT_MESSAGE_ID,
        conversation_id=conv_id,
        created_at=datetime.utcnow(),
        text=msg.text,
        sender="visitor",
        name=msg.name,
        email=msg.email,
        client_message_id=msg.client_message_id,
    )
    MESSAGES.append(new_msg)
    NEXT_MESSAGE_ID += 1

    if msg.client_message_id:
        client_key = f"{conv_id}:{msg.client_message_id}"
        MESSAGE_BY_CLIENT_KEY[client_key] = new_msg
        SEEN_CLIENT_MESSAGE_IDS.add(client_key)

    save_message_to_firestore(new_msg)

    # Push
    try:
        preview = msg.text[:80]
        send_push_to_admins(
            title="Ny chatbesked",
            body=preview,
            data={"conversation_id": str(conv_id), "message_id": str(new_msg.id)},
        )
    except Exception:
        logger.exception("Kunne ikke sende FCM-push for ny besked")

    return new_msg


@app.get("/messages", response_model=List[MessageOut], response_model_exclude_none=True)
def list_messages(conversation_id: Optional[int] = None):
    """
    Hent beskeder (fra Firestore hvis muligt, ellers RAM fallback).
    """
    try:
        db = get_db()
    except Exception:
        logger.exception("Kunne ikke få Firestore-klient til at liste messages")
        if conversation_id is None:
            return list(MESSAGES)
        return [m for m in MESSAGES if m.conversation_id == conversation_id]

    if conversation_id is not None:
        conv_ref = db.collection("conversations").document(str(conversation_id))
        query = conv_ref.collection("messages").order_by("created_at")
        docs = query.stream()
    else:
        if gc_firestore is None:
            if conversation_id is None:
                return list(MESSAGES)
            return [m for m in MESSAGES if m.conversation_id == conversation_id]
        query = db.collection_group("messages").order_by("created_at")
        docs = query.stream()

    results: List[MessageOut] = []
    for doc in docs:
        data = doc.to_dict() or {}
        try:
            results.append(
                MessageOut(
                    id=int(data.get("id")),
                    conversation_id=int(data.get("conversation_id")),
                    text=str(data.get("text", "")),
                    created_at=_parse_dt(data.get("created_at")),
                    sender=str(data.get("sender", "visitor")),
                    name=data.get("name"),
                    email=data.get("email"),
                    client_message_id=data.get("client_message_id"),
                )
            )
        except Exception:
            logger.exception("Kunne ikke parse message-dokument %s", getattr(doc, "id", "?"))

    return results


@app.get("/conversations", response_model=List[ConversationSummary])
def list_conversations(_: None = Depends(require_admin)):
    """
    Liste over alle samtaler – læst fra Firestore (fallback: RAM).
    """
    try:
        db = get_db()
    except Exception:
        logger.exception("Kunne ikke få Firestore-klient til at liste conversations")
        convs = list(CONVERSATIONS.values())
    else:
        docs = db.collection("conversations").stream()
        convs = []
        for doc in docs:
            data = doc.to_dict() or {}
            try:
                conv = ConversationSummary(
                    id=int(data.get("id")),
                    created_at=_parse_dt(data.get("created_at")),
                    last_message_at=_parse_dt(data.get("last_message_at")),
                    last_message_preview=str(data.get("last_message_preview", "")),
                    is_read=bool(data.get("is_read", False)),
                    status=str(data.get("status", "open")),
                )
                convs.append(conv)
                CONVERSATIONS[conv.id] = conv
            except Exception:
                logger.exception("Kunne ikke parse conversation-dokument %s", getattr(doc, "id", "?"))

    return sorted(convs, key=lambda c: (c.is_read, -c.last_message_at.timestamp()))


@app.get("/conversations/{conversation_id}/messages", response_model=List[MessageOut], response_model_exclude_none=True)
def get_conversation_messages(conversation_id: int, _: None = Depends(require_admin)):
    """
    Hent alle beskeder for én samtale – læst fra Firestore (fallback: RAM).
    """
    try:
        db = get_db()
    except Exception:
        logger.exception("Kunne ikke få Firestore-klient til at hente conversation messages")
        return [m for m in MESSAGES if m.conversation_id == conversation_id]

    conv_ref = db.collection("conversations").document(str(conversation_id))
    query = conv_ref.collection("messages").order_by("created_at")
    docs = query.stream()

    results: List[MessageOut] = []
    for doc in docs:
        data = doc.to_dict() or {}
        try:
            results.append(
                MessageOut(
                    id=int(data.get("id")),
                    conversation_id=int(data.get("conversation_id")),
                    text=str(data.get("text", "")),
                    created_at=_parse_dt(data.get("created_at")),
                    sender=str(data.get("sender", "visitor")),
                    name=data.get("name"),
                    email=data.get("email"),
                    client_message_id=data.get("client_message_id"),
                )
            )
        except Exception:
            logger.exception("Kunne ikke parse message-dokument %s", getattr(doc, "id", "?"))

    return results


@app.post("/conversations/{conversation_id}/reply", response_model=MessageOut, response_model_exclude_none=True)
def reply_to_conversation(conversation_id: int, msg: MessageIn, _: None = Depends(require_admin)):
    """
    Svar fra agent i en given samtale.
    """
    global NEXT_MESSAGE_ID
    init_counters_from_firestore()

    # sørg for conv eksisterer
    conv = CONVERSATIONS.get(conversation_id)
    if conv is None:
        existing = load_conversation_from_firestore(conversation_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        CONVERSATIONS[existing.id] = existing

    _ = touch_conversation(conversation_id, msg.text, from_visitor=False)

    new_msg = MessageOut(
        id=NEXT_MESSAGE_ID,
        conversation_id=conversation_id,
        created_at=datetime.utcnow(),
        text=msg.text,
        sender="agent",
        name=msg.name,
        email=msg.email,
        client_message_id=msg.client_message_id,
    )
    MESSAGES.append(new_msg)
    NEXT_MESSAGE_ID += 1

    save_message_to_firestore(new_msg)
    return new_msg


@app.patch("/conversations/{conversation_id}/read", response_model=ConversationSummary)
def mark_conversation_read(conversation_id: int, _: None = Depends(require_admin)):
    conv = CONVERSATIONS.get(conversation_id)
    if conv is None:
        conv = load_conversation_from_firestore(conversation_id)
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found")

    conv.is_read = True
    CONVERSATIONS[conv.id] = conv
    save_conversation_to_firestore(conv)
    return conv


@app.patch("/conversations/{conversation_id}/status", response_model=ConversationSummary)
def update_conversation_status(conversation_id: int, status: str, _: None = Depends(require_admin)):
    if status not in {"open", "closed"}:
        raise HTTPException(status_code=400, detail="Invalid status")

    conv = CONVERSATIONS.get(conversation_id)
    if conv is None:
        conv = load_conversation_from_firestore(conversation_id)
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found")

    conv.status = status
    CONVERSATIONS[conv.id] = conv
    save_conversation_to_firestore(conv)
    return conv


@app.post("/admin/register-device")
def register_admin_device(payload: RegisterAdminDevice, _: None = Depends(require_admin)):
    token = (payload.token or "").strip()
    if not token:
        raise HTTPException(status_code=400, detail="Empty token")

    ADMIN_DEVICE_TOKENS.add(token)
    save_admin_device_token(token)

    logger.info("Registrerede admin-device token (nu %d tokens)", len(ADMIN_DEVICE_TOKENS))
    return {"ok": True}


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

    q = db.collection(BILKA_JOBS_COLLECTION)
    if gc_firestore is not None:
        q = q.order_by("created_at", direction=gc_firestore.Query.DESCENDING)
    else:
        q = q.order_by("created_at")  # fallback

    docs = q.limit(limit).stream()
    jobs = [{"id": d.id, **(d.to_dict() or {})} for d in docs]
    return {"jobs": jobs}


@app.post("/bilka/worker/poll")
def bilka_worker_poll(_: None = Depends(require_bilka_worker)):
    db = get_db()

    q = db.collection(BILKA_JOBS_COLLECTION).where("status", "==", "pending")
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
        db.collection(BILKA_JOBS_COLLECTION).document(id).set({"pdf_url": pdf_url}, merge=True)
    except Exception:
        logger.exception("Kunne ikke opdatere pdf_url i bilka_jobs/%s", id)

    return {"pdf_url": pdf_url}


@app.get("/bilka/jobs/{job_id}/pdf")
def bilka_job_pdf(job_id: str, _: None = Depends(require_bilka_admin)):
    path = UPLOAD_DIR / f"{job_id}.pdf"
    if not path.exists():
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(str(path), media_type="application/pdf", filename="mealplan.pdf")
