from datetime import datetime
from typing import List, Dict, Optional, Set
import os
import json
import logging

from fastapi import FastAPI, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from firebase_client import get_db

# Try import Firestore query helpers (til counters og collection_group)
try:
    from google.cloud import firestore as gc_firestore
except ImportError:  # afhænger af firebase-admin extras
    gc_firestore = None  # type: ignore

# Firebase / FCM:
try:
    import firebase_admin
    from firebase_admin import credentials, messaging

    FCM_AVAILABLE = True
except ImportError:  # hvis firebase_admin ikke er installeret
    firebase_admin = None  # type: ignore
    credentials = None  # type: ignore
    messaging = None  # type: ignore
    FCM_AVAILABLE = False

logger = logging.getLogger("dietz_chat_backend")
logging.basicConfig(level=logging.INFO)


def init_firebase_app():
    """
    Initialiser firebase_admin, hvis credentials er sat.

    - Hvis env `FIREBASE_CREDENTIALS_JSON` er sat -> brug den (JSON-indhold).
    - Ellers, hvis `GOOGLE_APPLICATION_CREDENTIALS` peger på en fil -> brug den.
    - Ellers: ingen FCM
    """
    if not FCM_AVAILABLE:
        logger.info("firebase_admin ikke tilgængelig; FCM er slået fra.")
        return None

    if firebase_admin._apps:
        # Allerede initialiseret
        return firebase_admin.get_app()

    cred = None

    # 1) Prøv via env-variabel med JSON
    raw_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
    if raw_json:
        try:
            data = json.loads(raw_json)
            cred = credentials.Certificate(data)
            logger.info("Firebase credentials indlæst fra FIREBASE_CREDENTIALS_JSON.")
        except Exception as e:
            logger.exception("Kunne ikke parse FIREBASE_CREDENTIALS_JSON: %s", e)

    # 2) Prøv via GOOGLE_APPLICATION_CREDENTIALS (sti til fil)
    if cred is None:
        path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if path and os.path.exists(path):
            try:
                cred = credentials.Certificate(path)
                logger.info(
                    "Firebase credentials indlæst fra GOOGLE_APPLICATION_CREDENTIALS=%s",
                    path,
                )
            except Exception as e:
                logger.exception(
                    "Kunne ikke indlæse credentials fra fil %s: %s", path, e
                )

    try:
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

# Gemmer FCM tokens for admin-appen (midlertidigt i RAM)
ADMIN_DEVICE_TOKENS: Set[str] = set()

# Admin tokens gemmes i Firestore i collection "admin_devices"
ADMIN_TOKENS_INITIALIZED = False
ADMIN_TOKENS_COLLECTION = "admin_devices"


def save_admin_device_token(token: str) -> None:
    """
    Gem / opdater et admin-device token i Firestore.
    """
    try:
        db = get_db()
    except Exception:
        logger.exception("Kunne ikke gemme admin-device token i Firestore")
        return

    # Brug token som dokument-id, så du undgår dubletter
    doc_ref = db.collection(ADMIN_TOKENS_COLLECTION).document(token)
    try:
        doc_ref.set(
            {
                "token": token,
                "updated_at": datetime.utcnow(),
            },
            merge=True,  # så vi kan opdatere uden at slette andre felter senere
        )
    except Exception:
        logger.exception("Fejl ved skriv til admin_devices i Firestore")


def load_admin_device_tokens() -> None:
    """
    Læs alle admin-device tokens fra Firestore ind i ADMIN_DEVICE_TOKENS.
    """
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
    """
    Sørger for, at ADMIN_DEVICE_TOKENS er indlæst fra Firestore én gang.
    """
    global ADMIN_TOKENS_INITIALIZED
    if ADMIN_TOKENS_INITIALIZED:
        return

    ADMIN_TOKENS_INITIALIZED = True
    load_admin_device_tokens()


def send_push_to_admins(
    title: str,
    body: str,
    data: Optional[Dict[str, str]] = None,
) -> None:
    """
    Send en push til alle registrerede admin-devices.

    Hvis FCM ikke er konfigureret, logger vi bare hvad vi ville sende.
    Vi sender én besked pr. token (kompatibelt med ældre firebase-admin).
    """
    if not FCM_AVAILABLE or FIREBASE_APP is None:
        logger.info(
            "FCM ikke konfigureret; ville have sendt push: %r / %r med data=%r",
            title,
            body,
            data,
        )
        return

    # Sørg for at tokens er indlæst fra Firestore (kun én gang per proces)
    ensure_admin_tokens_loaded()

    tokens = list(ADMIN_DEVICE_TOKENS)
    if not tokens:
        logger.info("Ingen ADMIN_DEVICE_TOKENS registreret; ingen push sendt.")
        return

    successes = 0
    failures = 0

    for t in tokens:
        try:
            msg = messaging.Message(
                token=t,
                notification=messaging.Notification(title=title, body=body),
                data=data or {},
            )
            response = messaging.send(msg)
            logger.info("Sendte push til %s, response=%s", t, response)
            successes += 1
        except Exception:
            logger.exception("Fejl ved send push til token=%s", t)
            failures += 1

    logger.info(
        "send_push_to_admins: %d succes, %d fejl (ud af %d tokens)",
        successes,
        failures,
        len(tokens),
    )


# --- FastAPI app / CORS ---

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",  # Juster evt. til din hjemmeside, f.eks. "https://dietzcc.dk"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Models ---


class MessageIn(BaseModel):
    text: str = Field(..., min_length=1, description="Selve beskeden fra brugeren")
    conversation_id: Optional[int] = Field(
        None,
        description=(
            "Hvis sat, tilføjes beskeden til en eksisterende samtale. "
            "Hvis tom, oprettes en ny samtale."
        ),
    )
    name: Optional[str] = Field(
        None,
        description="Valgfrit navn – kun hvis du senere vil spørge om det",
    )
    email: Optional[str] = Field(
        None,
        description="Valgfri e-mail – fx hvis du senere vil kunne følge op på mail",
    )


class MessageOut(BaseModel):
    id: int
    conversation_id: int
    text: str
    created_at: datetime
    sender: str  # "visitor" eller "agent"
    name: Optional[str] = None
    email: Optional[str] = None


class ConversationSummary(BaseModel):
    id: int
    created_at: datetime
    last_message_at: datetime
    last_message_preview: str
    is_read: bool
    status: str  # "open" / "closed"


class RegisterAdminDevice(BaseModel):
    token: str = Field(..., description="FCM registration token for admin-appen")


# --- In-memory "database" (backup hvis Firestore er nede) ---


MESSAGES: List[MessageOut] = []
CONVERSATIONS: Dict[int, ConversationSummary] = {}
NEXT_MESSAGE_ID = 1
NEXT_CONVERSATION_ID = 1
COUNTERS_INITIALIZED = False


def init_counters_from_firestore() -> None:
    """
    Læs max conversation id og max message id fra Firestore én gang,
    så vi kan fortsætte tælling på tværs af reboots.
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

    # Læs alle conversations
    try:
        conv_docs = db.collection("conversations").stream()
        for doc in conv_docs:
            data = doc.to_dict() or {}
            if "id" in data:
                cid = int(data["id"])
                max_conv_id = max(max_conv_id, cid)
    except Exception:
        logger.exception("Fejl ved læsning af max conversation id fra Firestore")

    # Læs alle messages via collection group, hvis muligt
    if gc_firestore is not None:
        try:
            msg_docs = db.collection_group("messages").stream()
            for doc in msg_docs:
                data = doc.to_dict() or {}
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


def save_conversation_to_firestore(conv: ConversationSummary) -> None:
    """
    Gem/overskriv en ConversationSummary i Firestore.
    """
    try:
        db = get_db()
    except Exception:
        logger.exception("Kunne ikke få Firestore-klient til at gemme conversation")
        return

    doc_ref = db.collection("conversations").document(str(conv.id))
    try:
        doc_ref.set(
            {
                "id": conv.id,
                "created_at": conv.created_at,
                "last_message_at": conv.last_message_at,
                "last_message_preview": conv.last_message_preview,
                "is_read": conv.is_read,
                "status": conv.status,
            }
        )
    except Exception:
        logger.exception("Fejl ved skriv af conversation til Firestore")


def load_conversation_from_firestore(conv_id: int) -> Optional[ConversationSummary]:
    """
    Hent én conversation fra Firestore ud fra id.
    """
    try:
        db = get_db()
    except Exception:
        logger.exception("Kunne ikke få Firestore-klient til at hente conversation")
        return None

    doc_ref = db.collection("conversations").document(str(conv_id))
    try:
        doc = doc_ref.get()
    except Exception:
        logger.exception("Fejl ved læsning af conversation %s fra Firestore", conv_id)
        return None

    if not doc.exists:
        return None

    data = doc.to_dict() or {}
    try:
        return ConversationSummary(
            id=int(data["id"]),
            created_at=data["created_at"],
            last_message_at=data["last_message_at"],
            last_message_preview=data["last_message_preview"],
            is_read=bool(data.get("is_read", False)),
            status=data.get("status", "open"),
        )
    except Exception:
        logger.exception("Kunne ikke parse conversation-doc %s", conv_id)
        return None


def save_message_to_firestore(msg: MessageOut) -> None:
    """
    Gem én besked i Firestore under conversations/{id}/messages/{msg.id}
    """
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
            }
        )
    except Exception:
        logger.exception(
            "Fejl ved skriv af message %s (conversation %s) til Firestore",
            msg.id,
            msg.conversation_id,
        )


def create_conversation(first_message_text: str) -> ConversationSummary:
    """
    Opret en ny samtale (både i RAM og i Firestore).
    """
    global NEXT_CONVERSATION_ID

    init_counters_from_firestore()

    conv_id = NEXT_CONVERSATION_ID
    NEXT_CONVERSATION_ID += 1

    now = datetime.utcnow()
    conv = ConversationSummary(
        id=conv_id,
        created_at=now,
        last_message_at=now,
        last_message_preview=first_message_text[:120],
        is_read=False,
        status="open",
    )
    CONVERSATIONS[conv_id] = conv

    save_conversation_to_firestore(conv)
    return conv


def touch_conversation(
    conv_id: int, last_text: str, from_visitor: bool = False
) -> ConversationSummary:
    """
    Opdater last_message_at / last_message_preview / is_read for en conversation.

    Hvis conversation ikke fandtes i RAM, oprettes en "stub", som så kan overskrives
    når load_conversation_from_firestore senere kaldes.
    """
    now = datetime.utcnow()

    if conv_id in CONVERSATIONS:
        conv = CONVERSATIONS[conv_id]
    else:
        conv = ConversationSummary(
            id=conv_id,
            created_at=now,
            last_message_at=now,
            last_message_preview=last_text[:120],
            is_read=False,
            status="open",
        )

    conv.last_message_at = now
    conv.last_message_preview = last_text[:120]
    if from_visitor:
        conv.is_read = False

    CONVERSATIONS[conv_id] = conv
    save_conversation_to_firestore(conv)
    return conv


# --- Endpoints ---


@app.post(
    "/messages",
    response_model=MessageOut,
    response_model_exclude_none=True,
)
def create_message(msg: MessageIn):
    """
    Modtag en ny besked fra websitet (besøgende).

    - Hvis conversation_id er None  -> opret en ny samtale.
    - Hvis conversation_id er sat   -> brug den eksisterende samtale (også efter restart).
    """
    global NEXT_MESSAGE_ID

    init_counters_from_firestore()

    # 1) Find eller opret samtale
    if msg.conversation_id is None:
        conv = create_conversation(msg.text)
        conv_id = conv.id
    else:
        conv_id = msg.conversation_id
        if conv_id in CONVERSATIONS:
            conv = touch_conversation(conv_id, msg.text, from_visitor=True)
        else:
            # Prøv at hente fra Firestore, hvis API er blevet genstartet
            existing = load_conversation_from_firestore(conv_id)
            if existing is not None:
                CONVERSATIONS[conv_id] = existing
                conv = touch_conversation(conv_id, msg.text, from_visitor=True)
            else:
                # Hvis den slet ikke findes, opret en ny samtale med det id
                now = datetime.utcnow()
                conv = ConversationSummary(
                    id=conv_id,
                    created_at=now,
                    last_message_at=now,
                    last_message_preview=msg.text[:120],
                    is_read=False,
                    status="open",
                )
                CONVERSATIONS[conv_id] = conv
                save_conversation_to_firestore(conv)

    # 2) Opret selve beskeden
    message_id = NEXT_MESSAGE_ID
    NEXT_MESSAGE_ID += 1

    now = datetime.utcnow()
    out = MessageOut(
        id=message_id,
        conversation_id=conv_id,
        text=msg.text,
        created_at=now,
        sender="visitor",
        name=msg.name,
        email=msg.email,
    )

    MESSAGES.append(out)
    save_message_to_firestore(out)

    # 3) Opdater conversation (last_message_*)
    conv = touch_conversation(conv_id, msg.text, from_visitor=True)

    # 4) Send push til admin (best effort)
    send_push_to_admins(
        title="Ny besked på dietzcc.dk",
        body=msg.text[:120],
        data={"conversation_id": str(conv_id)},
    )

    return out


@app.get(
    "/messages",
    response_model=List[MessageOut],
    response_model_exclude_none=True,
)
def list_messages(conversation_id: Optional[int] = None):
    """
    Hent beskeder.

    - Uden parameter: alle beskeder (debug) fra Firestore.
    - Med ?conversation_id=123: kun beskeder for den samtale.
    """
    try:
        db = get_db()
    except Exception:
        logger.exception("Kunne ikke få Firestore-klient til at liste messages")
        # Fald tilbage til in-memory (dev / fejl)
        if conversation_id is None:
            return list(MESSAGES)
        return [m for m in MESSAGES if m.conversation_id == conversation_id]

    if conversation_id is not None:
        conv_ref = db.collection("conversations").document(str(conversation_id))
        query = conv_ref.collection("messages").order_by("created_at")
        docs = query.stream()
    else:
        # Bruger collection group til at hente alle beskeder
        if gc_firestore is None:
            # Fald tilbage til RAM
            return list(MESSAGES)
        query = db.collection_group("messages").order_by("created_at")
        docs = query.stream()

    results: List[MessageOut] = []
    for doc in docs:
        data = doc.to_dict() or {}
        try:
            results.append(
                MessageOut(
                    id=int(data["id"]),
                    conversation_id=int(data["conversation_id"]),
                    text=data["text"],
                    created_at=data["created_at"],
                    sender=data["sender"],
                    name=data.get("name"),
                    email=data.get("email"),
                )
            )
        except Exception:
            logger.exception("Kunne ikke parse message-dokument %s", doc.id)

    return results


@app.get("/conversations", response_model=List[ConversationSummary])
def list_conversations(_: None = Depends(require_admin)):
    """
    Liste over alle samtaler – læst fra Firestore.
    """
    try:
        db = get_db()
    except Exception:
        logger.exception("Kunne ikke få Firestore-klient til at liste conversations")
        # Fald tilbage til in-memory
        return list(CONVERSATIONS.values())

    try:
        docs = db.collection("conversations").stream()
    except Exception:
        logger.exception("Fejl ved hentning af conversations fra Firestore")
        return list(CONVERSATIONS.values())

    convs: List[ConversationSummary] = []
    for doc in docs:
        data = doc.to_dict() or {}
        try:
            convs.append(
                ConversationSummary(
                    id=int(data["id"]),
                    created_at=data["created_at"],
                    last_message_at=data["last_message_at"],
                    last_message_preview=data["last_message_preview"],
                    is_read=bool(data.get("is_read", False)),
                    status=data.get("status", "open"),
                )
            )
        except Exception:
            logger.exception("Kunne ikke parse conversation-dokument %s", doc.id)

    # Returnér først ulæste, dernæst efter seneste aktivitet
    return sorted(
        convs,
        key=lambda c: (c.is_read, -c.last_message_at.timestamp()),
    )


@app.get(
    "/conversations/{conversation_id}/messages",
    response_model=List[MessageOut],
    response_model_exclude_none=True,
)
def get_conversation_messages(
    conversation_id: int,
    _: None = Depends(require_admin),
):
    """
    Hent alle beskeder for én conversation.
    """
    try:
        db = get_db()
    except Exception:
        logger.exception("Kunne ikke få Firestore-klient til at hente messages")
        # Fald tilbage til RAM
        return [m for m in MESSAGES if m.conversation_id == conversation_id]

    conv_ref = db.collection("conversations").document(str(conversation_id))
    try:
        docs = conv_ref.collection("messages").order_by("created_at").stream()
    except Exception:
        logger.exception("Fejl ved hentning af messages for conversation %s", conversation_id)
        return [m for m in MESSAGES if m.conversation_id == conversation_id]

    results: List[MessageOut] = []
    for doc in docs:
        data = doc.to_dict() or {}
        try:
            results.append(
                MessageOut(
                    id=int(data["id"]),
                    conversation_id=int(data["conversation_id"]),
                    text=data["text"],
                    created_at=data["created_at"],
                    sender=data["sender"],
                    name=data.get("name"),
                    email=data.get("email"),
                )
            )
        except Exception:
            logger.exception("Kunne ikke parse message-doc %s", doc.id)

    return results


# --- Admin auth placeholder (for f.eks. simpel header) ---


def require_admin(x_admin_token: str = Header(..., alias="X-Admin-Token")):
    """
    Simpel beskyttelse: kræver at der sendes en header X-Admin-Token
    som matcher en hemmelig værdi i env ADMIN_API_TOKEN.
    """
    expected = os.getenv("ADMIN_API_TOKEN")
    if not expected:
        # Hvis der ikke er sat nogen token, tillader vi alt (dev-mode).
        return None

    if x_admin_token != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

    return None


# --- Ekstra endpoint til at markere conversation som læst/ulæst og ændre status ---


class UpdateConversationStatus(BaseModel):
    is_read: Optional[bool] = None
    status: Optional[str] = None  # "open" / "closed"


@app.post(
    "/conversations/{conversation_id}/status",
    response_model=ConversationSummary,
    response_model_exclude_none=True,
)
def update_conversation_status(
    conversation_id: int,
    payload: UpdateConversationStatus,
    _: None = Depends(require_admin),
):
    """
    Opdater status/is_read på en conversation.
    """
    # Forsøg først i RAM
    if conversation_id in CONVERSATIONS:
        conv = CONVERSATIONS[conversation_id]
    else:
        conv = load_conversation_from_firestore(conversation_id)
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        CONVERSATIONS[conversation_id] = conv

    if payload.is_read is not None:
        conv.is_read = payload.is_read

    if payload.status is not None:
        if payload.status not in ("open", "closed"):
            raise HTTPException(status_code=400, detail="Invalid status")
        conv.status = payload.status

    CONVERSATIONS[conv.id] = conv
    save_conversation_to_firestore(conv)

    return conv


# --- Ny route: registrer admin-device til push ---


@app.post("/admin/register-device")
def register_admin_device(
    payload: RegisterAdminDevice,
    _: None = Depends(require_admin),
):
    """
    Modtag og gem FCM-token for admin-appen.
    Kaldt fra Android-appen, så backend ved hvem der skal have push.
    """
    token = payload.token.strip()
    if not token:
        raise HTTPException(status_code=400, detail="Empty token")

    # Opdater in-memory sæt
    ADMIN_DEVICE_TOKENS.add(token)

    # Gem også i Firestore
    save_admin_device_token(token)
    logger.info(
        "Registrerede admin-device token (nu %d tokens)", len(ADMIN_DEVICE_TOKENS)
    )
    return {"ok": True}
