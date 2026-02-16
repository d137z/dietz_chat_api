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
    """
    Initialiser firebase_admin, hvis credentials er sat.

    - Hvis env `FIREBASE_CREDENTIALS_JSON` er sat -> brug den (JSON-indhold).
    - Ellers, hvis `GOOGLE_APPLICATION_CREDENTIALS` peger på en fil -> brug den.
    - Ellers: ingen FCM (vi logger bare og kører videre uden push).
    """
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
            message = messaging.Message(
                notification=messaging.Notification(
                    title=title,
                    body=body,
                ),
                data=data or {},
                token=t,
            )
            messaging.send(message)
            successes += 1
        except Exception as e:
            failures += 1
            logger.exception("Fejl ved send af FCM-notifikation til token %s: %s", t, e)

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
        "https://www.dietzcc.dk",
        "https://dietzcc.dk",
        "http://localhost:5173",  # til lokal test
        "https://prismatic-marzipan-6f51be.netlify.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Health / ping ---


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ping")
def ping():
    return {"message": "pong"}


# --- Modeller ---


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

    # Idempotency: gør det muligt at retry'e sikkert under Render cold-start.
    # - client_conversation_id bruges når conversation_id endnu ikke er kendt (første besked)
    # - client_message_id bruges til at undgå dubletter ved retry
    client_conversation_id: Optional[str] = Field(
        None,
        description=(
            "Stabil klient-id for samtalen (UUID). Bruges når conversation_id er None, "
            "så gentagne POSTs ikke opretter flere samtaler."
        ),
    )
    client_message_id: Optional[str] = Field(
        None,
        description=(
            "Stabil klient-id for beskeden (UUID). Bruges til at undgå dubletter ved retry."
        ),
    )


class MessageOut(BaseModel):
    id: int
    conversation_id: int
    text: str
    created_at: datetime
    sender: str  # "visitor" eller "agent"
    name: Optional[str] = None
    email: Optional[str] = None
    client_message_id: Optional[str] = None


class ConversationSummary(BaseModel):
    id: int
    created_at: datetime
    last_message_at: datetime
    last_message_preview: str
    is_read: bool
    status: str  # "open" / "closed"


class RegisterAdminDevice(BaseModel):
    token: str = Field(..., description="FCM registration token for admin-appen")


# --- In-memory "database" (cache) ---


MESSAGES: List[MessageOut] = []
CONVERSATIONS: Dict[int, ConversationSummary] = {}
NEXT_MESSAGE_ID = 1
NEXT_CONVERSATION_ID = 1
COUNTERS_INITIALIZED = False

# Idempotency caches (RAM-only fallback hvis Firestore ikke er tilgængelig)
# - Map fra client_conversation_id -> conversation_id
CLIENT_CONVERSATION_MAP: Dict[str, int] = {}
# - Set af (conversation_id, client_message_id) som allerede er oprettet
SEEN_CLIENT_MESSAGE_IDS: Set[str] = set()
# - Hurtig lookup til eksisterende MessageOut ved idempotent retry
MESSAGE_BY_CLIENT_KEY: Dict[str, MessageOut] = {}


# --- Simpel admin-auth ---


ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")


def require_admin(x_admin_token: Optional[str] = Header(default=None)):
    """
    Simpel admin-beskyttelse:
    - Sæt ADMIN_TOKEN som environment variable på Render.
    - Send headeren: X-Admin-Token: <samme værdi> fra din app.
    Hvis ADMIN_TOKEN ikke er sat, tillader vi alle (dev-mode).
    """
    if ADMIN_TOKEN is None:
        return
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


# --- Hjælpefunktioner ---


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

    # Find højeste conversation-id
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

    # Find højeste message-id via collection group
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
    """
    Hent én conversation fra Firestore ud fra id.
    """
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
        created_at=data.get("created_at", datetime.utcnow()),
        last_message_at=data.get("last_message_at", datetime.utcnow()),
        last_message_preview=data.get("last_message_preview", ""),
        is_read=bool(data.get("is_read", False)),
        status=data.get("status", "open"),
    )


def save_conversation_to_firestore(conv: ConversationSummary, client_conversation_id: Optional[str] = None) -> None:
    """Gem samtalens metadata i Firestore (best effort)."""
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

    doc_ref.set(payload)


def save_message_to_firestore(msg: MessageOut) -> None:
    """Gem en enkelt besked i Firestore under samtalens messages-subcollection."""
    try:
        db = get_db()
    except Exception:
        logger.exception("Kunne ikke få Firestore-klient til at gemme message")
        return

    conv_ref = db.collection("conversations").document(str(msg.conversation_id))
    msg_ref = conv_ref.collection("messages").document(str(msg.id))

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


def clear_all_chat_data() -> Dict[str, int]:
    """Slet alle samtaler og beskeder i Firestore og ryd in-memory cache."""
    global MESSAGES, CONVERSATIONS, NEXT_MESSAGE_ID, NEXT_CONVERSATION_ID, COUNTERS_INITIALIZED

    deleted_conversations = 0
    deleted_messages = 0

    try:
        db = get_db()
    except Exception:
        logger.exception("Kunne ikke få Firestore-klient til clear_all_chat_data")
        raise HTTPException(status_code=500, detail="Kunne ikke forbinde til Firestore")

    try:
        convs_ref = db.collection("conversations")
        docs = convs_ref.stream()
        for doc in docs:
            conv_ref = convs_ref.document(doc.id)
            try:
                # Slet alle messages i subcollection'en
                msg_stream = conv_ref.collection("messages").stream()
                for msg_doc in msg_stream:
                    msg_doc.reference.delete()
                    deleted_messages += 1
            except Exception:
                logger.exception("Fejl ved sletning af messages for conversation %s", doc.id)

            # Slet selve conversation-dokumentet
            try:
                conv_ref.delete()
                deleted_conversations += 1
            except Exception:
                logger.exception("Fejl ved sletning af conversation %s", doc.id)
    except Exception:
        logger.exception("Fejl ved clear_all_chat_data")
        raise HTTPException(status_code=500, detail="Fejl ved sletning af Firestore-data")

    # Ryd in-memory cache og counters
    MESSAGES = []
    CONVERSATIONS = {}
    NEXT_MESSAGE_ID = 1
    NEXT_CONVERSATION_ID = 1
    COUNTERS_INITIALIZED = False

    logger.info(
        "Clear-all gennemført: %d conversations og %d messages slettet",
        deleted_conversations,
        deleted_messages,
    )

    return {
        "deleted_conversations": deleted_conversations,
        "deleted_messages": deleted_messages,
    }



def create_conversation(initial_text: str, client_conversation_id: Optional[str] = None) -> ConversationSummary:
    """Opret en helt ny samtale med første besked-tekst som preview."""
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

    # Gem også i Firestore (best effort)
    save_conversation_to_firestore(conv, client_conversation_id=client_conversation_id)

    # Hold RAM-map i sync, så vi kan dedupe uden Firestore
    if client_conversation_id:
        CLIENT_CONVERSATION_MAP[client_conversation_id] = conv.id

    return conv


def touch_conversation(
    conv_id: int,
    new_text: str,
    from_visitor: bool,
) -> ConversationSummary:
    """
    Opdater metadata for en eksisterende samtale.
    from_visitor=True  -> markér som ulæst (ny besked fra kunden)
    from_visitor=False -> markér som læst (du har svaret)
    """
    now = datetime.utcnow()
    conv = CONVERSATIONS.get(conv_id)

    if conv is None:
        # Hvis der ikke ligger noget i RAM, prøv Firestore
        existing = load_conversation_from_firestore(conv_id)
        if existing is not None:
            conv = existing
        else:
            # Opret en ny samtale hvis helt ukendt id
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
    # Ny besked fra besøgende -> ulæst
    # Svar fra dig           -> læst
    conv.is_read = not from_visitor

    CONVERSATIONS[conv.id] = conv

    # Gem opdateret samtale i Firestore (best effort)
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
        # Hvis client_conversation_id er sat, kan vi genbruge samme samtale ved retry
        if msg.client_conversation_id:
            # Først: RAM-cache
            mapped = CLIENT_CONVERSATION_MAP.get(msg.client_conversation_id)
            if mapped:
                conv_id = mapped
                conv = touch_conversation(conv_id, msg.text, from_visitor=True)
            else:
                # Prøv Firestore lookup (best effort)
                conv = None
                conv_id = None  # type: ignore
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
                        conv = touch_conversation(conv_id, msg.text, from_visitor=True)
                except Exception:
                    # Hvis Firestore fejler, falder vi tilbage til ny samtale
                    conv = None

                if conv is None:
                    conv = create_conversation(msg.text, client_conversation_id=msg.client_conversation_id)
                    conv_id = conv.id
        else:
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
                CONVERSATIONS[existing.id] = existing
                conv = touch_conversation(existing.id, msg.text, from_visitor=True)
                conv_id = conv.id
            else:
                # Hvis conversation-id virkelig ikke findes, start ny
                conv = create_conversation(msg.text)
                conv_id = conv.id

    # 2) Idempotency: undgå at oprette samme besked flere gange ved retry
    if msg.client_message_id:
        client_key = f"{conv_id}:{msg.client_message_id}"

        # RAM-cache først
        existing = MESSAGE_BY_CLIENT_KEY.get(client_key)
        if existing is None:
            # Prøv Firestore lookup (best effort)
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
                        text=data.get("text", ""),
                        created_at=data.get("created_at", datetime.utcnow()),
                        sender=data.get("sender", "visitor"),
                        name=data.get("name"),
                        email=data.get("email"),
                        client_message_id=data.get("client_message_id"),
                    )
                    MESSAGE_BY_CLIENT_KEY[client_key] = existing
                    SEEN_CLIENT_MESSAGE_IDS.add(client_key)
            except Exception:
                # Ignorér Firestore fejl; vi fortsætter og opretter beskeden (RAM-dedup hjælper delvist)
                pass

        if existing is not None:
            return existing

    # 3) Opret selve beskeden
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

    # Gem beskeden i Firestore (best effort)
    save_message_to_firestore(new_msg)

    # 3) Send push til registrerede admin-devices (hvis FCM er sat op)
    try:
        preview = msg.text[:80]
        send_push_to_admins(
            title="Ny chatbesked",
            body=preview,
            data={
                "conversation_id": str(conv_id),
                "message_id": str(new_msg.id),
            },
        )
    except Exception:
        logger.exception("Kunne ikke sende FCM-push for ny besked")

    return new_msg


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
                    text=data.get("text", ""),
                    created_at=data.get("created_at", datetime.utcnow()),
                    sender=data.get("sender", "visitor"),
                    name=data.get("name"),
                    email=data.get("email"),
                    client_message_id=data.get("client_message_id"),
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
        # Fald tilbage til hvad vi har i RAM
        convs = list(CONVERSATIONS.values())
    else:
        docs = db.collection("conversations").stream()
        convs: List[ConversationSummary] = []
        for doc in docs:
            data = doc.to_dict() or {}
            try:
                conv = ConversationSummary(
                    id=int(data.get("id")),
                    created_at=data.get("created_at", datetime.utcnow()),
                    last_message_at=data.get("last_message_at", datetime.utcnow()),
                    last_message_preview=data.get("last_message_preview", ""),
                    is_read=bool(data.get("is_read", False)),
                    status=data.get("status", "open"),
                )
                convs.append(conv)
                # Hold RAM-cache nogenlunde i sync
                CONVERSATIONS[conv.id] = conv
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
    Hent alle beskeder for én specifik samtale – læst fra Firestore.
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
                    text=data.get("text", ""),
                    created_at=data.get("created_at", datetime.utcnow()),
                    sender=data.get("sender", "visitor"),
                    name=data.get("name"),
                    email=data.get("email"),
                    client_message_id=data.get("client_message_id"),
                )
            )
        except Exception:
            logger.exception("Kunne ikke parse message-dokument %s", doc.id)

    return results


@app.post(
    "/conversations/{conversation_id}/reply",
    response_model=MessageOut,
    response_model_exclude_none=True,
)
def reply_to_conversation(
    conversation_id: int,
    msg: MessageIn,
    _: None = Depends(require_admin),
):
    """
    Svar fra dig (agent) i en given samtale.
    Bruger samme MessageIn-model, men ignorerer evt. conversation_id i payload.
    """
    global NEXT_MESSAGE_ID

    init_counters_from_firestore()

    # Sørg for at samtalen findes
    conv = CONVERSATIONS.get(conversation_id)
    if conv is None:
        existing = load_conversation_from_firestore(conversation_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        CONVERSATIONS[existing.id] = existing
        conv = existing

    conv = touch_conversation(conversation_id, msg.text, from_visitor=False)

    new_msg = MessageOut(
        id=NEXT_MESSAGE_ID,
        conversation_id=conv.id,
        created_at=datetime.utcnow(),
        text=msg.text,
        sender="agent",
        name=msg.name,
        email=msg.email,
        client_message_id=msg.client_message_id,
    )
    MESSAGES.append(new_msg)
    NEXT_MESSAGE_ID += 1

    # Gem beskeden i Firestore (best effort)
    save_message_to_firestore(new_msg)

    return new_msg


@app.patch("/conversations/{conversation_id}/read", response_model=ConversationSummary)
def mark_conversation_read(
    conversation_id: int,
    _: None = Depends(require_admin),
):
    """
    Markér en samtale som læst (fx når du har åbnet den i appen).
    """
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
def update_conversation_status(
    conversation_id: int,
    status: str,
    _: None = Depends(require_admin),
):
    """
    Opdatér status på en samtale, fx 'open' eller 'closed'.
    """
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




@app.post("/admin/clear-all")
def admin_clear_all(
    _: None = Depends(require_admin),
):
    """Slet alle samtaler og beskeder i Firestore og ryd RAM-cache."""
    result = clear_all_chat_data()
    return {"ok": True, **result}


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
