from datetime import datetime
from typing import List, Dict, Optional, Set
import os
import json
import logging

from fastapi import FastAPI, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

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


def send_push_to_admins(
    title: str,
    body: str,
    data: Optional[Dict[str, str]] = None,
) -> None:
    """
    Send en push til alle registrerede admin-devices.

    Hvis FCM ikke er konfigureret, logger vi bare hvad vi ville sende.
    """
    if not FCM_AVAILABLE or FIREBASE_APP is None:
        logger.info(
            "FCM ikke konfigureret; ville have sendt push: %r / %r med data=%r",
            title,
            body,
            data,
        )
        return

    if not ADMIN_DEVICE_TOKENS:
        logger.info("Ingen admin device tokens registreret; springer push over.")
        return

    try:
        message = messaging.MulticastMessage(
            notification=messaging.Notification(
                title=title,
                body=body,
            ),
            data=data or {},
            tokens=list(ADMIN_DEVICE_TOKENS),
        )
        response = messaging.send_multicast(message)
        logger.info(
            "Sendte FCM multicast til %d tokens (success=%d, failure=%d)",
            len(ADMIN_DEVICE_TOKENS),
            response.success_count,
            response.failure_count,
        )
    except Exception as e:
        logger.exception("Fejl ved send af FCM-notifikationer: %s", e)


# --- FastAPI app / CORS ---

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.dietzcc.dk",
        "https://dietzcc.dk",
        "http://localhost:5173",  # til lokal test
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


# --- In-memory "database" ---


MESSAGES: List[MessageOut] = []
CONVERSATIONS: Dict[int, ConversationSummary] = {}
NEXT_MESSAGE_ID = 1
NEXT_CONVERSATION_ID = 1


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


def create_conversation(initial_text: str) -> ConversationSummary:
    """Opret en helt ny samtale med første besked-tekst som preview."""
    global NEXT_CONVERSATION_ID

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
        # Hvis der kommer et ukendt conversation_id, opretter vi en ny.
        conv = ConversationSummary(
            id=conv_id,
            created_at=now,
            last_message_at=now,
            last_message_preview=new_text[:120],
            is_read=not from_visitor,
            status="open",
        )
        CONVERSATIONS[conv.id] = conv
    else:
        conv.last_message_at = now
        conv.last_message_preview = new_text[:120]
        # Ny besked fra besøgende -> ulæst
        # Svar fra dig         -> læst
        conv.is_read = not from_visitor

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
    - Hvis conversation_id er sat   -> brug den eksisterende samtale.
    """
    global NEXT_MESSAGE_ID

    # 1) Find eller opret samtale
    if msg.conversation_id is None:
        conv = create_conversation(msg.text)
        conv_id = conv.id
    else:
        if msg.conversation_id in CONVERSATIONS:
            conv = touch_conversation(msg.conversation_id, msg.text, from_visitor=True)
            conv_id = conv.id
        else:
            conv = create_conversation(msg.text)
            conv_id = conv.id

    # 2) Opret selve beskeden
    new_msg = MessageOut(
        id=NEXT_MESSAGE_ID,
        conversation_id=conv_id,
        created_at=datetime.utcnow(),
        text=msg.text,
        sender="visitor",
        name=msg.name,
        email=msg.email,
    )
    MESSAGES.append(new_msg)
    NEXT_MESSAGE_ID += 1

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

    # Sørg for at samtalen findes / opdateres
    conv = touch_conversation(conversation_id, msg.text, from_visitor=False)

    new_msg = MessageOut(
        id=NEXT_MESSAGE_ID,
        conversation_id=conv.id,
        created_at=datetime.utcnow(),
        text=msg.text,
        sender="agent",
        name=msg.name,
        email=msg.email,
    )
    MESSAGES.append(new_msg)
    NEXT_MESSAGE_ID += 1

    return new_msg


@app.get(
    "/messages",
    response_model=List[MessageOut],
    response_model_exclude_none=True,
)
def list_messages(conversation_id: Optional[int] = None):
    """
    Hent beskeder.

    - Uden parameter: alle beskeder (debug).
    - Med ?conversation_id=123: kun beskeder for den samtale.
    """
    if conversation_id is None:
        return list(MESSAGES)
    return [m for m in MESSAGES if m.conversation_id == conversation_id]


@app.get("/conversations", response_model=List[ConversationSummary])
def list_conversations(_: None = Depends(require_admin)):
    """
    Liste over alle samtaler – kan bruges i din app til at vise
    "hvem har skrevet ind".
    """
    # Returnér først ulæste, dernæst efter seneste aktivitet
    return sorted(
        CONVERSATIONS.values(),
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
    Hent alle beskeder for én specifik samtale.
    """
    return [m for m in MESSAGES if m.conversation_id == conversation_id]


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
        raise HTTPException(status_code=404, detail="Conversation not found")

    conv.is_read = True
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
        raise HTTPException(status_code=404, detail="Conversation not found")

    conv.status = status
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

    ADMIN_DEVICE_TOKENS.add(token)
    logger.info(
        "Registrerede admin-device token (nu %d tokens)", len(ADMIN_DEVICE_TOKENS)
    )
    return {"ok": True}
