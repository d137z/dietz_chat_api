from datetime import datetime
from typing import List, Dict, Optional

import os

from fastapi import FastAPI, Depends, Header, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()


# --- Health / ping ---


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ping")
def ping():
    return {"message": "pong"}


# --- Modeller til samtaler og beskeder ---


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


# --- In-memory "database" (nulstilles ved restart) ---


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


def touch_conversation(conv_id: int, new_text: str, from_visitor: bool) -> ConversationSummary:
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
        # Svar fra dig -> læst
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
    Perfekt til din Android-app senere.
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
