from datetime import datetime
from typing import List, Dict, Optional

from fastapi import FastAPI
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
    name: Optional[str] = None
    email: Optional[str] = None


class ConversationSummary(BaseModel):
    id: int
    created_at: datetime
    last_message_at: datetime
    last_message_preview: str


# --- In-memory "database" (nulstilles ved restart) ---


MESSAGES: List[MessageOut] = []
CONVERSATIONS: Dict[int, ConversationSummary] = {}
NEXT_MESSAGE_ID = 1
NEXT_CONVERSATION_ID = 1


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
    )
    CONVERSATIONS[conv.id] = conv
    NEXT_CONVERSATION_ID += 1
    return conv


def update_conversation(conv_id: int, new_text: str) -> ConversationSummary:
    """Opdater metadata for en eksisterende samtale."""
    now = datetime.utcnow()
    conv = CONVERSATIONS.get(conv_id)

    if conv is None:
        # Hvis der på en eller anden måde kommer et ukendt conversation_id,
        # opretter vi en ny samtale med det id.
        conv = ConversationSummary(
            id=conv_id,
            created_at=now,
            last_message_at=now,
            last_message_preview=new_text[:120],
        )
        CONVERSATIONS[conv.id] = conv
    else:
        conv.last_message_at = now
        conv.last_message_preview = new_text[:120]

    return conv


# --- Endpoints ---


@app.post(
    "/messages",
    response_model=MessageOut,
    response_model_exclude_none=True,
)
def create_message(msg: MessageIn):
    """
    Modtag en ny besked fra websitet eller appen.

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
            conv = update_conversation(msg.conversation_id, msg.text)
            conv_id = conv.id
        else:
            # Ukendt conversation_id: opret en ny i stedet
            conv = create_conversation(msg.text)
            conv_id = conv.id

    # 2) Opret selve beskeden
    new_msg = MessageOut(
        id=NEXT_MESSAGE_ID,
        conversation_id=conv_id,
        created_at=datetime.utcnow(),
        text=msg.text,
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
def list_conversations():
    """
    Liste over alle samtaler – kan bruges i din app til at vise
    "hvem har skrevet ind".
    """
    # Returnér seneste først
    return sorted(
        CONVERSATIONS.values(),
        key=lambda c: c.last_message_at,
        reverse=True,
    )


@app.get(
    "/conversations/{conversation_id}/messages",
    response_model=List[MessageOut],
    response_model_exclude_none=True,
)
def get_conversation_messages(conversation_id: int):
    """
    Hent alle beskeder for én specifik samtale.
    Perfekt til din Android-app senere.
    """
    return [m for m in MESSAGES if m.conversation_id == conversation_id]

