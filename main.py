from datetime import datetime
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()


# --- Health / ping (som før) ---


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ping")
def ping():
    return {"message": "pong"}


# --- Simpel besked-model (V1) ---


class MessageIn(BaseModel):
    name: str = Field(..., description="Navn eller label på afsender (fx 'Anon', 'Web visitor')")
    email: str | None = Field(None, description="Valgfri e-mail, hvis brugeren skriver den")
    text: str = Field(..., description="Selve beskeden fra brugeren")


class MessageOut(MessageIn):
    id: int
    created_at: datetime


# Midlertidig in-memory storage (bliver nulstillet ved restart)
MESSAGES: List[MessageOut] = []
NEXT_ID = 1


@app.post("/messages", response_model=MessageOut)
def create_message(msg: MessageIn):
    """Modtag en ny besked fra fx websitet."""
    global NEXT_ID
    new_msg = MessageOut(
        id=NEXT_ID,
        created_at=datetime.utcnow(),
        **msg.model_dump(),
    )
    MESSAGES.append(new_msg)
    NEXT_ID += 1
    return new_msg


@app.get("/messages", response_model=List[MessageOut])
def list_messages():
    """Hent alle beskeder (senere kan vi filtrere pr. kunde/conversation)."""
    return list(MESSAGES)
