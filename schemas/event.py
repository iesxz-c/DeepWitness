from pydantic import BaseModel, Field
import re
from typing import Optional


class Event(BaseModel):
    time: str = Field(..., pattern=r"^\d{2}:\d{2}:\d{2}$", description="HH:MM:SS format")
    camera: str
    event_type: str
    description: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    case_id: Optional[str] = Field(default=None, description="Investigation case this event belongs to")


if __name__ == "__main__":
    e = Event(
        time="14:32:07",
        camera="cam-lobby-01",
        event_type="loitering",
        description="Unknown individual loitering near entrance for 5+ minutes",
        confidence=0.87,
    )
    print(e)
