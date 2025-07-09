from datetime import date, time

from pydantic import CustomModel, Field, HttpUrl

from src.config_schema import Accounts


class BookingSettings(CustomModel):
    api_url: HttpUrl


class Settings(CustomModel):
    accounts: Accounts
    booking: BookingSettings


# HTTP-schemas
class MusicRoomSlot(CustomModel):
    date: date
    start: time
    end: time


class AvailabilityResponse(CustomModel):
    available: bool
    message: str = None


class BookingRequest(MusicRoomSlot):
    confirm: bool = Field(False, description="Confirm the booking.")


class BookingResponse(CustomModel):
    booking_id: str
    status: str
    date: date
    start: time
    end: time
