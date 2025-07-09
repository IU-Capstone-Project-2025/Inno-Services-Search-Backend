from fastapi import APIRouter, Depends, Header, HTTPException

from src.modules.act.music_room.repository import ActRepository
from src.modules.act.music_room.schemas import AvailabilityResponse, BookingRequest, BookingResponse

router = APIRouter(prefix="/act/music-room", tags=["act"])
repo = ActRepository()


async def get_token(authorization: str = Header(...)) -> str:
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    return token


@router.post("/availability", response_model=AvailabilityResponse)
async def check_availability(
    req: BookingRequest,
    token: str = Depends(get_token),
):
    return await repo.check_availability(req, token)


@router.post("/book", response_model=BookingResponse)
async def book_room(
    req: BookingRequest,
    token: str = Depends(get_token),
):
    if not req.confirm:
        raise HTTPException(status_code=400, detail="Confirm must be True to book")
    return await repo.create_booking(req, token)
