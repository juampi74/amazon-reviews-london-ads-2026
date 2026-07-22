from typing import Generator
from fastapi import Depends
from sqlalchemy.orm import Session

from app.clients.supabase import SupabaseRestClient
from app.core.auth import AuthenticatedUser, get_current_user
from app.core.config import Settings, get_settings
from app.repositories.launchly import LaunchlyRepository
from app.database import SessionLocal


def get_repository(
    user: AuthenticatedUser = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> LaunchlyRepository:
    return LaunchlyRepository(SupabaseRestClient(settings, user.access_token), owner_id=user.id)

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()