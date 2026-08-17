"""
Authentication & authorization.

Deliberately built as a realistic (not intentionally backdoored) JWT + RBAC
layer, so that anything you find by attacking it reflects genuine classes of
bugs (e.g. IDOR, missing role checks on a sub-route, JWT misconfig) rather
than a planted flag. Good angles to test:
  - token tampering (alg=none, expired tokens, role field tampering)
  - IDOR: patient A requesting patient B's /patients/{id} or /chat history
  - privilege escalation: patient-role token hitting doctor/admin routes
  - missing rate limiting on /auth/login (credential stuffing)
"""
import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from database import get_db, User

# In a real deployment this MUST come from a secret manager / env var.
SECRET_KEY = os.environ.get("HEALTHCARE_DEMO_JWT_SECRET", "dev-only-secret-change-me")
ALGORITHM = "HS256"
# Default is generous for local dev/testing so you don't get logged out
# mid-session. Set HEALTHCARE_DEMO_TOKEN_EXPIRE_MINUTES=60 (or lower) in
# .env if you specifically want to test token-expiration behavior.
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("HEALTHCARE_DEMO_TOKEN_EXPIRE_MINUTES", "480"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()
    if user is None or not user.is_active:
        raise credentials_exception
    return user


def require_roles(*allowed_roles: str):
    """FastAPI dependency factory for role-based route protection."""
    def _checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{current_user.role}' is not permitted to access this resource",
            )
        return current_user
    return _checker
