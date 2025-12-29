from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from typing import Optional

from config import settings
from database import get_db
import models
import utils

# Use HTTPBearer instead of OAuth2PasswordBearer
security = HTTPBearer()
# Optional security for endpoints that can work with or without auth
security_optional = HTTPBearer(auto_error=False)

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security), 
    db: Session = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        token = credentials.credentials  
        payload = jwt.decode(
            token, 
            settings.SECRET_KEY, 
            algorithms=[settings.ALGORITHM]
        )
        email: str = payload.get("sub")
        if email is None:
            print("⚠️ Token has no 'sub'")
            raise credentials_exception
    except JWTError as e:
        print(f"⚠️ JWT Decode Error: {e}")
        raise credentials_exception
        
    user = db.query(models.User).filter(models.User.email == email).first()
    if user is None:
        print(f"⚠️ No user found for email {email}")
        raise credentials_exception
    return user

def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_optional), 
    db: Session = Depends(get_db)
) -> Optional[models.User]:
    """
    Returns user if authenticated, else None.
    Does NOT raise generic 401/403 errors if no token provided.
    """
    if not credentials:
        return None
        
    try:
        token = credentials.credentials
        payload = jwt.decode(
            token, 
            settings.SECRET_KEY, 
            algorithms=[settings.ALGORITHM]
        )
        email: str = payload.get("sub")
        if email is None:
            return None
            
        user = db.query(models.User).filter(models.User.email == email).first()
        return user
        
    except JWTError:
        return None
    except Exception:
        return None

# Add this function at the end of utils.py
async def get_current_admin(current_user: models.User = Depends(get_current_user)):
    """
    Dependency to get current admin user. Raises 403 if user is not admin.
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user
    

async def get_current_user_from_reset_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """
    Special dependency for password reset token
    """
    token = credentials.credentials
    
    try:
        payload = utils.verify_reset_token(token)
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid reset token"
            )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired reset token"
        )

    user = db.query(models.User).filter(models.User.email == email).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user