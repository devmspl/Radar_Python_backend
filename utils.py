import random
import string
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status
from config import settings
import redis
import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from database import get_db  
import models

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# Add this after the existing imports
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
security = HTTPBearer()
# Redis connection
redis_client = redis.Redis.from_url(settings.REDIS_URL)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt
def generate_otp(length=6):
    return ''.join(random.choices(string.digits, k=length))

async def send_email(to_email: str, subject: str, body: str):
    message = MIMEMultipart()
    message["From"] = settings.SMTP_USERNAME
    message["To"] = to_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "html"))

    try:
        if settings.SMTP_PORT == 587:  # STARTTLS
            await aiosmtplib.send(
                message,
                hostname=settings.SMTP_SERVER,
                port=settings.SMTP_PORT,
                start_tls=True,   # <-- use STARTTLS
                username=settings.SMTP_USERNAME,
                password=settings.SMTP_PASSWORD,
            )
        elif settings.SMTP_PORT == 465:  # SSL
            await aiosmtplib.send(
                message,
                hostname=settings.SMTP_SERVER,
                port=settings.SMTP_PORT,
                use_tls=True,     # <-- direct SSL
                username=settings.SMTP_USERNAME,
                password=settings.SMTP_PASSWORD,
            )
        else:
            raise ValueError(f"Unsupported SMTP port: {settings.SMTP_PORT}")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send email: {str(e)}"
        )


def store_otp(email: str, otp: str, purpose: str):
    expires = datetime.utcnow() + timedelta(minutes=settings.OTP_EXPIRE_MINUTES)
    key = f"{purpose}:{email}"
    redis_client.setex(key, settings.OTP_EXPIRE_MINUTES * 60, f"{otp}:{expires.isoformat()}")

def verify_otp(email: str, otp: str, purpose: str) -> bool:
    key = f"{purpose}:{email}"
    stored_data = redis_client.get(key)
    
    if not stored_data:
        return False
    
    stored_otp, expires_str = stored_data.decode().split(":", 1)  # <-- fix here
    expires = datetime.fromisoformat(expires_str)
    
    if datetime.utcnow() > expires:
        redis_client.delete(key)
        return False
    
    return stored_otp == otp


def delete_otp(email: str, purpose: str):
    key = f"{purpose}:{email}"
    redis_client.delete(key)
    # Add these functions to your existing utils.py

def create_reset_token(data: dict):
    """
    Create a short-lived token specifically for password reset
    Valid for 15 minutes only
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire, "type": "password_reset"})
    
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.SECRET_KEY, 
        algorithm=settings.ALGORITHM
    )
    return encoded_jwt

def verify_reset_token(token: str):
    """
    Verify password reset token
    """
    try:
        payload = jwt.decode(
            token, 
            settings.SECRET_KEY, 
            algorithms=[settings.ALGORITHM]
        )
        
        # Check if this is a reset token
        if payload.get("type") != "password_reset":
            raise JWTError("Invalid token type")
            
        email: str = payload.get("sub")
        if email is None:
            raise JWTError("Invalid token payload")
            
        return payload
        
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired reset token"
        )