from fastapi import APIRouter, FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from sqlalchemy.orm import Session
from typing import List
import models, schemas, utils, dependencies
from database import get_db, engine
from config import settings

# Import routers
from admin import router as admin_router
from youtube_router import router as youtube_router
from scraping import router as scrapping_router
from feed_router import router as feed_router
from publish_router import router as publish_router
# Auth router
auth_router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)

# Create tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="RADAR_API",
    docs_url=None,   # Disable default docs
    redoc_url=None,  # Disable default redoc
)

# Custom Swagger UI with simpler authentication
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="API Documentation",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css",
        swagger_ui_parameters={
            "defaultModelsExpandDepth": -1,  # Hide models section by default
            "docExpansion": "none",          # Collapse all operations by default
            "persistAuthorization": True,    # Keep authorization between refreshes
        }
    )

@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    openapi_schema = get_openapi(
        title="API Documentation",
        version="1.0.0",
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    for path in openapi_schema["paths"].values():
        for method in path.values():
            method.setdefault("security", [{"BearerAuth": []}])

    # 🚀 Remove default tag from schema
    if "tags" in openapi_schema:
        openapi_schema["tags"] = [
            t for t in openapi_schema["tags"] if t["name"].lower() != "default"
        ]

    return openapi_schema

# Include routers with explicit tags
app.include_router(admin_router)    
app.include_router(youtube_router)
app.include_router(scrapping_router)  
app.include_router(feed_router)
app.include_router(publish_router)     
      # tags=["Authentication"]

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*","https://admin-radar.careergraph.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- AUTH ROUTES ----------------

@auth_router.post("/register", response_model=schemas.UserResponse)
async def register(
    user: schemas.UserCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    hashed_password = utils.get_password_hash(user.password)
    db_user = models.User(
        full_name=user.full_name,
        last_name=user.last_name,
        email=user.email,
        hashed_password=hashed_password,
        agreed_terms=user.agreed_terms
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    otp = utils.generate_otp()
    utils.store_otp(user.email, otp, "verification")

    email_body = f"""
    <h2>Email Verification</h2>
    <p>Your verification code is: <strong>{otp}</strong></p>
    <p>This code will expire in {settings.OTP_EXPIRE_MINUTES} minutes.</p>
    """

    background_tasks.add_task(
        utils.send_email,
        user.email,
        "Email Verification",
        email_body
    )

    return db_user


@auth_router.post("/verify-email")
async def verify_email(otp_data: schemas.OTPVerify, db: Session = Depends(get_db)):
    if not utils.verify_otp(otp_data.email, otp_data.otp_code, "verification"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OTP"
        )

    db_user = db.query(models.User).filter(models.User.email == otp_data.email).first()
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    db_user.is_verified = True
    db.commit()

    utils.delete_otp(otp_data.email, "verification")

    return {"message": "Email verified successfully"}


@auth_router.post("/login")
async def login(user_data: schemas.UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user_data.email).first()
    if not db_user or not utils.verify_password(user_data.password, db_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )

    if not db_user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email not verified"
        )

    access_token = utils.create_access_token(data={"sub": db_user.email})

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": db_user.id,
        "full_name": db_user.full_name,
        "last_name": db_user.last_name,
        "role": "admin" if db_user.is_admin else "user"
    }


@auth_router.get("/users/{user_id}", response_model=schemas.UserResponse)
async def get_user_by_id(user_id: int, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return db_user


@auth_router.post("/forgot-password")
async def forgot_password(
    email_data: schemas.OTPRequest,
    background_tasks: BackgroundTasks
):
    otp = utils.generate_otp()
    utils.store_otp(email_data.email, otp, "password_reset")

    email_body = f"""
    <h2>Password Reset</h2>
    <p>Your password reset code is: <strong>{otp}</strong></p>
    <p>This code will expire in {settings.OTP_EXPIRE_MINUTES} minutes.</p>
    """

    background_tasks.add_task(
        utils.send_email,
        email_data.email,
        "Password Reset",
        email_body
    )

    return {"message": "Password reset OTP sent to your email"}


@auth_router.post("/reset-password")
async def reset_password(reset_data: schemas.PasswordReset, db: Session = Depends(get_db)):
    if not utils.verify_otp(reset_data.email, reset_data.otp_code, "password_reset"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OTP"
        )

    db_user = db.query(models.User).filter(models.User.email == reset_data.email).first()
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    db_user.hashed_password = utils.get_password_hash(reset_data.new_password)
    db.commit()

    utils.delete_otp(reset_data.email, "password_reset")

    return {"message": "Password reset successfully"}


@auth_router.get("/users", response_model=List[schemas.UserResponse])
async def get_users(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(dependencies.get_current_admin)
):
    return db.query(models.User).all()


@auth_router.post("/promote-to-admin/{user_id}")
def promote_to_admin(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(dependencies.get_current_admin)
):
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can promote users"
        )

    db_user = db.query(models.User).filter(models.User.id == user_id).first()

    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    db_user.is_admin = True
    db.commit()
    db.refresh(db_user)

    return {"message": "User promoted to admin successfully"}

app.include_router(auth_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7878)
