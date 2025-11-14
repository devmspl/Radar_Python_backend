# file_utils.py
import os
import uuid
from fastapi import UploadFile, HTTPException
from PIL import Image
import io
import aiofiles

# Ensure upload directory exists
UPLOAD_DIR = "uploads/profile_photos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

async def save_profile_photo(file: UploadFile, user_id: int) -> str:
    """
    Save profile photo and return the file path
    """
    # Validate file type
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Only JPEG, PNG, GIF, and WebP are allowed."
        )
    
    # Validate file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 5MB."
        )
    
    # Generate unique filename
    file_extension = os.path.splitext(file.filename)[1]
    filename = f"{user_id}_{uuid.uuid4().hex}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if necessary
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        
        # Resize image to max 500x500 while maintaining aspect ratio
        image.thumbnail((500, 500), Image.Resampling.LANCZOS)
        
        # Save optimized image
        image.save(file_path, "JPEG", quality=85, optimize=True)
        
        return f"/{file_path}"
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing image: {str(e)}"
        )

def delete_profile_photo(file_path: str):
    """
    Delete profile photo file
    """
    if file_path and os.path.exists(file_path.lstrip('/')):
        try:
            os.remove(file_path.lstrip('/'))
        except Exception:
            pass  # Silently fail if file doesn't exist

async def handle_profile_photo_upload(file: UploadFile, user_id: int, current_photo: str = None) -> str:
    """
    Handle profile photo upload and delete old photo if exists
    """
    # Delete old photo if exists
    if current_photo:
        delete_profile_photo(current_photo)
    
    # Save new photo
    return await save_profile_photo(file, user_id)