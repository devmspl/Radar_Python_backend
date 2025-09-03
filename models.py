from sqlalchemy import Boolean, Column, Integer, String, DateTime, Text, ForeignKey, Enum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import uuid
from database import Base
import enum

def generate_uuid():
    return str(uuid.uuid4())

# Enums for job status and content type
class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class ContentType(str, enum.Enum):
    CHANNEL = "channel"
    PLAYLIST = "playlist"
    VIDEO = "video"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_verified = Column(Boolean, default=False)
    is_admin = Column(Boolean, default=False)
    agreed_terms = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationship to categories created by this admin
    categories = relationship("Category", back_populates="admin_user")
    # Relationship to transcript jobs created by this user
    transcript_jobs = relationship("TranscriptJob", back_populates="user")

class OTP(Base):
    __tablename__ = "otps"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, index=True, nullable=False)
    otp_code = Column(String, nullable=False)
    purpose = Column(String, nullable=False)  # 'verification' or 'password_reset'
    is_used = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)

class Category(Base):
    __tablename__ = "categories"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String, unique=True, default=generate_uuid, index=True)
    name = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)
    note = Column(Text, nullable=True)
    admin_note = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False   # enforce at DB level
    )
    
    # Foreign key to track which admin created this category
    admin_id = Column(Integer, ForeignKey("users.id"))
    admin_user = relationship("User", back_populates="categories")

# Transcript Job Model
class TranscriptJob(Base):
    __tablename__ = "transcript_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, unique=True, index=True, default=generate_uuid)
    url = Column(String, nullable=False)
    type = Column(Enum(ContentType), nullable=False)
    status = Column(Enum(JobStatus), default=JobStatus.QUEUED)
    model_size = Column(String, default="base")
    store_in_db = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    total_items = Column(Integer, default=0)
    processed_items = Column(Integer, default=0)
    description = Column(Text, nullable=True)
    content_name = Column(String, nullable=True)
    playlists = Column(Text, nullable=True)  # JSON string for playlists
    
    # Foreign key to track which user created this job
    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", back_populates="transcript_jobs")
    
    # Relationship to transcripts
    transcripts = relationship("Transcript", back_populates="job")

# Transcript Model
class Transcript(Base):
    __tablename__ = "transcripts"
    
    id = Column(Integer, primary_key=True, index=True)
    transcript_id = Column(String, unique=True, index=True, default=generate_uuid)
    video_id = Column(String, nullable=False, index=True)
    playlist_id = Column(String, nullable=True, index=True)
    title = Column(String, nullable=False)
    transcript_text = Column(Text, nullable=False)
    duration = Column(String, default="Unknown")
    word_count = Column(Integer, default=0)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Foreign key to the job
    job_id = Column(Integer, ForeignKey("transcript_jobs.id"))
    job = relationship("TranscriptJob", back_populates="transcripts")