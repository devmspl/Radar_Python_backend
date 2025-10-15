from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey, ARRAY, JSON, Enum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import uuid
from database import Base
import enum
from datetime import datetime
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
    __table_args__ = {"extend_existing": True}
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

    quiz_scores = relationship("UserQuizScore", back_populates="user")
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


class Blog(Base):
    __tablename__ = "blogs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    website = Column(String, index=True)  
    category = Column(String, index=True)  
    title = Column(String)
    description = Column(Text)
    content = Column(Text)
    url = Column(String, index=True)    
    job_uid = Column(String, index=True)

    feeds = relationship("Feed", back_populates="blog")
class ScrapeJob(Base):
    __tablename__ = "scrape_jobs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    uid = Column(String, unique=True, index=True)  # unique job ID
    website = Column(String, index=True)           # domain like outreach.io
    url = Column(String)                            # link provided for scraping
    status = Column(String, default="inprocess")   # inprocess / done / failed
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
from sqlalchemy.types import TypeDecorator, TEXT
import json

class JSONEncodedList(TypeDecorator):
    impl = TEXT

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return json.dumps(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return json.loads(value)
class Feed(Base):
    __tablename__ = "feeds"
    
    id = Column(Integer, primary_key=True, index=True)
    blog_id = Column(Integer, ForeignKey("blogs.id"), nullable=True)
    title = Column(String(500), nullable=True)
    categories = Column(JSONEncodedList, default=list)
    status = Column(String(50), default="processing")
    ai_generated_content = Column(JSON, nullable=True)
    image_generation_enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    blog = relationship("Blog", back_populates="feeds")
    slides = relationship("Slide", back_populates="feed", cascade="all, delete-orphan")
    transcript_id = Column(String, ForeignKey('transcripts.transcript_id'), nullable=True)  # Link to YouTube transcripts
    source_type = Column(String, default='blog')  # 'blog' or 'youtube'
    published_feed = relationship("PublishedFeed", back_populates="feed", uselist=False)
    quizzes = relationship("Quiz", back_populates="feed")

class Slide(Base):
    __tablename__ = "slides"
    
    id = Column(Integer, primary_key=True, index=True)
    feed_id = Column(Integer, ForeignKey("feeds.id"), nullable=False)
    order = Column(Integer, nullable=False)
    title = Column(String(500), nullable=False)
    body = Column(Text, nullable=False)
    bullets = Column(JSONEncodedList, default=list)
    background_image_url = Column(String(500), nullable=True)
    background_image_prompt = Column(String(1000), nullable=True)
    source_refs = Column(JSONEncodedList, default=list)
    render_markdown = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    feed = relationship("Feed", back_populates="slides")
# Add this to your models.py file
class PublishedFeed(Base):
    __tablename__ = "published_feeds"
    
    id = Column(Integer, primary_key=True, index=True)
    feed_id = Column(Integer, ForeignKey('feeds.id',ondelete='CASCADE'), nullable=False)
    admin_id = Column(Integer, ForeignKey('users.id'), nullable=False)  # Reference to users table
    admin_name = Column(String, nullable=False)
    published_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    feed = relationship("Feed", backref="published_feeds")
    admin_user = relationship("User")  # Relationship to User model

from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, JSON, ForeignKey, Float

class QuizCategory(Base):
    __tablename__ = "quiz_categories"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)  # UI/UX Design, Product Management, etc.
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    quizzes = relationship("Quiz", back_populates="category")
class Quiz(Base):
    __tablename__ = "quizzes"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    category_id = Column(Integer, ForeignKey('quiz_categories.id'), nullable=False)
    difficulty = Column(String, default="medium")  # easy, medium, hard
    questions = Column(JSON)  # Store questions as JSON array
    source_type = Column(String)  # 'blog' or 'youtube'
    source_id = Column(String)  # blog_id or transcript_id
    feed_id = Column(Integer, ForeignKey('feeds.id'), nullable=True)
    is_active = Column(Boolean, default=True)
    version = Column(Integer, default=1)  # Track quiz versions for updates
    last_updated = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    category = relationship("QuizCategory", back_populates="quizzes")
    feed = relationship("Feed", back_populates="quizzes")
    user_scores = relationship("UserQuizScore", back_populates="quiz")

class UserQuizScore(Base):
    __tablename__ = "user_quiz_scores"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)  # Assuming you have a User model
    quiz_id = Column(Integer, ForeignKey('quizzes.id'), nullable=False)
    score = Column(Float, nullable=False)  # Percentage score
    correct_answers = Column(Integer, nullable=False)
    total_questions = Column(Integer, nullable=False)
    time_taken = Column(Integer, nullable=True)  # Time in seconds
    answers = Column(JSON)  # Store user's answers
    completed_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="quiz_scores")
    quiz = relationship("Quiz", back_populates="user_scores")

# Add to existing models:
# QuizCategory.quizzes = relationship("Quiz", back_populates="category")
# Feed.quizzes = relationship("Quiz", back_populates="feed")
# User.quiz_scores = relationship("UserQuizScore", back_populates="user")