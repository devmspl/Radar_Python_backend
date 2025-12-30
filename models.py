from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey, ARRAY, JSON, Enum, UniqueConstraint
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import uuid
from database import Base
import enum
from datetime import datetime
from sqlalchemy.types import TypeDecorator, TEXT
import json


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

class FilterType(str, enum.Enum):
    WEBINAR = "Webinar"
    BLOG = "Blog"
    PODCAST = "Podcast"
    VIDEO = "Video"


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
    is_blocked = Column(Boolean, default=False)
    # Relationship to categories created by this admin
    categories = relationship("Category", back_populates="admin_user")
    # Relationship to transcript jobs created by this user
    transcript_jobs = relationship("TranscriptJob", back_populates="user")
    profile_photo = Column(String(500), nullable=True)
    quiz_scores = relationship("UserQuizScore", back_populates="user")
     # Relationship to onboarding data
    onboarding_data = relationship("UserOnboarding", back_populates="user", uselist=False)
    # Add this relationship
    bookmarks = relationship("Bookmark", back_populates="user")

     # Add these relationships
    skill_tools = relationship("UserSkillTool", back_populates="user")
    roles = relationship("UserRole", back_populates="user")
class UserOnboarding(Base):
    __tablename__ = "user_onboarding"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    
    # Question 1: Domains of interest
    domains_of_interest = Column(JSON, nullable=True)
    
    # Question 2: Skills & Tools
    skills_tools = Column(JSON, nullable=True)
    
    # Question 3: Roles interested in
    interested_roles = Column(JSON, nullable=True)
    
    # Question 4: Social links
    social_links = Column(JSON, nullable=True)
    
    # Question 5: Emails
    work_email = Column(String(255), nullable=True)
    personal_email = Column(String(255), nullable=True)
    
    # Question 6: Job seeking status
    looking_for_job = Column(String(50), nullable=True)
    
    # Question 7: Career stage
    career_stage = Column(String(50), nullable=True)
    
    # Question 8: Years of experience
    years_experience = Column(String(50), nullable=True)
    
    # Question 9: Goals
    goals = Column(JSON, nullable=True)
    
    # Question 10: Market/Geography
    market_geography = Column(JSON, nullable=True)
    
    # Question 11: Qualifications
    qualifications = Column(JSON, nullable=True)
    
    # Question 12: Education
    education = Column(JSON, nullable=True)
    
    # Question 13: Companies
    companies = Column(JSON, nullable=True)
    
    # Question 14: Certifications
    certifications = Column(JSON, nullable=True)
    
    # Completion status
    is_completed = Column(Boolean, default=False)
    completed_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    user = relationship("User", back_populates="onboarding_data")
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
    subcategories = relationship("SubCategory", back_populates="category", cascade="all, delete-orphan")
    feeds = relationship("Feed", back_populates="category")
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
    generate_feed = Column(Boolean, default=False)  # NEW: Auto-generate feed
    feed_generated = Column(Boolean, default=False)  

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
    generate_feed = Column(Boolean, default=False)  # NEW: Auto-generate feed
    feed_generated = Column(Boolean, default=False)  # NEW: Track if feed was generated
class ScrapeJob(Base):
    __tablename__ = "scrape_jobs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    uid = Column(String, unique=True, index=True)  # unique job ID
    website = Column(String, index=True)           # domain like outreach.io
    url = Column(String)                            # link provided for scraping
    status = Column(String, default="inprocess")   # inprocess / done / failed
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

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
    content_type = Column(Enum(FilterType), default=FilterType.BLOG)
    skills = Column(JSONEncodedList, default=list)
    tools = Column(JSONEncodedList, default=list)
    roles = Column(JSONEncodedList, default=list)
    status = Column(String(50), default="processing")
    ai_generated_content = Column(JSON, nullable=True)
    image_generation_enabled = Column(Boolean, default=True)
    embedding = Column(JSON, nullable=True)  # Vector embedding for semantic search
    click_count = Column(Integer, default=0) # CTR for behavioral learning
    language = Column(String(10), default="en") # Content language
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    blog = relationship("Blog", back_populates="feeds")
    slides = relationship("Slide", back_populates="feed", cascade="all, delete-orphan")
    transcript_id = Column(String, ForeignKey('transcripts.transcript_id'), nullable=True)
    source_type = Column(String, default='blog')
    published_feed = relationship("PublishedFeed", back_populates="feed", uselist=False, overlaps="published_feed")
    quizzes = relationship("Quiz", back_populates="feed")
    bookmarks = relationship("Bookmark", back_populates="feed")
    category_id = Column(Integer, ForeignKey('categories.id'), nullable=True)  # Changed to Integer
    subcategory_id = Column(Integer, ForeignKey('subcategories.id'), nullable=True)  # Changed to Integer
    # Relationships
    category = relationship("Category", back_populates="feeds")
    subcategory = relationship("SubCategory", back_populates="feeds")
    concepts = relationship("Concept", secondary="feed_concepts", backref="feeds")
class Slide(Base):
    __tablename__ = "slides"
    
    id = Column(Integer, primary_key=True, index=True)
    feed_id = Column(Integer, ForeignKey("feeds.id"), nullable=False)
    order = Column(Integer, nullable=False)
    title = Column(String(500), nullable=False)
    body = Column(Text, nullable=False)
    bullets = Column(JSONEncodedList, default=list)
    background_color = Column(String, default="#FFFFFF")
    source_refs = Column(JSONEncodedList, default=list)
    render_markdown = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    feed = relationship("Feed", back_populates="slides",overlaps="published_feed")

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

class Topic(Base):
    __tablename__ = "topics"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)
    follower_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Source(Base):
    __tablename__ = "sources"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    website = Column(String, nullable=False)
    source_type = Column(String, nullable=False)  # 'blog' or 'youtube'
    follower_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class UserTopicFollow(Base):
    __tablename__ = "user_topic_follows"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)  # You can integrate with your user system
    topic_id = Column(Integer, ForeignKey('topics.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    topic = relationship("Topic")

class UserSourceFollow(Base):
    __tablename__ = "user_source_follows"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)  # You can integrate with your user system
    source_id = Column(Integer, ForeignKey('sources.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    source = relationship("Source")


# Add this to your existing models
class Bookmark(Base):
    __tablename__ = "bookmarks"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    feed_id = Column(Integer, ForeignKey("feeds.id"), nullable=False, index=True)
    
    # Additional metadata
    notes = Column(Text, nullable=True)  # User can add personal notes
    tags = Column(JSON, nullable=True)   # User-defined tags for organization
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="bookmarks")
    feed = relationship("Feed", back_populates="bookmarks")
    
    # Unique constraint to prevent duplicate bookmarks
    __table_args__ = (UniqueConstraint('user_id', 'feed_id', name='unique_user_feed_bookmark'),)

    # Add to existing models

class SkillTool(Base):
    __tablename__ = "skill_tools"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    category = Column(String, index=True, nullable=False)  # e.g., "Programming", "Design", "Analytics"
    description = Column(Text, nullable=True)
    popularity = Column(Integer, default=0)  # Track how often it's selected
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Role(Base):
    __tablename__ = "roles"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, unique=True, index=True, nullable=False)
    category = Column(String, index=True, nullable=False)  # e.g., "Engineering", "Design", "Product"
    description = Column(Text, nullable=True)
    seniority_levels = Column(JSON, nullable=True)  # ["Junior", "Mid", "Senior", "Lead"]
    popularity = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class UserSkillTool(Base):
    __tablename__ = "user_skill_tools"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    skill_tool_id = Column(Integer, ForeignKey("skill_tools.id"), nullable=False)
    proficiency_level = Column(String, default="intermediate")  # beginner, intermediate, advanced, expert
    years_of_experience = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User")
    skill_tool = relationship("SkillTool")

class UserRole(Base):
    __tablename__ = "user_roles"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    role_id = Column(Integer, ForeignKey("roles.id"), nullable=False)
    seniority_level = Column(String, nullable=True)  # e.g., "Junior", "Senior", "Lead"
    is_current = Column(Boolean, default=False)
    is_target = Column(Boolean, default=False)  # Target role for career growth
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User")
    role = relationship("Role")



class SubCategory(Base):
    __tablename__ = "subcategories"
    
    id = Column(Integer, primary_key=True, index=True)  # Changed to Integer
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    category_id = Column(Integer, ForeignKey('categories.id'), nullable=False)  # Changed to Integer
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    uuid = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True)  # Optional: Keep UUID for external reference
    
    # Relationship back to category
    category = relationship("Category", back_populates="subcategories")
    
    # Relationship to feeds
    feeds = relationship("Feed", back_populates="subcategory")



# Add to your models.py

class Domain(Base):
    __tablename__ = "domains"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)
    parent_domain_id = Column(Integer, ForeignKey('domains.id'), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    parent = relationship("Domain", remote_side=[id], backref="subdomains")
    concepts = relationship("Concept", secondary="domain_concepts", backref="domains")

class Concept(Base):
    __tablename__ = "concepts"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)
    related_concepts = Column(JSON, default=list)  # Array of related concept names
    popularity_score = Column(Integer, default=0)
    embedding = Column(JSON, nullable=True)
    click_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ContentList(Base):
    __tablename__ = "content_lists"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    source_type = Column(String)  # 'youtube_playlist' or 'custom'
    source_id = Column(String)  # playlist_id for YouTube, custom_id for manual lists
    feed_ids = Column(JSON, default=list)  # Array of feed IDs in this list
    embedding = Column(JSON, nullable=True)
    click_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Association tables for many-to-many relationships
class DomainConcept(Base):
    __tablename__ = "domain_concepts"
    
    id = Column(Integer, primary_key=True, index=True)
    domain_id = Column(Integer, ForeignKey('domains.id'), nullable=False)
    concept_id = Column(Integer, ForeignKey('concepts.id'), nullable=False)
    relevance_score = Column(Float, default=1.0)
    created_at = Column(DateTime, default=datetime.utcnow)

class FeedConcept(Base):
    __tablename__ = "feed_concepts"
    
    id = Column(Integer, primary_key=True, index=True)
    feed_id = Column(Integer, ForeignKey('feeds.id'), nullable=False)
    concept_id = Column(Integer, ForeignKey('concepts.id'), nullable=False)
    confidence_score = Column(Float, default=1.0)  # How confident LLM is about this concept
    created_at = Column(DateTime, default=datetime.utcnow)
