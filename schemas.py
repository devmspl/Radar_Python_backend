from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum

# Enums for job status and content type
class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class ContentType(str, Enum):
    CHANNEL = "channel"
    PLAYLIST = "playlist"
    VIDEO = "video"

# ... (existing schemas remain the same)

class UserBase(BaseModel):
    email: EmailStr
    full_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)
    agreed_terms: bool = Field(..., description="Must agree to terms and conditions")
    
    @validator('agreed_terms')
    def check_terms_agreed(cls, v):
        if not v:
            raise ValueError('You must agree to the terms and conditions')
        return v

class UserResponse(UserBase):
    id: int
    is_verified: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class OTPRequest(BaseModel):
    email: EmailStr

class OTPVerify(BaseModel):
    email: EmailStr
    otp_code: str

class PasswordReset(BaseModel):
    email: EmailStr
    otp_code: str
    new_password: str = Field(..., min_length=8)

class CategoryBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    note: Optional[str] = None
    admin_note: Optional[str] = None

class CategoryCreate(CategoryBase):
    pass

class CategoryResponse(CategoryBase):
    id: int
    uuid: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
    admin_id: int
    
    class Config:
        from_attributes = True

class CategoryUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    note: Optional[str] = None
    admin_note: Optional[str] = None
    is_active: Optional[bool] = None

# YouTube Transcript Schemas
class TranscriptRequest(BaseModel):
    youtube_url: str
    model_size: str = "base"
    store_in_db: bool = True

class TranscriptResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str
    content_type: ContentType
    content_name: Optional[str] = None
    playlists: List[Dict] = []

class TranscriptItem(BaseModel):
    id: str
    video_id: str
    playlist_id: Optional[str]
    title: str
    transcript: str
    duration: str
    word_count: int
    created_at: datetime
    description: Optional[str] = None

class JobStatusResponse(BaseModel):
    job_id: str
    url: str
    type: ContentType
    status: JobStatus
    created_at: datetime
    completed_at: Optional[datetime]
    total_items: int
    processed_items: int
    description: Optional[str] = None
    content_name: Optional[str] = None
    playlists: List[Dict] = []
    # transcripts: List[TranscriptItem] = []
    
    class Config:
        from_attributes = True
class JobSimpleStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    processed_items: Optional[int] = None
    total_items: Optional[int] = None


# --- YouTube Channel / Playlist / Video Schemas ---
class PlaylistBase(BaseModel):
    id: str
    title: str
    description: Optional[str] = None


class VideoBase(BaseModel):
    id: str
    title: str
    description: Optional[str] = None



class VideoWithTranscript(VideoBase):
    transcript: Optional[str] = None


class ChannelPlaylistsResponse(BaseModel):
    channel_id: str
    playlists: List[PlaylistBase]


class VideoTranscriptResponse(BaseModel):
    video_id: str
    video: VideoWithTranscript

class PlaylistVideo(BaseModel):
    video_id: str
    title: str
   
class PlaylistVideosResponse(BaseModel):
    playlist_id: str
    playlist_name: Optional[str] = None
    videos: List[PlaylistVideo]
class ChannelPlaylist(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    videos: List[PlaylistVideo] = []

class ChannelPlaylistsResponse(BaseModel):
    channel_id: str
    playlists: List[ChannelPlaylist]
class JobContentStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    processed_items: Optional[int] = None
    total_items: Optional[int] = None
    type: ContentType
    # content: Dict
class ChannelWithPlaylists(BaseModel):
    channel_id: str
    title: Optional[str] = None
    description: Optional[str] = None
    playlists: List[ChannelPlaylist] = []


class PlaylistWithVideos(BaseModel):
    playlist_id: str
    title: Optional[str] = None
    description: Optional[str] = None
    videos: List[PlaylistVideo] = []


# app/schemas.py
from pydantic import BaseModel
from typing import List, Optional

class BlogBase(BaseModel):
    id : int
    website : str   # outreach, gong, etc.
    category : str  # blog, webinars, reports, etc.
    title : str
    description : str
    content : str
    url : str


class BlogCreate(BlogBase):
    pass

class Blog(BlogBase):
    id: int

    class Config:
        orm_mode = True

class BlogListResponse(BaseModel):
    blogs: List[Blog]
