from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.exc import IntegrityError
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging

from database import get_db
from models import Bookmark, User, Feed, Blog, Slide,Transcript
from schemas import BookmarkCreate, BookmarkUpdate, BookmarkResponse, BookmarkListResponse,BookmarkCreateResponse

# Configure logging
logger = logging.getLogger(__name__)

# Create router
bookmark_router = APIRouter(prefix="/bookmarks", tags=["Bookmarks"])

# Enhanced helper function to get complete feed data with slides
def get_complete_feed_data(db: Session, feed_id: int) -> Dict[str, Any]:
    """Get complete feed data including slides, blog info, and metadata."""
    feed = db.query(Feed).options(
        joinedload(Feed.blog), 
        joinedload(Feed.slides),
        joinedload(Feed.published_feed)
    ).filter(Feed.id == feed_id).first()
    
    if not feed:
        return {}
    
    # Determine source metadata based on source_type with proper URL formatting
    if feed.source_type == "youtube":
        # Get the actual YouTube video ID from the Transcript table
        video_id = None
        author = "YouTube Creator"
        channel_name = None
        
        # Query the Transcript table using the transcript_id (UUID)
        if feed.transcript_id:
            transcript = db.query(Transcript).filter(Transcript.transcript_id == feed.transcript_id).first()
            if transcript:
                # Get the actual YouTube video ID
                if hasattr(transcript, 'video_id') and transcript.video_id:
                    video_id = transcript.video_id
                # Get author/channel information
                if hasattr(transcript, 'channel_name') and transcript.channel_name:
                    author = transcript.channel_name
                    channel_name = transcript.channel_name
                elif hasattr(transcript, 'author') and transcript.author:
                    author = transcript.author
        
        meta = {
            "title": feed.title,
            "original_title": feed.title,
            "author": author,
            "source_url": f"https://www.youtube.com/watch?v={video_id}" if video_id else "#",
            "source_type": "youtube",
            "video_id": video_id,
            "channel_name": channel_name,
            "transcript_id": feed.transcript_id  # Include the UUID for reference
        }
    else:
        meta = {
            "title": feed.title,
            "original_title": feed.blog.title if feed.blog else "Unknown",
            "author": getattr(feed.blog, 'author', 'Admin'),
            "source_url": getattr(feed.blog, 'url', '#'),
            "source_type": "blog",
            "website": getattr(feed.blog, 'website', '') if feed.blog else ''
        }
    
    ai_content = getattr(feed, 'ai_generated_content', {})
    
    # Get slides data
    slides = sorted([
        {
            "id": slide.id,
            "order": slide.order,
            "title": slide.title,
            "body": slide.body,
            "bullets": slide.bullets,
            "background_color": slide.background_color,
            "background_image_prompt": None,
            "source_refs": slide.source_refs,
            "render_markdown": bool(slide.render_markdown),
            "created_at": slide.created_at.isoformat() if slide.created_at else None,
            "updated_at": slide.updated_at.isoformat() if slide.updated_at else None
        } for slide in feed.slides
    ], key=lambda x: x["order"])
    
    # Check if published
    is_published = feed.published_feed is not None
    
    return {
        "id": feed.id,
        "blog_id": feed.blog_id,
        "transcript_id": feed.transcript_id,
        "title": feed.title,
        "categories": feed.categories,
        "status": feed.status,
        "source_type": feed.source_type or "blog",
        "ai_generated_content": ai_content,
        "is_published": is_published,
        "published_at": feed.published_feed.published_at.isoformat() if is_published else None,
        "meta": meta,
        "slides": slides,
        "slides_count": len(slides),
        "created_at": feed.created_at.isoformat() if feed.created_at else None,
        "updated_at": feed.updated_at.isoformat() if feed.updated_at else None,
        "ai_generated": hasattr(feed, 'ai_generated_content') and feed.ai_generated_content is not None,
        "images_generated": False
    }

def extract_youtube_video_id(feed: Feed) -> Optional[str]:
    """Extract proper YouTube video ID from feed data."""
    # Try different possible fields that might contain the video ID
    
    # 1. First try transcript_id (if it's a proper YouTube ID)
    if feed.transcript_id:
        # Check if transcript_id looks like a YouTube video ID (typically 11 characters)
        if len(feed.transcript_id) == 11 and feed.transcript_id.isalnum():
            return feed.transcript_id
        # If transcript_id is a UUID, we need to look elsewhere
    
    # 2. Check if there's a related Transcript object with video_id
    try:
        from models import Transcript
        transcript = feed.transcript  # If relationship exists
        if transcript and transcript.video_id:
            # Validate it looks like a YouTube video ID
            if len(transcript.video_id) == 11 and transcript.video_id.isalnum():
                return transcript.video_id
    except (AttributeError, Exception):
        pass
    
    # 3. Check if title or other fields might contain hints
    # You might need to implement additional logic based on your data structure
    
    # 4. As a fallback, check if there's any field that might contain YouTube ID
    # This depends on your specific data structure
    
    logger.warning(f"Could not extract valid YouTube video ID for feed {feed.id}")
    return None
# ------------------ Bookmark API Endpoints ------------------

@bookmark_router.post("/", response_model=BookmarkCreateResponse, status_code=status.HTTP_201_CREATED)
def create_bookmark(
    bookmark_data: BookmarkCreate,
    user_id: int = 1,  # In production, get from auth token
    db: Session = Depends(get_db)
):
    """Create a new bookmark for a feed."""
    try:
        # Check if feed exists
        feed = db.query(Feed).filter(Feed.id == bookmark_data.feed_id).first()
        if not feed:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="Feed not found"
            )
        
        # Check if bookmark already exists
        existing_bookmark = db.query(Bookmark).filter(
            Bookmark.user_id == user_id,
            Bookmark.feed_id == bookmark_data.feed_id
        ).first()
        
        if existing_bookmark:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="Feed already bookmarked"
            )
        
        # Create new bookmark
        bookmark = Bookmark(
            user_id=user_id,
            feed_id=bookmark_data.feed_id,
            notes=bookmark_data.notes,
            tags=bookmark_data.tags
        )
        
        db.add(bookmark)
        db.commit()
        db.refresh(bookmark)
        
        logger.info(f"Bookmark created: {bookmark.id} for user {user_id} on feed {bookmark_data.feed_id}")
        
        return {
            "message": "Bookmark created successfully for user",
            "bookmark_id": bookmark.id,
            "feed_id": bookmark.feed_id,
            "user_id": user_id
        }
        
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Bookmark already exists"
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating bookmark: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Failed to create bookmark"
        )

@bookmark_router.get("/{bookmark_id}", response_model=BookmarkResponse)
def get_bookmark(
    bookmark_id: int,
    user_id: int = 1,  # In production, get from auth token
    db: Session = Depends(get_db)
):
    """Get a specific bookmark by ID."""
    bookmark = db.query(Bookmark).filter(
        Bookmark.id == bookmark_id,
        Bookmark.user_id == user_id
    ).first()
    
    if not bookmark:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Bookmark not found"
        )
    
    # Get complete feed data
    feed_with_details = get_complete_feed_data(db, bookmark.feed_id)
    
    return {
        "id": bookmark.id,
        "user_id": bookmark.user_id,
        "feed_id": bookmark.feed_id,
        "notes": bookmark.notes,
        "tags": bookmark.tags,
        "created_at": bookmark.created_at,
        "updated_at": bookmark.updated_at,
        "feed": feed_with_details
    }

@bookmark_router.get("/", response_model=BookmarkListResponse)
def get_all_bookmarks(
    user_id: int = 1,  # In production, get from auth token
    page: int = 1,
    limit: int = 20,
    tags: Optional[str] = None,  # Comma-separated tags to filter by
    include_slides: bool = True,  # New parameter to include slides data
    db: Session = Depends(get_db)
):
    """Get all bookmarks for a user with pagination, filtering, and complete feed data."""
    # Validate pagination parameters
    if page < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Page must be greater than 0"
        )
    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Limit must be between 1 and 100"
        )
    
    # Build query
    query = db.query(Bookmark).filter(Bookmark.user_id == user_id)
    
    # Filter by tags if provided
    if tags:
        tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        if tag_list:
            # Filter bookmarks that have any of the specified tags
            query = query.filter(Bookmark.tags.op('&&')(tag_list))
    
    # Count total
    total = query.count()
    
    # Apply pagination
    bookmarks = query.order_by(Bookmark.created_at.desc()).offset((page - 1) * limit).limit(limit).all()
    
    # Format response with complete feed data
    items = []
    for bookmark in bookmarks:
        feed_with_details = get_complete_feed_data(db, bookmark.feed_id)
        
        items.append({
            "id": bookmark.id,
            "user_id": bookmark.user_id,
            "feed_id": bookmark.feed_id,
            "notes": bookmark.notes,
            "tags": bookmark.tags,
            "created_at": bookmark.created_at,
            "updated_at": bookmark.updated_at,
            "feed": feed_with_details
        })
    
    has_more = (page * limit) < total
    
    logger.info(f"Retrieved {len(items)} bookmarks for user {user_id} with complete feed data")
    
    return {
        "items": items,
        "total": total,
        "page": page,
        "limit": limit,
        "has_more": has_more
    }

@bookmark_router.put("/{bookmark_id}", response_model=BookmarkResponse)
def update_bookmark(
    bookmark_id: int,
    bookmark_data: BookmarkUpdate,
    user_id: int = 1,  # In production, get from auth token
    db: Session = Depends(get_db)
):
    """Update a bookmark (notes and tags)."""
    bookmark = db.query(Bookmark).filter(
        Bookmark.id == bookmark_id,
        Bookmark.user_id == user_id
    ).first()
    
    if not bookmark:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Bookmark not found"
        )
    
    try:
        # Update fields
        if bookmark_data.notes is not None:
            bookmark.notes = bookmark_data.notes
        if bookmark_data.tags is not None:
            bookmark.tags = bookmark_data.tags
        
        bookmark.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(bookmark)
        
        # Get complete feed data
        feed_with_details = get_complete_feed_data(db, bookmark.feed_id)
        
        logger.info(f"Bookmark updated: {bookmark.id}")
        
        return {
            "id": bookmark.id,
            "user_id": bookmark.user_id,
            "feed_id": bookmark.feed_id,
            "notes": bookmark.notes,
            "tags": bookmark.tags,
            "created_at": bookmark.created_at,
            "updated_at": bookmark.updated_at,
            "feed": feed_with_details
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating bookmark {bookmark_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Failed to update bookmark"
        )

@bookmark_router.delete("/{bookmark_id}", response_model=dict)
def delete_bookmark(
    bookmark_id: int,
    user_id: int = 1,  # In production, get from auth token
    db: Session = Depends(get_db)
):
    """Delete a bookmark by ID."""
    bookmark = db.query(Bookmark).filter(
        Bookmark.id == bookmark_id,
        Bookmark.user_id == user_id
    ).first()
    
    if not bookmark:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Bookmark not found"
        )
    
    try:
        bookmark_id = bookmark.id
        feed_id = bookmark.feed_id
        
        db.delete(bookmark)
        db.commit()
        
        logger.info(f"Bookmark deleted: {bookmark_id} for user {user_id}")
        
        return {
            "message": "Bookmark deleted successfully",
            "bookmark_id": bookmark_id,
            "feed_id": feed_id
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting bookmark {bookmark_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Failed to delete bookmark"
        )

@bookmark_router.delete("/feed/{feed_id}", response_model=dict)
def delete_bookmark_by_feed(
    feed_id: int,
    user_id: int = 1,  # In production, get from auth token
    db: Session = Depends(get_db)
):
    """Delete a bookmark by feed ID."""
    bookmark = db.query(Bookmark).filter(
        Bookmark.feed_id == feed_id,
        Bookmark.user_id == user_id
    ).first()
    
    if not bookmark:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Bookmark not found for this feed"
        )
    
    try:
        bookmark_id = bookmark.id
        
        db.delete(bookmark)
        db.commit()
        
        logger.info(f"Bookmark deleted by feed: {bookmark_id} for feed {feed_id}")
        
        return {
            "message": "Bookmark deleted successfully",
            "bookmark_id": bookmark_id,
            "feed_id": feed_id
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting bookmark by feed {feed_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Failed to delete bookmark"
        )

@bookmark_router.get("/check/{feed_id}", response_model=dict)
def check_bookmark_status(
    feed_id: int,
    user_id: int = 1,  # In production, get from auth token
    db: Session = Depends(get_db)
):
    """Check if a feed is bookmarked by the user."""
    bookmark = db.query(Bookmark).filter(
        Bookmark.feed_id == feed_id,
        Bookmark.user_id == user_id
    ).first()
    
    bookmark_data = None
    if bookmark:
        # Get minimal feed info for the check endpoint
        feed = db.query(Feed).filter(Feed.id == feed_id).first()
        feed_info = {
            "id": feed.id,
            "title": feed.title,
            "source_type": feed.source_type,
            "categories": feed.categories
        } if feed else {}
        
        bookmark_data = {
            "id": bookmark.id,
            "notes": bookmark.notes,
            "tags": bookmark.tags,
            "created_at": bookmark.created_at.isoformat(),
            "updated_at": bookmark.updated_at.isoformat(),
            "feed": feed_info
        }
    
    return {
        "is_bookmarked": bookmark is not None,
        "feed_id": feed_id,
        "bookmark": bookmark_data
    }

@bookmark_router.get("/user/stats", response_model=dict)
def get_bookmark_stats(
    user_id: int = 1,  # In production, get from auth token
    db: Session = Depends(get_db)
):
    """Get bookmark statistics for a user."""
    total_bookmarks = db.query(Bookmark).filter(Bookmark.user_id == user_id).count()
    
    # Get tags statistics
    bookmarks_with_tags = db.query(Bookmark).filter(
        Bookmark.user_id == user_id,
        Bookmark.tags.isnot(None)
    ).all()
    
    tag_stats = {}
    for bookmark in bookmarks_with_tags:
        if bookmark.tags:
            for tag in bookmark.tags:
                tag_stats[tag] = tag_stats.get(tag, 0) + 1
    
    # Get recent bookmarks count (last 7 days)
    week_ago = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    week_ago = week_ago.replace(day=week_ago.day-7)
    
    recent_bookmarks = db.query(Bookmark).filter(
        Bookmark.user_id == user_id,
        Bookmark.created_at >= week_ago
    ).count()
    
    # Get bookmarks by source type
    bookmarks_with_feeds = db.query(Bookmark).join(Feed).filter(Bookmark.user_id == user_id).all()
    source_stats = {}
    for bookmark in bookmarks_with_feeds:
        source_type = bookmark.feed.source_type or "blog"
        source_stats[source_type] = source_stats.get(source_type, 0) + 1
    
    return {
        "total_bookmarks": total_bookmarks,
        "recent_bookmarks_7_days": recent_bookmarks,
        "tag_statistics": tag_stats,
        "tag_count": len(tag_stats),
        "source_statistics": source_stats
    }
