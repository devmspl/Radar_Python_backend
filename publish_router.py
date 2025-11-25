from fastapi import APIRouter, Depends, HTTPException, status, Query, Header
from sqlalchemy.orm import Session, joinedload
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
import os
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from database import get_db
from models import PublishedFeed, Feed, User, Blog, Slide, Transcript
from schemas import (
    PublishFeedRequest, 
    PublishedFeedResponse, 
    PublishedFeedDetailResponse,
    PublishStatusResponse, 
    DeletePublishResponse,
    BulkPublishRequest,
    SlideResponse,
    FeedDetailResponse,
    UnpublishFeedRequest
)

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/publish", tags=["Publish Management"])

# Initialize YouTube API client
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube_service = None

if YOUTUBE_API_KEY:
    try:
        youtube_service = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        logger.info("YouTube API client initialized successfully in publish router")
    except Exception as e:
        logger.error(f"Failed to initialize YouTube API client: {e}")
        youtube_service = None
else:
    logger.warning("YOUTUBE_API_KEY not found in environment variables")

# ------------------ Helper Functions ------------------

def get_current_admin(db: Session, authorization: str = Header(...)) -> User:
    """
    Extract admin user from authorization header.
    In a real app, you'd use proper JWT authentication.
    This is a simplified version.
    """
    try:
        # Extract user ID from header (simplified - use proper auth in production)
        # Format: "Bearer {user_id}" or just "{user_id}"
        if authorization.startswith("Bearer "):
            user_id = authorization.replace("Bearer ", "").strip()
        else:
            user_id = authorization.strip()
        
        user_id = int(user_id)
        
        admin = db.query(User).filter(
            User.id == user_id, 
            User.is_admin == True
        ).first()
        
        if not admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User is not authorized as admin"
            )
        
        return admin
    except (ValueError, Exception):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header"
        )

def validate_feed_for_publishing(db: Session, feed_id: int) -> Feed:
    """Validate if a feed can be published and return the feed object."""
    feed = db.query(Feed).options(
        joinedload(Feed.slides),
        joinedload(Feed.blog)
    ).filter(Feed.id == feed_id).first()
    
    if not feed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feed with ID {feed_id} not found"
        )
    
    if feed.status != "ready":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Feed with ID {feed_id} is not ready for publishing. Status: {feed.status}"
        )
    
    if not feed.slides or len(feed.slides) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Feed with ID {feed_id} has no slides"
        )
    
    return feed

def format_slide_response(slide: Slide) -> SlideResponse:
    """Convert Slide model to SlideResponse schema."""
    return SlideResponse(
        id=slide.id,
        order=slide.order,
        title=slide.title,
        body=slide.body,
        bullets=slide.bullets or [],
        background_color=slide.background_color,
        source_refs=slide.source_refs or [],
        render_markdown=bool(slide.render_markdown),
        created_at=slide.created_at,
        updated_at=slide.updated_at
    )

def format_feed_response(feed: Feed) -> FeedDetailResponse:
    """Convert Feed model to FeedDetailResponse schema."""
    slides = sorted(feed.slides, key=lambda x: x.order) if feed.slides else []
    
    return FeedDetailResponse(
        id=feed.id,
        title=feed.title,
        categories=feed.categories or [],
        status=feed.status,
        ai_generated_content=feed.ai_generated_content,
        image_generation_enabled=feed.image_generation_enabled,
        created_at=feed.created_at,
        updated_at=feed.updated_at,
        slides=[format_slide_response(slide) for slide in slides]
    )

def get_youtube_channel_info(video_id: str) -> Dict[str, Any]:
    """Get comprehensive channel information from YouTube API."""
    if not youtube_service or not video_id:
        return {
            "channel_name": "YouTube Creator", 
            "channel_id": None,
            "thumbnails": {}
        }
    
    try:
        # First, get video details to extract channel ID
        video_response = youtube_service.videos().list(
            part="snippet",
            id=video_id
        ).execute()
        
        if not video_response.get('items'):
            logger.warning(f"No video found for ID: {video_id}")
            return {
                "channel_name": "YouTube Creator",
                "channel_id": None,
                "thumbnails": {}
            }
        
        video_snippet = video_response['items'][0]['snippet']
        channel_id = video_snippet.get('channelId')
        channel_title = video_snippet.get('channelTitle', 'YouTube Creator')
        
        if not channel_id:
            return {
                "channel_name": channel_title,
                "channel_id": None,
                "thumbnails": {}
            }
        
        # Get detailed channel information
        channel_response = youtube_service.channels().list(
            part="snippet,statistics",
            id=channel_id
        ).execute()
        
        if channel_response.get('items'):
            channel_snippet = channel_response['items'][0]['snippet']
            channel_stats = channel_response['items'][0].get('statistics', {})
            
            return {
                "channel_name": channel_snippet.get('title', channel_title),
                "channel_id": channel_id,
                "description": channel_snippet.get('description', ''),
                "custom_url": channel_snippet.get('customUrl', ''),
                "published_at": channel_snippet.get('publishedAt', ''),
                "thumbnails": channel_snippet.get('thumbnails', {}),
                "subscriber_count": channel_stats.get('subscriberCount'),
                "video_count": channel_stats.get('videoCount'),
                "view_count": channel_stats.get('viewCount')
            }
        
        return {
            "channel_name": channel_title,
            "channel_id": channel_id,
            "thumbnails": {}
        }
        
    except HttpError as e:
        logger.error(f"YouTube API error for video {video_id}: {e}")
        return {
            "channel_name": "YouTube Creator",
            "channel_id": None,
            "thumbnails": {}
        }
    except Exception as e:
        logger.error(f"Unexpected error fetching YouTube channel info for {video_id}: {e}")
        return {
            "channel_name": "YouTube Creator", 
            "channel_id": None,
            "thumbnails": {}
        }

def get_feed_metadata(db: Session, feed: Feed, blog: Blog = None) -> Dict[str, Any]:
    """Extract proper metadata for feeds including YouTube channel names and correct URLs."""
    if feed.source_type == "youtube":
        # Get the transcript to access YouTube-specific data
        transcript = db.query(Transcript).filter(Transcript.transcript_id == feed.transcript_id).first()
        
        if transcript:
            # Extract video ID from transcript data
            video_id = getattr(transcript, 'video_id', None)
            if not video_id:
                # Try to extract from transcript_id or other fields
                video_id = getattr(transcript, 'youtube_video_id', feed.transcript_id)
            
            # Get original title from transcript
            original_title = transcript.title if transcript else feed.title
            
            # Get channel information from YouTube API
            channel_info = get_youtube_channel_info(video_id)
            channel_name = channel_info.get("channel_name", "YouTube Creator")
            
            # Construct proper YouTube URL
            source_url = f"https://www.youtube.com/watch?v={video_id}"
            
            return {
                "title": feed.title,
                "original_title": original_title,
                "author": channel_name,
                "source_url": source_url,
                "source_type": "youtube",
                "channel_name": channel_name,
                "channel_id": channel_info.get("channel_id"),
                "video_id": video_id,
                # "channel_info": channel_info,  # Include full channel info for frontend
                # Enhanced fields
                "website_name": "YouTube",
                "favicon": "https://www.youtube.com/favicon.ico",
                "channel_logo": channel_info.get("thumbnails", {}).get('default', {}).get('url') if channel_info.get("thumbnails") else None,
                "channel_description": channel_info.get("description", ""),
                "channel_custom_url": channel_info.get("custom_url", ""),
                "subscriber_count": channel_info.get("subscriber_count"),
                "video_count": channel_info.get("video_count"),
                "view_count": channel_info.get("view_count"),
                "published_at": channel_info.get("published_at")
            }
        else:
            # Fallback if transcript not found
            video_id = feed.transcript_id
            channel_info = get_youtube_channel_info(video_id)
            channel_name = channel_info.get("channel_name", "YouTube Creator")
            
            return {
                "title": feed.title,
                "original_title": feed.title,
                "author": channel_name,
                "source_url": f"https://www.youtube.com/watch?v={video_id}",
                "source_type": "youtube",
                "channel_name": channel_name,
                "channel_id": channel_info.get("channel_id"),
                "video_id": video_id,
                "channel_info": channel_info,
                # Enhanced fields
                "website_name": "YouTube",
                "favicon": "https://www.youtube.com/favicon.ico",
                "channel_logo": channel_info.get("thumbnails", {}).get('default', {}).get('url') if channel_info.get("thumbnails") else None,
                "channel_description": channel_info.get("description", ""),
                "channel_custom_url": channel_info.get("custom_url", ""),
                "subscriber_count": channel_info.get("subscriber_count"),
                "video_count": channel_info.get("video_count"),
                "view_count": channel_info.get("view_count"),
                "published_at": channel_info.get("published_at")
            }
    
    else:  # blog source type
        blog = feed.blog
        if blog:
            website_name = blog.website.replace("https://", "").replace("http://", "").split("/")[0]
            author = getattr(blog, 'author', 'Admin') or 'Admin'
            
            return {
                "title": feed.title,
                "original_title": blog.title,
                "author": author,
                "source_url": blog.url,
                "source_type": "blog",
                "website_name": website_name,
                "website": blog.website,
                # Enhanced fields
                "favicon": f"https://{website_name}/favicon.ico",
                "channel_name": website_name,
                "channel_logo": f"https://{website_name}/favicon.ico",
                "channel_description": "",
                "channel_custom_url": "",
                "subscriber_count": None,
                "video_count": None,
                "view_count": None,
                "published_at": getattr(blog, 'published_at', None),
                "video_id": None,
                "channel_id": None
            }
        else:
            # Fallback if blog not found
            return {
                "title": feed.title,
                "original_title": "Unknown",
                "author": "Admin",
                "source_url": "#",
                "source_type": "blog",
                "website_name": "Unknown",
                "website": "Unknown",
                # Enhanced fields
                "favicon": None,
                "channel_name": "Unknown",
                "channel_logo": None,
                "channel_description": "",
                "channel_custom_url": "",
                "subscriber_count": None,
                "video_count": None,
                "view_count": None,
                "published_at": None,
                "video_id": None,
                "channel_id": None
            }

# ------------------ API Endpoints ------------------

@router.post("/feed", response_model=PublishStatusResponse)
def publish_feed(
    request: PublishFeedRequest,
    db: Session = Depends(get_db)
):
    """
    Publish a feed by feed_id and admin_id (provided in request body).
    No Authorization header required.
    """
    try:
        # Validate admin from request.admin_id
        admin = db.query(User).filter(
            User.id == request.admin_id,
            User.is_admin == True
        ).first()

        if not admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User is not authorized as admin"
            )
        
        # Validate the feed exists and is ready for publishing
        feed = validate_feed_for_publishing(db, request.feed_id)
        
        # Check if feed is already published
        existing_published = db.query(PublishedFeed).filter(
            PublishedFeed.feed_id == request.feed_id,
            PublishedFeed.is_active == True
        ).first()
        
        if existing_published:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Feed with ID {request.feed_id} is already published"
            )
        
        # Create published feed entry
        published_feed = PublishedFeed(
            feed_id=request.feed_id,
            admin_id=admin.id,
            admin_name=admin.full_name,
            published_at=datetime.utcnow(),
            is_active=True
        )
        
        db.add(published_feed)
        db.commit()
        db.refresh(published_feed)
        
        logger.info(f"Feed {request.feed_id} published by admin {admin.full_name} (ID: {admin.id})")
        
        return PublishStatusResponse(
            message="Feed published successfully",
            published_feed_id=published_feed.id,
            feed_id=request.feed_id,
            admin_id=admin.id,
            admin_name=admin.full_name,
            published_at=published_feed.published_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error publishing feed {request.feed_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to publish feed"
        )

@router.post("/feeds/bulk", response_model=dict)
def bulk_publish_feeds(
    request: BulkPublishRequest,
    db: Session = Depends(get_db)
):
    """
    Publish multiple feeds at once by their IDs.
    Admin ID is provided in the request body.
    """
    try:
        # Validate admin from request.admin_id
        admin = db.query(User).filter(
            User.id == request.admin_id,
            User.is_admin == True
        ).first()

        if not admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User is not authorized as admin"
            )

        results = {
            "successful": [],
            "failed": [],
            "already_published": []
        }

        for feed_id in request.feed_ids:
            try:
                # Validate feed exists and is ready
                feed = validate_feed_for_publishing(db, feed_id)

                # Check if a PublishedFeed exists for this feed
                existing_published = db.query(PublishedFeed).filter(
                    PublishedFeed.feed_id == feed_id
                ).first()

                if existing_published:
                    if existing_published.is_active:
                        results["already_published"].append(feed_id)
                        continue
                    # Reactivate existing record if inactive
                    existing_published.is_active = True
                    existing_published.admin_id = admin.id
                    existing_published.admin_name = admin.full_name
                    existing_published.published_at = datetime.utcnow()
                    db.commit()
                    results["successful"].append(feed_id)
                    continue

                # Create new PublishedFeed
                published_feed = PublishedFeed(
                    feed_id=feed_id,
                    admin_id=admin.id,
                    admin_name=admin.full_name,
                    published_at=datetime.utcnow(),
                    is_active=True
                )
                db.add(published_feed)
                db.commit()
                results["successful"].append(feed_id)

            except HTTPException as e:
                results["failed"].append({
                    "feed_id": feed_id,
                    "error": e.detail
                })
            except Exception as e:
                results["failed"].append({
                    "feed_id": feed_id,
                    "error": str(e)
                })

        logger.info(
            f"Bulk publish completed by admin {admin.full_name}. "
            f"Successful: {len(results['successful'])}, "
            f"Failed: {len(results['failed'])}, "
            f"Already published: {len(results['already_published'])}"
        )

        return {
            "message": "Bulk publish operation completed",
            "admin_id": admin.id,
            "admin_name": admin.full_name,
            "results": results
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error in bulk publish operation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process bulk publish request"
        )

@router.get("/feeds", response_model=List[PublishedFeedResponse])
def get_published_feeds(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    active_only: bool = Query(True, description="Show only active published feeds"),
    db: Session = Depends(get_db)
):
    """
    Get all published feeds with slides included and enhanced metadata.
    """
    try:
        # Query with proper joins to load all related data
        query = db.query(PublishedFeed).options(
            joinedload(PublishedFeed.feed).joinedload(Feed.slides),
            joinedload(PublishedFeed.feed).joinedload(Feed.blog)
        )
        
        # Filter by active status
        if active_only:
            query = query.filter(PublishedFeed.is_active == True)
        
        # Order by most recent first
        query = query.order_by(PublishedFeed.published_at.desc())
        
        # Pagination
        published_feeds = query.offset((page - 1) * limit).limit(limit).all()
        
        response_data = []
        for pf in published_feeds:
            # Safely get feed data
            if not pf.feed:
                continue  # Skip if no feed associated
                
            feed_title = pf.feed.title
            blog_title = pf.feed.blog.title if pf.feed.blog else None
            categories = pf.feed.categories or []
            
            # Process slides
            slides_data = []
            if pf.feed.slides:
                # Sort slides by order and format them
                sorted_slides = sorted(pf.feed.slides, key=lambda x: x.order)
                slides_data = [format_slide_response(slide) for slide in sorted_slides]
            
            # Count slides
            slides_count = len(slides_data)
            
            # Get enhanced metadata with channel info, logos, and website data
            meta_data = get_feed_metadata(db, pf.feed, pf.feed.blog)
            
            response_data.append(PublishedFeedResponse(
                id=pf.id,
                feed_id=pf.feed_id,
                admin_id=pf.admin_id,
                admin_name=pf.admin_name,
                published_at=pf.published_at,
                is_active=pf.is_active,
                feed_title=feed_title,
                blog_title=blog_title,
                feed_categories=categories,
                slides_count=slides_count,
                slides=slides_data if slides_data else None,
                meta=meta_data
            ))
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error fetching published feeds: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch published feeds"
        )

@router.get("/feed/{published_feed_id}/slides/with-metadata", response_model=Dict[str, Any])
def get_published_feed_slides_with_metadata(
    published_feed_id: int,
    db: Session = Depends(get_db)
):
    """
    Get slides for a specific published feed with comprehensive metadata.
    """
    try:
        published_feed = db.query(PublishedFeed).options(
            joinedload(PublishedFeed.feed).joinedload(Feed.slides),
            joinedload(PublishedFeed.feed).joinedload(Feed.blog)
        ).filter(PublishedFeed.id == published_feed_id).first()
        
        if not published_feed:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Published feed with ID {published_feed_id} not found"
            )
        
        # Check if feed exists and has the necessary attributes
        if not published_feed.feed:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Associated feed not found for published feed ID {published_feed_id}"
            )
        
        # Safely get feed attributes
        feed = published_feed.feed
        feed_title = feed.title if feed else "Unknown Feed"
        
        # Process slides
        slides = sorted(feed.slides, key=lambda x: x.order) if feed.slides else []
        formatted_slides = [format_slide_response(slide) for slide in slides]
        
        # Get metadata with enhanced information
        meta_data = get_feed_metadata(db, feed, feed.blog if feed else None)
        
        return {
            "published_feed_id": published_feed_id,
            "feed_id": feed.id,
            "feed_title": feed_title,
            "slides": formatted_slides,
            "meta": meta_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching slides with metadata for published feed {published_feed_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch slides"
        )

@router.delete("/feed", response_model=DeletePublishResponse)
def unpublish_feed(
    request: UnpublishFeedRequest,
    db: Session = Depends(get_db)
):
    """
    Unpublish a feed (soft delete) using feed_id and admin_id from request body.
    """
    try:
        # Validate admin from request.admin_id
        admin = db.query(User).filter(
            User.id == request.admin_id,
            User.is_admin == True
        ).first()

        if not admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User is not authorized as admin"
            )

        # Find the active published feed
        published_feed = db.query(PublishedFeed).filter(
            PublishedFeed.feed_id == request.feed_id,
            PublishedFeed.is_active == True
        ).first()

        if not published_feed:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No active published feed found for feed ID {request.feed_id}"
            )

        # Soft delete
        published_feed.is_active = False
        db.commit()

        logger.info(f"Feed {request.feed_id} unpublished by admin {admin.full_name}")

        return DeletePublishResponse(
            message="Feed unpublished successfully",
            deleted_feed_id=request.feed_id,
            admin_id=admin.id,
            admin_name=admin.full_name
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error unpublishing feed {request.feed_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to unpublish feed"
        )

@router.get("/debug/feed/{feed_id}/transcript-info")
def debug_feed_transcript_info(feed_id: int, db: Session = Depends(get_db)):
    """Debug endpoint to see transcript information for a feed."""
    feed = db.query(Feed).filter(Feed.id == feed_id).first()
    if not feed:
        raise HTTPException(status_code=404, detail="Feed not found")
    
    transcript_info = None
    if feed.transcript_id:
        transcript = db.query(Transcript).filter(Transcript.transcript_id == feed.transcript_id).first()
        if transcript:
            transcript_info = {
                "transcript_id": transcript.transcript_id,
                "video_id": transcript.video_id,
                "title": transcript.title,
                "all_attributes": {column.name: getattr(transcript, column.name) for column in transcript.__table__.columns}
            }
    
    return {
        "feed_id": feed.id,
        "feed_title": feed.title,
        "transcript_id": feed.transcript_id,
        "source_type": feed.source_type,
        "transcript_info": transcript_info
    }

from sqlalchemy import or_, and_,String

@router.get("/personalized", response_model=Dict[str, Any])
def get_personalized_feeds(
    user_id: int = Query(..., description="Logged-in user ID"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    db: Session = Depends(get_db)
):
    """
    Get personalized published feeds based on user's followed topics and interests.
    """
    try:
        # Get user's followed topics
        from models import UserTopicFollow, UserSourceFollow
        
        # Get user's followed topics
        followed_topics = db.query(UserTopicFollow).options(
            joinedload(UserTopicFollow.topic)
        ).filter(UserTopicFollow.user_id == user_id).all()
        
        # Get user's followed sources
        followed_sources = db.query(UserSourceFollow).options(
            joinedload(UserSourceFollow.source)
        ).filter(UserSourceFollow.user_id == user_id).all()
        
        # Debug logging
        logger.info(f"User {user_id} follows {len(followed_topics)} topics and {len(followed_sources)} sources")
        
        # Log followed topic names
        topic_names = [ft.topic.name for ft in followed_topics if ft.topic]
        logger.info(f"Followed topics: {topic_names}")
        
        # Start with base query for published feeds with proper JOIN
        query = db.query(PublishedFeed).options(
            joinedload(PublishedFeed.feed).joinedload(Feed.slides),
            joinedload(PublishedFeed.feed).joinedload(Feed.blog)
        ).join(Feed, PublishedFeed.feed_id == Feed.id).filter(
            PublishedFeed.is_active == True
        )
        
        # If user has followed topics or sources, personalize the feed
        if followed_topics or followed_sources:
            # Build conditions for personalization
            conditions = []
            
            # Add conditions for followed topics - FIXED: Check if array contains the topic name
            for topic_follow in followed_topics:
                if topic_follow.topic:
                    topic_name = topic_follow.topic.name
                    # FIX: Use string containment for JSON array
                    # This checks if the categories array contains the topic name as a string
                    conditions.append(Feed.categories.cast(String).ilike(f'%"{topic_name}"%'))
                    logger.info(f"Adding topic condition for: {topic_name}")
            
            # Add conditions for followed sources
            for source_follow in followed_sources:
                if source_follow.source:
                    source = source_follow.source
                    if source.source_type == "blog":
                        # Need to join with Blog for website conditions
                        query = query.join(Blog, Feed.blog_id == Blog.id)
                        conditions.append(Blog.website.ilike(f"%{source.website}%"))
                        logger.info(f"Adding blog source condition for: {source.website}")
                    elif source.source_type == "youtube":
                        conditions.append(Feed.source_type == "youtube")
                        logger.info(f"Adding YouTube source condition")
            
            if conditions:
                query = query.filter(or_(*conditions))
                logger.info(f"Applied {len(conditions)} personalization conditions")
            else:
                # If no valid conditions, add a dummy condition to avoid returning all feeds
                conditions.append(Feed.id == -1)  # This will return no results
                logger.info("No valid conditions, applying dummy condition")
        
        # Order by published date (newest first)
        query = query.order_by(PublishedFeed.published_at.desc())
        
        # Log the final query
        logger.info(f"Final query will fetch published feeds with personalization")
        
        # Pagination
        total = query.count()
        logger.info(f"Total matching feeds: {total}")
        
        published_feeds = query.offset((page - 1) * limit).limit(limit).all()
        logger.info(f"Fetched {len(published_feeds)} feeds for page {page}")
        
        # Format response
        response_data = []
        for pf in published_feeds:
            if not pf.feed:
                continue
                
            feed_title = pf.feed.title
            blog_title = pf.feed.blog.title if pf.feed.blog else None
            categories = pf.feed.categories or []
            
            # Process slides
            slides_data = []
            if pf.feed.slides:
                sorted_slides = sorted(pf.feed.slides, key=lambda x: x.order)
                slides_data = [format_slide_response(slide) for slide in sorted_slides]
            
            slides_count = len(slides_data)
            
            # Get enhanced metadata
            meta_data = get_feed_metadata(db, pf.feed, pf.feed.blog)
            
            response_data.append({
                "id": pf.id,
                "feed_id": pf.feed_id,
                "admin_id": pf.admin_id,
                "admin_name": pf.admin_name,
                "published_at": pf.published_at,
                "is_active": pf.is_active,
                "feed_title": feed_title,
                "blog_title": blog_title,
                "feed_categories": categories,
                "slides_count": slides_count,
                "slides": slides_data if slides_data else None,
                "meta": meta_data
            })
        
        return {
            "items": response_data,
            "page": page,
            "limit": limit,
            "total": total,
            "has_more": (page * limit) < total,
            "personalization_info": {
                "followed_topics_count": len(followed_topics),
                "followed_sources_count": len(followed_sources),
                "is_personalized": len(followed_topics) > 0 or len(followed_sources) > 0,
                "followed_topic_names": topic_names,
                "followed_source_names": [fs.source.name for fs in followed_sources if fs.source]
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching personalized feeds for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch personalized feeds"
        )


from sqlalchemy import or_, and_, String, func, distinct
from typing import List

# Add these new endpoints to your existing router

@router.get("/categories", response_model=Dict[str, Any])
def get_published_feed_categories(
    db: Session = Depends(get_db),
    active_only: bool = Query(True, description="Only include categories from active published feeds")
):
    """
    Get all unique categories from published feeds with counts.
    Returns both category names and their frequencies.
    """
    try:
        # Base query for published feeds
        query = db.query(PublishedFeed).join(Feed, PublishedFeed.feed_id == Feed.id)
        
        if active_only:
            query = query.filter(PublishedFeed.is_active == True)
        
        # Get all published feeds with their categories
        published_feeds = query.options(
            joinedload(PublishedFeed.feed)
        ).all()
        
        # Extract and count categories
        category_count = {}
        all_categories = set()
        
        for pf in published_feeds:
            if pf.feed and pf.feed.categories:
                for category in pf.feed.categories:
                    if category:  # Skip empty categories
                        all_categories.add(category)
                        category_count[category] = category_count.get(category, 0) + 1
        
        # Convert to sorted list of categories with counts
        sorted_categories = sorted([
            {
                "name": category,
                "count": category_count[category],
                "id": f"cat_{idx}"  # Generate simple ID for frontend use
            }
            for idx, category in enumerate(sorted(all_categories))
        ], key=lambda x: x["count"], reverse=True)
        
        return {
            "categories": sorted_categories,
            "total_categories": len(sorted_categories),
            "total_feeds": len(published_feeds),
            "active_only": active_only
        }
        
    except Exception as e:
        logger.error(f"Error fetching published feed categories: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch categories"
        )

@router.get("/feeds/by-category", response_model=Dict[str, Any])
def get_published_feeds_by_category(
    category: str = Query(..., description="Category name to filter by"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    active_only: bool = Query(True, description="Show only active published feeds"),
    db: Session = Depends(get_db)
):
    """
    Get published feeds by category name with enhanced metadata.
    Uses case-insensitive partial matching for category names.
    """
    try:
        # Base query with proper joins
        query = db.query(PublishedFeed).options(
            joinedload(PublishedFeed.feed).joinedload(Feed.slides),
            joinedload(PublishedFeed.feed).joinedload(Feed.blog)
        ).join(Feed, PublishedFeed.feed_id == Feed.id)
        
        # Filter by active status
        if active_only:
            query = query.filter(PublishedFeed.is_active == True)
        
        # Filter by category - using JSON array containment with string matching
        # This handles the case where categories is a JSON array of strings
        category_filter = Feed.categories.cast(String).ilike(f'%"{category}"%')
        query = query.filter(category_filter)
        
        # Order by most recent first
        query = query.order_by(PublishedFeed.published_at.desc())
        
        # Get total count for pagination
        total = query.count()
        
        # Pagination
        published_feeds = query.offset((page - 1) * limit).limit(limit).all()
        
        response_data = []
        for pf in published_feeds:
            if not pf.feed:
                continue
                
            feed_title = pf.feed.title
            blog_title = pf.feed.blog.title if pf.feed.blog else None
            categories = pf.feed.categories or []
            
            # Process slides
            slides_data = []
            if pf.feed.slides:
                sorted_slides = sorted(pf.feed.slides, key=lambda x: x.order)
                slides_data = [format_slide_response(slide) for slide in sorted_slides]
            
            slides_count = len(slides_data)
            
            # Get enhanced metadata
            meta_data = get_feed_metadata(db, pf.feed, pf.feed.blog)
            
            response_data.append({
                "id": pf.id,
                "feed_id": pf.feed_id,
                "admin_id": pf.admin_id,
                "admin_name": pf.admin_name,
                "published_at": pf.published_at,
                "is_active": pf.is_active,
                "feed_title": feed_title,
                "blog_title": blog_title,
                "feed_categories": categories,
                "slides_count": slides_count,
                "slides": slides_data if slides_data else None,
                "meta": meta_data
            })
        
        return {
            "items": response_data,
            "page": page,
            "limit": limit,
            "total": total,
            "has_more": (page * limit) < total,
            "category": category,
            "active_only": active_only
        }
        
    except Exception as e:
        logger.error(f"Error fetching published feeds by category '{category}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch feeds for category '{category}'"
        )

@router.get("/feeds/by-category-id", response_model=Dict[str, Any])
def get_published_feeds_by_category_id(
    category_id: str = Query(..., description="Category ID from the categories list"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    active_only: bool = Query(True, description="Show only active published feeds"),
    db: Session = Depends(get_db)
):
    """
    Get published feeds by category ID.
    The category ID should be from the categories list endpoint (format: 'cat_1', 'cat_2', etc.)
    """
    try:
        # First, get all categories to map ID to name
        categories_response = get_published_feed_categories(db, active_only)
        categories_map = {cat["id"]: cat["name"] for cat in categories_response["categories"]}
        
        # Find category name by ID
        category_name = categories_map.get(category_id)
        if not category_name:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Category with ID '{category_id}' not found"
            )
        
        # Now use the existing category name endpoint logic
        return get_published_feeds_by_category(category_name, page, limit, active_only, db)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching published feeds by category ID '{category_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch feeds for category ID '{category_id}'"
        )

# Also update the schemas.py file to include the response models if needed