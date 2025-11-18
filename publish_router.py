from fastapi import APIRouter, Depends, HTTPException, status, Query, Header
from sqlalchemy.orm import Session, joinedload
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging

from database import get_db
from models import PublishedFeed, Feed, User, Blog, Slide
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
        # background_image_url=slide.background_image_url,
        # background_image_prompt=slide.background_image_prompt,
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
# Add this helper function to your publish_router.py

def get_feed_metadata(db: Session, feed: Feed, blog: Blog = None) -> Dict[str, Any]:
    """Extract metadata from feed for the meta field."""
    try:
        # Check if feed exists and has source_type
        if not feed or not hasattr(feed, 'source_type'):
            logger.warning("Feed is None or missing source_type attribute")
            return {
                "title": "Unknown",
                "original_title": "Unknown", 
                "author": "Unknown",
                "source_url": "#",
                "source_type": "unknown"
            }
        
        if feed.source_type == "youtube":
            # Get the actual YouTube video ID from the transcript
            video_id = None
            
            if feed.transcript_id:
                try:
                    # Query the transcript to get the video_id
                    from models import Transcript
                    transcript = db.query(Transcript).filter(Transcript.transcript_id == feed.transcript_id).first()
                    
                    if transcript and transcript.video_id:
                        video_id = transcript.video_id
                        logger.info(f"Found YouTube video_id: {video_id} for feed {feed.id}")
                    else:
                        logger.warning(f"No transcript or video_id found for transcript_id: {feed.transcript_id}")
                        
                except Exception as e:
                    logger.error(f"Error fetching transcript for {feed.transcript_id}: {e}")
            
            # Validate video_id format (YouTube IDs are typically 11 characters)
            if video_id and len(video_id) == 11:
                source_url = f"https://www.youtube.com/watch?v={video_id}"
                logger.info(f"Created valid YouTube URL: {source_url}")
            else:
                # If we can't get a valid video_id, use a placeholder
                source_url = "#"
                logger.warning(f"Invalid video_id format: {video_id}. Using placeholder.")
            
            return {
                "title": feed.title if feed.title else "Unknown Title",
                "original_title": feed.title if feed.title else "Unknown Title",
                "author": "YouTube Creator",
                "source_url": source_url,
                "source_type": "youtube"
            }
        else:
            # For blog sources with safe attribute access
            blog_title = blog.title if blog else "Unknown"
            blog_author = getattr(blog, 'author', 'Admin') if blog else 'Admin'
            blog_url = getattr(blog, 'url', '#') if blog else '#'
            
            return {
                "title": feed.title if feed.title else "Unknown Title",
                "original_title": blog_title,
                "author": blog_author,
                "source_url": blog_url,
                "source_type": "blog"
            }
            
    except Exception as e:
        logger.error(f"Error in get_feed_metadata: {e}")
        # Return a safe fallback
        return {
            "title": "Error",
            "original_title": "Error",
            "author": "Unknown", 
            "source_url": "#",
            "source_type": "unknown"
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
    Get all published feeds with slides included.
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
            feed_title = pf.feed.title if pf.feed else "Unknown Feed"
            blog_title = pf.feed.blog.title if pf.feed and pf.feed.blog else None
            categories = pf.feed.categories if pf.feed else []
            
            # Process slides
            slides_data = []
            if pf.feed and pf.feed.slides:
                # Sort slides by order and format them
                sorted_slides = sorted(pf.feed.slides, key=lambda x: x.order)
                slides_data = [format_slide_response(slide) for slide in sorted_slides]
            
            # Count slides
            slides_count = len(slides_data)
            
            # Get metadata - UPDATED: Pass db session as first parameter
            meta_data = get_feed_metadata(db, pf.feed, pf.feed.blog if pf.feed else None)
            
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
# @router.get("/feed/{published_feed_id}", response_model=PublishedFeedDetailResponse)
# def get_published_feed_by_id(
#     published_feed_id: int,
#     db: Session = Depends(get_db)
# ):
#     """
#     Get specific published feed by its published ID with complete feed data including slides.
#     """
#     try:
#         published_feed = db.query(PublishedFeed).options(
#             joinedload(PublishedFeed.feed).joinedload(Feed.slides),
#             joinedload(PublishedFeed.feed).joinedload(Feed.blog)
#         ).filter(PublishedFeed.id == published_feed_id).first()
        
#         if not published_feed:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail=f"Published feed with ID {published_feed_id} not found"
#             )
        
#         if not published_feed.feed:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail=f"Associated feed not found for published feed ID {published_feed_id}"
#             )
        
#         blog_title = published_feed.feed.blog.title if published_feed.feed.blog else "Unknown Blog"
        
#         return PublishedFeedDetailResponse(
#             id=published_feed.id,
#             feed_id=published_feed.feed_id,
#             admin_id=published_feed.admin_id,
#             admin_name=published_feed.admin_name,
#             published_at=published_feed.published_at,
#             is_active=published_feed.is_active,
#             feed=format_feed_response(published_feed.feed),
#             blog_title=blog_title
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error fetching published feed {published_feed_id}: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to fetch published feed"
#         )
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
        
        # Get metadata with safe access
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

# @router.get("/feed/{feed_id}/status", response_model=dict)
# def get_feed_publish_status(
#     feed_id: int,
#     db: Session = Depends(get_db)
# ):
#     """
#     Check if a feed is published and get its publish status.
#     """
#     try:
#         published_feed = db.query(PublishedFeed).options(
#             joinedload(PublishedFeed.feed)
#         ).filter(PublishedFeed.feed_id == feed_id).first()
        
#         if not published_feed:
#             return {
#                 "is_published": False,
#                 "feed_id": feed_id,
#                 "message": "Feed is not published"
#             }
        
#         return {
#             "is_published": published_feed.is_active,
#             "feed_id": feed_id,
#             "published_feed_id": published_feed.id,
#             "admin_id": published_feed.admin_id,
#             "admin_name": published_feed.admin_name,
#             "published_at": published_feed.published_at,
#             "is_active": published_feed.is_active
#         }
        
#     except Exception as e:
#         logger.error(f"Error checking publish status for feed {feed_id}: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to check publish status"
#         )

# @router.get("/stats", response_model=dict)
# def get_publishing_stats(db: Session = Depends(get_db)):
#     """
#     Get statistics about published feeds.
#     """
#     try:
#         from sqlalchemy import func
        
#         total_published = db.query(PublishedFeed).count()
#         active_published = db.query(PublishedFeed).filter(PublishedFeed.is_active == True).count()
#         inactive_published = db.query(PublishedFeed).filter(PublishedFeed.is_active == False).count()
        
#         # Count by admin
#         admin_stats = db.query(
#             PublishedFeed.admin_id,
#             PublishedFeed.admin_name,
#             func.count(PublishedFeed.id).label('feed_count')
#         ).group_by(PublishedFeed.admin_id, PublishedFeed.admin_name).all()
        
#         return {
#             "total_published": total_published,
#             "active_published": active_published,
#             "inactive_published": inactive_published,
#             "admin_stats": [
#                 {
#                     "admin_id": stat.admin_id,
#                     "admin_name": stat.admin_name,
#                     "feed_count": stat.feed_count
#                 } for stat in admin_stats
#             ]
#         }
        
#     except Exception as e:
#         logger.error(f"Error fetching publishing stats: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to fetch publishing statistics"
#         )
@router.get("/debug/feed/{feed_id}/transcript-info")
def debug_feed_transcript_info(feed_id: int, db: Session = Depends(get_db)):
    """Debug endpoint to see transcript information for a feed."""
    feed = db.query(Feed).filter(Feed.id == feed_id).first()
    if not feed:
        raise HTTPException(status_code=404, detail="Feed not found")
    
    transcript_info = None
    if feed.transcript_id:
        from models import Transcript
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