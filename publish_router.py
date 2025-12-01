# publish_router.py
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session, joinedload
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
import os
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from pydantic import BaseModel, Field  # ADD THIS IMPORT

from database import get_db
from models import PublishedFeed, Category, Feed, User, Blog, Slide, Transcript, Source, UserOnboarding
from schemas import (
    PublishFeedRequest, 
    PublishStatusResponse, 
    DeletePublishResponse,
    BulkPublishRequest,
    SlideResponse,
    UnpublishFeedRequest,
    PublishedFeedResponse
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

# ------------------ Request Models ------------------

class ContentTypeFilterRequest(BaseModel):
    content_types: List[str] = Field(..., description="List of content types to filter by")
    page: int = Field(1, ge=1, description="Page number")
    limit: int = Field(20, ge=1, le=100, description="Items per page")
    include_source_info: bool = Field(True, description="Include source information")

class AdvancedFilterRequest(BaseModel):
    content_types: Optional[List[str]] = Field(None, description="Filter by content types")
    categories: Optional[List[str]] = Field(None, description="Filter by categories")
    skills: Optional[List[str]] = Field(None, description="Filter by skills")
    tools: Optional[List[str]] = Field(None, description="Filter by tools")
    roles: Optional[List[str]] = Field(None, description="Filter by roles")
    page: int = Field(1, ge=1, description="Page number")
    limit: int = Field(20, ge=1, le=100, description="Items per page")

# ------------------ Helper Functions ------------------

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

def get_youtube_channel_info(video_id: str) -> Dict[str, Any]:
    """Get comprehensive channel information from YouTube API."""
    if not youtube_service or not video_id:
        return {
            "channel_name": "YouTube Creator", 
            "channel_id": None,
            "thumbnails": {}
        }
    
    try:
        video_response = youtube_service.videos().list(
            part="snippet",
            id=video_id
        ).execute()
        
        if not video_response.get('items'):
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
        
    except Exception as e:
        logger.error(f"Error fetching YouTube channel info for {video_id}: {e}")
        return {
            "channel_name": "YouTube Creator",
            "channel_id": None,
            "thumbnails": {}
        }

def get_feed_metadata(db: Session, feed: Feed, blog: Blog = None) -> Dict[str, Any]:
    """Extract proper metadata for feeds including YouTube channel names and correct URLs."""
    if feed.source_type == "youtube":
        transcript = db.query(Transcript).filter(Transcript.transcript_id == feed.transcript_id).first()
        
        if transcript:
            video_id = getattr(transcript, 'video_id', None) or getattr(transcript, 'youtube_video_id', feed.transcript_id)
            original_title = transcript.title if transcript else feed.title
            channel_info = get_youtube_channel_info(video_id)
            channel_name = channel_info.get("channel_name", "YouTube Creator")
            
            return {
                "title": feed.title,
                "original_title": original_title,
                "author": channel_name,
                "source_url": f"https://www.youtube.com/watch?v={video_id}",
                "source_type": "youtube",
                "channel_name": channel_name,
                "channel_id": channel_info.get("channel_id"),
                "video_id": video_id,
                "website_name": "YouTube",
                "favicon": "https://www.youtube.com/favicon.ico",
                "channel_logo": channel_info.get("thumbnails", {}).get('default', {}).get('url') if channel_info.get("thumbnails") else None,
            }
        else:
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
                "website_name": "YouTube",
                "favicon": "https://www.youtube.com/favicon.ico",
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
                "favicon": f"https://{website_name}/favicon.ico",
                "channel_name": website_name,
            }
        else:
            return {
                "title": feed.title,
                "original_title": "Unknown",
                "author": "Admin",
                "source_url": "#",
                "source_type": "blog",
                "website_name": "Unknown",
                "website": "Unknown",
            }

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

def format_published_feed_response(published_feed: PublishedFeed, db: Session) -> Optional[Dict[str, Any]]:
    """Format published feed response with all necessary data."""
    try:
        if not published_feed.feed:
            return None
            
        feed = published_feed.feed
        
        # Process slides
        slides_data = []
        if feed.slides:
            sorted_slides = sorted(feed.slides, key=lambda x: x.order)
            slides_data = [format_slide_response(slide) for slide in sorted_slides]
        
        # Get enhanced metadata
        meta_data = get_feed_metadata(db, feed, feed.blog)
        
        # Safely get content_type with fallback
        feed_content_type = getattr(feed, 'content_type', 'BLOG')
        # Convert to display format
        if feed_content_type == 'BLOG':
            feed_content_type = 'Blog'
        elif feed_content_type == 'WEBINAR':
            feed_content_type = 'Webinar'
        elif feed_content_type == 'PODCAST':
            feed_content_type = 'Podcast'
        elif feed_content_type == 'VIDEO':
            feed_content_type = 'Video'
        
        return {
            "id": published_feed.id,
            "feed_id": published_feed.feed_id,
            "admin_id": published_feed.admin_id,
            "admin_name": published_feed.admin_name,
            "published_at": published_feed.published_at.isoformat(),
            "is_active": published_feed.is_active,
            "feed_title": feed.title,
            "blog_title": feed.blog.title if feed.blog else None,
            "feed_categories": feed.categories or [],
            "content_type": feed_content_type,
            "skills": getattr(feed, 'skills', []) or [],
            "tools": getattr(feed, 'tools', []) or [],
            "roles": getattr(feed, 'roles', []) or [],
            "slides_count": len(slides_data),
            "slides": slides_data,
            "meta": meta_data
        }
        
    except Exception as e:
        logger.error(f"Error formatting published feed {published_feed.id}: {e}")
        return None

def get_source_for_feed(db: Session, feed: Feed) -> Optional[Dict[str, Any]]:
    """Get source information for a feed."""
    try:
        if feed.source_type == "blog" and feed.blog:
            # Find source by website
            source = db.query(Source).filter(
                Source.website == feed.blog.website,
                Source.source_type == "blog"
            ).first()
            
            if source:
                # Count published feeds for this source
                published_feed_count = db.query(PublishedFeed).join(Feed).join(Blog).filter(
                    PublishedFeed.is_active == True,
                    Blog.website == source.website
                ).count()
                
                return {
                    "id": source.id,
                    "name": source.name,
                    "website": source.website,
                    "source_type": source.source_type,
                    "published_feed_count": published_feed_count,
                    "follower_count": source.follower_count,
                    "is_active": source.is_active
                }
        
        elif feed.source_type == "youtube":
            # For YouTube, we might have multiple sources or a generic one
            source = db.query(Source).filter(
                Source.source_type == "youtube"
            ).first()
            
            if source:
                # Count published YouTube feeds
                published_feed_count = db.query(PublishedFeed).join(Feed).filter(
                    PublishedFeed.is_active == True,
                    Feed.source_type == "youtube"
                ).count()
                
                return {
                    "id": source.id,
                    "name": source.name,
                    "website": source.website,
                    "source_type": source.source_type,
                    "published_feed_count": published_feed_count,
                    "follower_count": source.follower_count,
                    "is_active": source.is_active
                }
        
        # Fallback: create a basic source info from feed data
        if feed.source_type == "blog" and feed.blog:
            website_name = feed.blog.website.replace("https://", "").replace("http://", "").split("/")[0]
            return {
                "id": None,
                "name": website_name,
                "website": feed.blog.website,
                "source_type": "blog",
                "published_feed_count": 1,  # Approximate
                "follower_count": 0,
                "is_active": True
            }
        else:
            return {
                "id": None,
                "name": "YouTube",
                "website": "https://www.youtube.com",
                "source_type": "youtube",
                "published_feed_count": 1,  # Approximate
                "follower_count": 0,
                "is_active": True
            }
            
    except Exception as e:
        logger.error(f"Error getting source for feed {feed.id}: {e}")
        return None

def get_content_type_breakdown(db: Session, source: Source) -> Dict[str, int]:
    """Get content type breakdown for a source."""
    try:
        if source.source_type == "blog":
            query = db.query(Feed.content_type, db.func.count(Feed.id)).join(Blog).filter(
                Blog.website == source.website
            ).group_by(Feed.content_type)
        else:
            query = db.query(Feed.content_type, db.func.count(Feed.id)).filter(
                Feed.source_type == "youtube"
            ).group_by(Feed.content_type)
        
        results = query.all()
        
        breakdown = {}
        for content_type, count in results:
            # Convert to display format
            if content_type == 'BLOG':
                display_type = 'Blog'
            elif content_type == 'WEBINAR':
                display_type = 'Webinar'
            elif content_type == 'PODCAST':
                display_type = 'Podcast'
            elif content_type == 'VIDEO':
                display_type = 'Video'
            else:
                display_type = content_type or 'Unknown'
            
            breakdown[display_type] = count
        
        return breakdown
        
    except Exception as e:
        logger.error(f"Error getting content type breakdown for source {source.id}: {e}")
        return {}

# ------------------ CONTENT TYPE FILTER ENDPOINTS ------------------

@router.post("/feeds/filter-by-type", response_model=dict)
def filter_published_feeds_by_type(
    request: ContentTypeFilterRequest,
    db: Session = Depends(get_db)
):
    """
    Filter published feeds by content types (Webinar, Blog, Podcast, Video).
    Returns feeds that match ANY of the specified content types.
    """
    try:
        # Validate content types
        valid_content_types = ['WEBINAR', 'BLOG', 'PODCAST', 'VIDEO']
        filtered_types = []
        
        for content_type in request.content_types:
            content_type_upper = content_type.upper()
            if content_type_upper in valid_content_types:
                filtered_types.append(content_type_upper)
            else:
                logger.warning(f"Invalid content_type provided: {content_type}")
        
        if not filtered_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid content types provided. Valid types: Webinar, Blog, Podcast, Video"
            )
        
        # Build query for PUBLISHED feeds
        query = db.query(PublishedFeed).options(
            joinedload(PublishedFeed.feed).joinedload(Feed.slides),
            joinedload(PublishedFeed.feed).joinedload(Feed.blog)
        ).filter(
            PublishedFeed.is_active == True
        )
        
        # Apply content type filter
        if filtered_types:
            query = query.filter(Feed.content_type.in_(filtered_types))
        
        query = query.order_by(PublishedFeed.published_at.desc())
        
        total = query.count()
        published_feeds = query.offset((request.page - 1) * request.limit).limit(request.limit).all()
        
        # Format response
        feeds_data = []
        sources_map = {}  # Track unique sources
        
        for pf in published_feeds:
            if pf.feed:
                feed_data = format_published_feed_response(pf, db)
                if feed_data:
                    # Add source information if requested
                    if request.include_source_info:
                        source_info = get_source_for_feed(db, pf.feed)
                        feed_data["source"] = source_info
                        
                        # Track source statistics
                        if source_info and source_info.get("id"):
                            source_id = source_info["id"]
                            if source_id not in sources_map:
                                sources_map[source_id] = {
                                    "source": source_info,
                                    "feed_count": 0
                                }
                            sources_map[source_id]["feed_count"] += 1
                    
                    feeds_data.append(feed_data)
        
        # Prepare content type breakdown
        content_type_breakdown = {}
        for content_type in filtered_types:
            count_query = db.query(PublishedFeed).join(Feed).filter(
                PublishedFeed.is_active == True,
                Feed.content_type == content_type
            )
            content_type_breakdown[content_type] = count_query.count()
        
        # Prepare sources summary
        sources_summary = [
            {
                "source": source_data["source"],
                "published_feeds_count": source_data["feed_count"],
                "percentage_of_total": round((source_data["feed_count"] / len(feeds_data)) * 100, 2) if feeds_data else 0
            }
            for source_data in sources_map.values()
        ]
        
        # Sort sources by feed count (most popular first)
        sources_summary.sort(key=lambda x: x["published_feeds_count"], reverse=True)
        
        return {
            "feeds": feeds_data,
            "filters": {
                "content_types": [ct.capitalize() for ct in filtered_types],
                "applied_filters": len(filtered_types)
            },
            "statistics": {
                "total_matching_feeds": total,
                "content_type_breakdown": content_type_breakdown,
                "sources_summary": sources_summary
            },
            "page": request.page,
            "limit": request.limit,
            "has_more": (request.page * request.limit) < total
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error filtering published feeds by type: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to filter published feeds by type"
        )

@router.get("/feeds/types/available")
def get_available_content_types(db: Session = Depends(get_db)):
    """
    Get available content types with counts of published feeds for each type.
    """
    try:
        content_types = ['WEBINAR', 'BLOG', 'PODCAST', 'VIDEO']
        type_breakdown = {}
        
        for content_type in content_types:
            count = db.query(PublishedFeed).join(Feed).filter(
                PublishedFeed.is_active == True,
                Feed.content_type == content_type
            ).count()
            
            type_breakdown[content_type.capitalize()] = count
        
        # Get total published feeds
        total_feeds = db.query(PublishedFeed).filter(
            PublishedFeed.is_active == True
        ).count()
        
        return {
            "available_content_types": [
                {
                    "type": content_type.capitalize(),
                    "count": count,
                    "percentage": round((count / total_feeds) * 100, 2) if total_feeds > 0 else 0
                }
                for content_type, count in type_breakdown.items()
            ],
            "total_published_feeds": total_feeds
        }
        
    except Exception as e:
        logger.error(f"Error getting available content types: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get available content types"
        )

# Alternative GET endpoint for simpler filtering
@router.get("/feeds/filter-by-type-get", response_model=dict)
def filter_published_feeds_by_type_get(
    content_types: List[str] = Query(..., description="Content types to filter by (Webinar, Blog, Podcast, Video)"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    include_source_info: bool = Query(True, description="Include source information"),
    db: Session = Depends(get_db)
):
    """
    Filter published feeds by content types (GET version).
    """
    request = ContentTypeFilterRequest(
        content_types=content_types,
        page=page,
        limit=limit,
        include_source_info=include_source_info
    )
    return filter_published_feeds_by_type(request, db)

@router.post("/feeds/advanced-filter", response_model=dict)
def advanced_filter_published_feeds(
    request: AdvancedFilterRequest,
    db: Session = Depends(get_db)
):
    """
    Advanced filtering for published feeds with multiple criteria.
    """
    try:
        query = db.query(PublishedFeed).options(
            joinedload(PublishedFeed.feed).joinedload(Feed.slides),
            joinedload(PublishedFeed.feed).joinedload(Feed.blog)
        ).filter(
            PublishedFeed.is_active == True
        )
        
        # Apply content type filter
        if request.content_types:
            valid_content_types = ['WEBINAR', 'BLOG', 'PODCAST', 'VIDEO']
            filtered_types = [ct.upper() for ct in request.content_types if ct.upper() in valid_content_types]
            if filtered_types:
                query = query.filter(Feed.content_type.in_(filtered_types))
        
        # Apply category filter
        if request.categories:
            from sqlalchemy import or_
            category_conditions = []
            for category in request.categories:
                category_conditions.append(Feed.categories.contains([category]))
            if category_conditions:
                query = query.filter(or_(*category_conditions))
        
        # Apply skills filter
        if request.skills:
            skills_conditions = []
            for skill in request.skills:
                skills_conditions.append(Feed.skills.contains([skill]))
            if skills_conditions:
                query = query.filter(or_(*skills_conditions))
        
        # Apply tools filter
        if request.tools:
            tools_conditions = []
            for tool in request.tools:
                tools_conditions.append(Feed.tools.contains([tool]))
            if tools_conditions:
                query = query.filter(or_(*tools_conditions))
        
        # Apply roles filter
        if request.roles:
            roles_conditions = []
            for role in request.roles:
                roles_conditions.append(Feed.roles.contains([role]))
            if roles_conditions:
                query = query.filter(or_(*roles_conditions))
        
        query = query.order_by(PublishedFeed.published_at.desc())
        
        total = query.count()
        published_feeds = query.offset((request.page - 1) * request.limit).limit(request.limit).all()
        
        # Format response
        feeds_data = []
        for pf in published_feeds:
            feed_data = format_published_feed_response(pf, db)
            if feed_data:
                feeds_data.append(feed_data)
        
        return {
            "feeds": feeds_data,
            "filters_applied": {
                "content_types": request.content_types or [],
                "categories": request.categories or [],
                "skills": request.skills or [],
                "tools": request.tools or [],
                "roles": request.roles or []
            },
            "total": total,
            "page": request.page,
            "limit": request.limit,
            "has_more": (request.page * request.limit) < total
        }
        
    except Exception as e:
        logger.error(f"Error in advanced filtering: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to apply advanced filters"
        )

@router.get("/feeds/with-sources", response_model=dict)
def get_published_feeds_with_sources(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    active_only: bool = Query(True, description="Show only active published feeds"),
    include_source_info: bool = Query(True, description="Include source information"),
    db: Session = Depends(get_db)
):
    """
    Get all published feeds with detailed source information including:
    - Source name, website, type
    - Number of published contents for each source
    - Follower count
    """
    try:
        query = db.query(PublishedFeed).options(
            joinedload(PublishedFeed.feed).joinedload(Feed.slides),
            joinedload(PublishedFeed.feed).joinedload(Feed.blog)
        )
        
        if active_only:
            query = query.filter(PublishedFeed.is_active == True)
        
        query = query.order_by(PublishedFeed.published_at.desc())
        
        total = query.count()
        published_feeds = query.offset((page - 1) * limit).limit(limit).all()
        
        # Format response with source information
        feeds_data = []
        sources_map = {}  # Track unique sources
        
        for pf in published_feeds:
            if pf.feed:
                feed_data = format_published_feed_response(pf, db)
                if feed_data:
                    # Add source information
                    if include_source_info:
                        source_info = get_source_for_feed(db, pf.feed)
                        feed_data["source"] = source_info
                        
                        # Track source statistics
                        if source_info and source_info.get("id"):
                            source_id = source_info["id"]
                            if source_id not in sources_map:
                                sources_map[source_id] = {
                                    "source": source_info,
                                    "feed_count": 0
                                }
                            sources_map[source_id]["feed_count"] += 1
                    
                    feeds_data.append(feed_data)
        
        # Prepare sources summary
        sources_summary = [
            {
                "source": source_data["source"],
                "published_feeds_count": source_data["feed_count"],
                "percentage_of_total": round((source_data["feed_count"] / len(feeds_data)) * 100, 2) if feeds_data else 0
            }
            for source_data in sources_map.values()
        ]
        
        # Sort sources by feed count (most popular first)
        sources_summary.sort(key=lambda x: x["published_feeds_count"], reverse=True)
        
        return {
            "feeds": feeds_data,
            "sources_summary": sources_summary,
            "total_feeds": total,
            "unique_sources": len(sources_map),
            "page": page,
            "limit": limit,
            "has_more": (page * limit) < total
        }
        
    except Exception as e:
        logger.error(f"Error fetching published feeds with sources: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch published feeds with sources"
        )

@router.get("/source/{source_id}/related-content", response_model=dict)
def get_related_content_by_source_id(
    source_id: int,
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    content_type: Optional[str] = Query(None, description="Filter by content type"),
    exclude_current_feed: Optional[int] = Query(None, description="Exclude a specific feed ID"),
    db: Session = Depends(get_db)
):
    """
    Get related published content by source ID.
    This returns all published feeds from the same source.
    """
    try:
        # Get the source
        source = db.query(Source).filter(Source.id == source_id).first()
        if not source:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Source with ID {source_id} not found"
            )
        
        # Build query for PUBLISHED feeds from this source
        if source.source_type == "blog":
            query = db.query(PublishedFeed).options(
                joinedload(PublishedFeed.feed).joinedload(Feed.slides),
                joinedload(PublishedFeed.feed).joinedload(Feed.blog)
            ).join(Feed).join(Blog).filter(
                PublishedFeed.is_active == True,
                Blog.website == source.website
            )
        else:
            query = db.query(PublishedFeed).options(
                joinedload(PublishedFeed.feed).joinedload(Feed.slides)
            ).join(Feed).filter(
                PublishedFeed.is_active == True,
                Feed.source_type == "youtube"
            )
        
        # Apply content type filter
        if content_type:
            content_type_upper = content_type.upper()
            valid_content_types = ['BLOG', 'WEBINAR', 'PODCAST', 'VIDEO']
            if content_type_upper in valid_content_types:
                query = query.filter(Feed.content_type == content_type_upper)
        
        # Exclude specific feed if provided
        if exclude_current_feed:
            query = query.filter(PublishedFeed.feed_id != exclude_current_feed)
        
        query = query.order_by(PublishedFeed.published_at.desc())
        
        total = query.count()
        published_feeds = query.offset((page - 1) * limit).limit(limit).all()
        
        # Format response
        feeds_data = []
        for pf in published_feeds:
            feed_data = format_published_feed_response(pf, db)
            if feed_data:
                feeds_data.append(feed_data)
        
        # Get source statistics
        source_stats = {
            "total_published_feeds": total,
            "content_type_breakdown": get_content_type_breakdown(db, source)
        }
        
        return {
            "source": {
                "id": source.id,
                "name": source.name,
                "website": source.website,
                "source_type": source.source_type,
                "follower_count": source.follower_count,
                "is_active": source.is_active
            },
            "related_content": feeds_data,
            "source_statistics": source_stats,
            "total": total,
            "page": page,
            "limit": limit,
            "has_more": (page * limit) < total
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching related content for source {source_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch related content for source {source_id}"
        )

# ------------------ EXISTING ENDPOINTS (Fixed) ------------------

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

@router.get("/source/{source_id}/feeds", response_model=dict)
def get_published_feeds_by_source_id(
    source_id: int,
    page: int = 1,
    limit: int = 20,
    content_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get only PUBLISHED feeds by source ID."""
    try:
        source = db.query(Source).filter(Source.id == source_id).first()
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        
        # Build query for PUBLISHED feeds only
        if source.source_type == "blog":
            query = db.query(PublishedFeed).options(
                joinedload(PublishedFeed.feed).joinedload(Feed.slides),
                joinedload(PublishedFeed.feed).joinedload(Feed.blog)
            ).join(Feed).join(Blog).filter(
                PublishedFeed.is_active == True,
                Blog.website == source.website
            )
        else:
            query = db.query(PublishedFeed).options(
                joinedload(PublishedFeed.feed).joinedload(Feed.slides)
            ).join(Feed).filter(
                PublishedFeed.is_active == True,
                Feed.source_type == "youtube"
            )
        
        # Apply content type filter
        if content_type:
            content_type_upper = content_type.upper()
            valid_content_types = ['BLOG', 'WEBINAR', 'PODCAST', 'VIDEO']
            if content_type_upper in valid_content_types:
                query = query.filter(Feed.content_type == content_type_upper)
        
        query = query.order_by(PublishedFeed.published_at.desc())
        
        total = query.count()
        published_feeds = query.offset((page - 1) * limit).limit(limit).all()
        
        # Format response
        feeds_data = []
        for pf in published_feeds:
            feed_data = format_published_feed_response(pf, db)
            if feed_data:
                feeds_data.append(feed_data)
        
        return {
            "source": {
                "id": source.id,
                "name": source.name,
                "website": source.website,
                "source_type": source.source_type,
                "published_feed_count": total
            },
            "feeds": feeds_data,
            "total": total,
            "page": page,
            "limit": limit,
            "has_more": (page * limit) < total
        }
        
    except Exception as e:
        logger.error(f"Error fetching published feeds for source {source_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch published feeds for source")

# ------------------ PUBLISHING ENDPOINTS ------------------

@router.post("/feed", response_model=PublishStatusResponse)
def publish_feed(
    request: PublishFeedRequest,
    db: Session = Depends(get_db)
):
    """Publish a feed by feed_id and admin_id."""
    try:
        admin = db.query(User).filter(
            User.id == request.admin_id,
            User.is_admin == True
        ).first()

        if not admin:
            raise HTTPException(status_code=403, detail="User is not authorized as admin")
        
        # Validate the feed exists
        feed = db.query(Feed).filter(Feed.id == request.feed_id).first()
        if not feed:
            raise HTTPException(status_code=404, detail="Feed not found")
        
        if feed.status != "ready":
            raise HTTPException(status_code=400, detail="Feed is not ready for publishing")
        
        # Check if feed is already published
        existing_published = db.query(PublishedFeed).filter(
            PublishedFeed.feed_id == request.feed_id,
            PublishedFeed.is_active == True
        ).first()
        
        if existing_published:
            raise HTTPException(status_code=400, detail="Feed is already published")
        
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
        
        logger.info(f"Feed {request.feed_id} published by admin {admin.full_name}")
        
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
        raise HTTPException(status_code=500, detail="Failed to publish feed")

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

# ------------------ ADDITIONAL UTILITY ENDPOINTS ------------------

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


# Add this to your publish_router.py after the existing endpoints

from typing import List
from pydantic import BaseModel,Field

class ContentTypeFilterRequest(BaseModel):
    content_types: List[str] = Field(..., description="List of content types to filter by")
    page: int = Field(1, ge=1, description="Page number")
    limit: int = Field(20, ge=1, le=100, description="Items per page")
    include_source_info: bool = Field(True, description="Include source information")

@router.post("/feeds/filter-by-type", response_model=dict)
def filter_published_feeds_by_type(
    request: ContentTypeFilterRequest,
    db: Session = Depends(get_db)
):
    """
    Filter published feeds by content types (Webinar, Blog, Podcast, Video).
    Returns feeds that match ANY of the specified content types.
    """
    try:
        # Validate content types
        valid_content_types = ['WEBINAR', 'BLOG', 'PODCAST', 'VIDEO']
        filtered_types = []
        
        for content_type in request.content_types:
            content_type_upper = content_type.upper()
            if content_type_upper in valid_content_types:
                filtered_types.append(content_type_upper)
            else:
                logger.warning(f"Invalid content_type provided: {content_type}")
        
        if not filtered_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid content types provided. Valid types: Webinar, Blog, Podcast, Video"
            )
        
        # Build query for PUBLISHED feeds
        query = db.query(PublishedFeed).options(
            joinedload(PublishedFeed.feed).joinedload(Feed.slides),
            joinedload(PublishedFeed.feed).joinedload(Feed.blog)
        ).filter(
            PublishedFeed.is_active == True
        )
        
        # Apply content type filter
        if filtered_types:
            query = query.filter(Feed.content_type.in_(filtered_types))
        
        query = query.order_by(PublishedFeed.published_at.desc())
        
        total = query.count()
        published_feeds = query.offset((request.page - 1) * request.limit).limit(request.limit).all()
        
        # Format response
        feeds_data = []
        sources_map = {}  # Track unique sources
        
        for pf in published_feeds:
            if pf.feed:
                feed_data = format_published_feed_response(pf, db)
                if feed_data:
                    # Add source information if requested
                    if request.include_source_info:
                        source_info = get_source_for_feed(db, pf.feed)
                        feed_data["source"] = source_info
                        
                        # Track source statistics
                        if source_info and source_info.get("id"):
                            source_id = source_info["id"]
                            if source_id not in sources_map:
                                sources_map[source_id] = {
                                    "source": source_info,
                                    "feed_count": 0
                                }
                            sources_map[source_id]["feed_count"] += 1
                    
                    feeds_data.append(feed_data)
        
        # Prepare content type breakdown
        content_type_breakdown = {}
        for content_type in filtered_types:
            count_query = db.query(PublishedFeed).join(Feed).filter(
                PublishedFeed.is_active == True,
                Feed.content_type == content_type
            )
            content_type_breakdown[content_type] = count_query.count()
        
        # Prepare sources summary
        sources_summary = [
            {
                "source": source_data["source"],
                "published_feeds_count": source_data["feed_count"],
                "percentage_of_total": round((source_data["feed_count"] / len(feeds_data)) * 100, 2) if feeds_data else 0
            }
            for source_data in sources_map.values()
        ]
        
        # Sort sources by feed count (most popular first)
        sources_summary.sort(key=lambda x: x["published_feeds_count"], reverse=True)
        
        return {
            "feeds": feeds_data,
            "filters": {
                "content_types": [ct.capitalize() for ct in filtered_types],
                "applied_filters": len(filtered_types)
            },
            "statistics": {
                "total_matching_feeds": total,
                "content_type_breakdown": content_type_breakdown,
                "sources_summary": sources_summary
            },
            "page": request.page,
            "limit": request.limit,
            "has_more": (request.page * request.limit) < total
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error filtering published feeds by type: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to filter published feeds by type"
        )

@router.get("/feeds/types/available")
def get_available_content_types(db: Session = Depends(get_db)):
    """
    Get available content types with counts of published feeds for each type.
    """
    try:
        content_types = ['WEBINAR', 'BLOG', 'PODCAST', 'VIDEO']
        type_breakdown = {}
        
        for content_type in content_types:
            count = db.query(PublishedFeed).join(Feed).filter(
                PublishedFeed.is_active == True,
                Feed.content_type == content_type
            ).count()
            
            type_breakdown[content_type.capitalize()] = count
        
        # Get total published feeds
        total_feeds = db.query(PublishedFeed).filter(
            PublishedFeed.is_active == True
        ).count()
        
        return {
            "available_content_types": [
                {
                    "type": content_type.capitalize(),
                    "count": count,
                    "percentage": round((count / total_feeds) * 100, 2) if total_feeds > 0 else 0
                }
                for content_type, count in type_breakdown.items()
            ],
            "total_published_feeds": total_feeds
        }
        
    except Exception as e:
        logger.error(f"Error getting available content types: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get available content types"
        )

# Alternative GET endpoint for simpler filtering
@router.get("/feeds/filter-by-type", response_model=dict)
def filter_published_feeds_by_type_get(
    content_types: List[str] = Query(..., description="Content types to filter by (Webinar, Blog, Podcast, Video)"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    include_source_info: bool = Query(True, description="Include source information"),
    db: Session = Depends(get_db)
):
    """
    Filter published feeds by content types (GET version).
    """
    request = ContentTypeFilterRequest(
        content_types=content_types,
        page=page,
        limit=limit,
        include_source_info=include_source_info
    )
    return filter_published_feeds_by_type(request, db)


from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from sqlalchemy import String  # and Column if needed

class AddCategoriesRequest(BaseModel):
    category_ids: List[int] = Field(..., description="List of category IDs to add")
    replace_existing: bool = Field(False, description="Whether to replace existing categories or add to them")

class CategoriesResponse(BaseModel):
    user_id: int
    categories: List[Dict[str, Any]]
    total_categories: int
    message: str
class CategoryFilterRequest(BaseModel):
    category: str = Field(..., description="Category ID or name")
    filter_type: str = Field(..., description="Filter type: topics, sources, concepts, questions, summary")
    user_id: Optional[int] = None

@router.post("/user/{user_id}/categories", response_model=CategoriesResponse)
def add_user_categories(
    user_id: int,
    request: AddCategoriesRequest,
    db: Session = Depends(get_db)
):
    """
    Add or update categories of interest for a user using category IDs.
    Path parameter `user_id` determines the user.
    """
    try:
        # Validate that all category IDs exist
        existing_categories = db.query(Category).filter(
            Category.id.in_(request.category_ids)
        ).all()

        existing_category_ids = [cat.id for cat in existing_categories]
        invalid_ids = set(request.category_ids) - set(existing_category_ids)

        if invalid_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid category IDs: {invalid_ids}"
            )

        # Find or create user's onboarding data
        onboarding = db.query(UserOnboarding).filter(
            UserOnboarding.user_id == user_id
        ).first()

        if not onboarding:
            onboarding = UserOnboarding(
                user_id=user_id,
                domains_of_interest=existing_category_ids,
                is_completed=False
            )
            db.add(onboarding)
            message = "New onboarding created with categories"
        else:
            current_category_ids = set(onboarding.domains_of_interest or [])
            if request.replace_existing:
                updated_category_ids = existing_category_ids
                message = "Categories replaced successfully"
            else:
                updated_category_ids = list(current_category_ids.union(existing_category_ids))
                message = "Categories added successfully"

            onboarding.domains_of_interest = updated_category_ids
            onboarding.updated_at = datetime.utcnow()

        db.commit()
        db.refresh(onboarding)

        # Get category details for response
        user_categories = db.query(Category).filter(
            Category.id.in_(onboarding.domains_of_interest or [])
        ).all()

        return CategoriesResponse(
            user_id=user_id,
            categories=[
                {
                    "id": cat.id,
                    "name": cat.name,
                    "description": cat.description,
                    "is_active": cat.is_active
                }
                for cat in user_categories
            ],
            total_categories=len(user_categories),
            message=message
        )

    except Exception as e:
        db.rollback()
        logger.error(f"Error updating categories for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to update user categories"
        )


@router.get("/user/{user_id}/categories", response_model=CategoriesResponse)
def get_user_categories(
    user_id: int,
    db: Session = Depends(get_db)
):
    """
    Get user's current categories of interest with full category details.
    """
    try:
        onboarding = db.query(UserOnboarding).filter(
            UserOnboarding.user_id == user_id
        ).first()

        if not onboarding or not onboarding.domains_of_interest:
            return CategoriesResponse(
                user_id=user_id,
                categories=[],
                total_categories=0,
                message="No categories found for user"
            )

        # Get full category details
        user_categories = db.query(Category).filter(
            Category.id.in_(onboarding.domains_of_interest)
        ).all()

        return CategoriesResponse(
            user_id=user_id,
            categories=[
                {
                    "id": cat.id,
                    "name": cat.name,
                    "description": cat.description,
                    "is_active": cat.is_active
                }
                for cat in user_categories
            ],
            total_categories=len(user_categories),
            message="Categories retrieved successfully"
        )

    except Exception as e:
        logger.error(f"Error fetching categories for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch user categories"
        )

@router.delete("/user/{user_id}/categories")
def remove_user_categories(
    user_id: int,
    category_ids: List[int] = Query(..., description="Category IDs to remove"),
    db: Session = Depends(get_db)
):
    """
    Remove specific categories from user's interests by category ID.
    """
    try:
        onboarding = db.query(UserOnboarding).filter(
            UserOnboarding.user_id == user_id
        ).first()

        if not onboarding:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User onboarding data not found"
            )

        current_category_ids = set(onboarding.domains_of_interest or [])
        
        # Remove specified category IDs
        updated_category_ids = list(current_category_ids - set(category_ids))
        
        onboarding.domains_of_interest = updated_category_ids
        onboarding.updated_at = datetime.utcnow()
        
        db.commit()

        return {
            "user_id": user_id,
            "removed_category_ids": category_ids,
            "current_category_ids": updated_category_ids,
            "total_categories": len(updated_category_ids),
            "message": f"Successfully removed {len(category_ids)} categories"
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error removing categories for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to remove user categories"
        )

@router.get("/categories/available")
def get_available_categories(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    search: str = Query(None, description="Search categories by name"),
    db: Session = Depends(get_db)
):
    """
    Get all available categories with their feed counts.
    """
    try:
        query = db.query(Category).filter(Category.is_active == True)
        
        if search:
            query = query.filter(Category.name.ilike(f'%{search}%'))
        
        # Get total count
        total = query.count()
        
        # Pagination
        categories = query.offset((page - 1) * limit).limit(limit).all()
        
        # Get feed counts for each category
        categories_with_counts = []
        for category in categories:
            # Count published feeds that have this category
            feed_count = db.query(PublishedFeed).join(Feed).filter(
                PublishedFeed.is_active == True,
                Feed.categories.cast(String).ilike(f'%"{category.name}"%')
            ).count()
            
            categories_with_counts.append({
                "id": category.id,
                "name": category.name,
                "description": category.description,
                "feed_count": feed_count,
                "is_active": category.is_active,
                "created_at": category.created_at
            })
        
        return {
            "categories": categories_with_counts,
            "page": page,
            "limit": limit,
            "total": total,
            "has_more": (page * limit) < total
        }
        
    except Exception as e:
        logger.error(f"Error fetching available categories: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch available categories"
        )

@router.get("/feeds/by-category/{category_id}")
def get_feeds_by_category_id(
    category_id: int,
    content_type: Optional[str] = Query(None, description="Filter by content type (Webinar, Blog, Podcast, Video)"),
    skills: Optional[List[str]] = Query(None, description="Filter by skills"),
    tools: Optional[List[str]] = Query(None, description="Filter by tools"),
    roles: Optional[List[str]] = Query(None, description="Filter by roles"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    include_source_info: bool = Query(True, description="Include source information"),
    debug: bool = Query(False, description="Enable debug mode to see filtering details"),
    db: Session = Depends(get_db)
):
    """
    Get published feeds by category ID with optional source information and filtering.
    """
    try:
        # First get the category name from ID
        category = db.query(Category).filter(Category.id == category_id).first()
        
        if not category:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Category with ID {category_id} not found"
            )

        # Validate content_type if provided
        valid_content_types = ['WEBINAR', 'BLOG', 'PODCAST', 'VIDEO']
        if content_type:
            content_type_upper = content_type.upper()
            if content_type_upper not in valid_content_types:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid content_type: {content_type}. Valid types: Webinar, Blog, Podcast, Video"
                )
        
        # Debug info
        debug_info = {}
        if debug:
            debug_info["category"] = {
                "id": category.id,
                "name": category.name,
                "description": category.description
            }
            debug_info["filters"] = {
                "content_type": content_type,
                "skills": skills,
                "tools": tools,
                "roles": roles
            }
        
        # Get all active published feeds (more efficient query)
        query = db.query(PublishedFeed).options(
            joinedload(PublishedFeed.feed).joinedload(Feed.slides),
            joinedload(PublishedFeed.feed).joinedload(Feed.blog)
        ).filter(
            PublishedFeed.is_active == True
        ).order_by(PublishedFeed.published_at.desc())

        all_feeds = query.all()
        
        if debug:
            debug_info["total_published_feeds"] = len(all_feeds)
            debug_info["sample_feed_categories"] = []
            for pf in all_feeds[:3]:  # Sample first 3
                if pf.feed and pf.feed.categories:
                    debug_info["sample_feed_categories"].append({
                        "feed_id": pf.feed.id,
                        "title": pf.feed.title,
                        "categories": pf.feed.categories,
                        "content_type": pf.feed.content_type.value if pf.feed.content_type else None
                    })
        
        # Filter feeds that have the target category in their categories list
        matching_feeds = []
        sources_map = {}  # Track unique sources for statistics
        
        for pf in all_feeds:
            if not pf.feed:
                continue
                
            feed = pf.feed
            
            # Check if category name is in the categories list
            # Handle both list of strings and potential JSON issues
            feed_categories = feed.categories or []
            if not isinstance(feed_categories, list):
                # Try to convert if it's stored as string
                try:
                    import json
                    if isinstance(feed_categories, str):
                        feed_categories = json.loads(feed_categories)
                    else:
                        feed_categories = []
                except:
                    feed_categories = []
            
            # Case-insensitive category matching
            category_match = False
            for cat in feed_categories:
                if isinstance(cat, str) and cat.lower() == category.name.lower():
                    category_match = True
                    break
            
            if not category_match:
                continue
            
            # Apply content type filter if specified
            if content_type:
                feed_content_type = feed.content_type
                if not feed_content_type or feed_content_type.value != content_type_upper:
                    continue
            
            # Apply skills filter if specified
            if skills:
                feed_skills = feed.skills or []
                if not isinstance(feed_skills, list):
                    try:
                        if isinstance(feed_skills, str):
                            feed_skills = json.loads(feed_skills)
                        else:
                            feed_skills = []
                    except:
                        feed_skills = []
                
                # Case-insensitive skills matching
                feed_skills_lower = [s.lower() for s in feed_skills if isinstance(s, str)]
                requested_skills_lower = [s.lower() for s in skills if isinstance(s, str)]
                
                if not any(skill in feed_skills_lower for skill in requested_skills_lower):
                    continue
            
            # Apply tools filter if specified
            if tools:
                feed_tools = feed.tools or []
                if not isinstance(feed_tools, list):
                    try:
                        if isinstance(feed_tools, str):
                            feed_tools = json.loads(feed_tools)
                        else:
                            feed_tools = []
                    except:
                        feed_tools = []
                
                # Case-insensitive tools matching
                feed_tools_lower = [t.lower() for t in feed_tools if isinstance(t, str)]
                requested_tools_lower = [t.lower() for t in tools if isinstance(t, str)]
                
                if not any(tool in feed_tools_lower for tool in requested_tools_lower):
                    continue
            
            # Apply roles filter if specified
            if roles:
                feed_roles = feed.roles or []
                if not isinstance(feed_roles, list):
                    try:
                        if isinstance(feed_roles, str):
                            feed_roles = json.loads(feed_roles)
                        else:
                            feed_roles = []
                    except:
                        feed_roles = []
                
                # Case-insensitive roles matching
                feed_roles_lower = [r.lower() for r in feed_roles if isinstance(r, str)]
                requested_roles_lower = [r.lower() for r in roles if isinstance(r, str)]
                
                if not any(role in feed_roles_lower for role in requested_roles_lower):
                    continue
            
            # Format feed response
            feed_data = format_published_feed_response(pf, db)
            if feed_data:
                # Add source information if requested
                if include_source_info:
                    source_info = get_source_for_feed(db, feed)
                    feed_data["source"] = source_info
                    
                    # Track source statistics
                    if source_info and source_info.get("id"):
                        source_id = source_info["id"]
                        if source_id not in sources_map:
                            sources_map[source_id] = {
                                "source": source_info,
                                "feed_count": 0
                            }
                        sources_map[source_id]["feed_count"] += 1
                
                matching_feeds.append(feed_data)
        
        # Debug: Show what feeds were found
        if debug:
            debug_info["matching_feeds_count"] = len(matching_feeds)
            debug_info["matching_feed_ids"] = [pf.feed_id for pf in all_feeds if pf.feed and pf.feed.id in [f.get("feed_id") for f in matching_feeds]]
        
        # Apply pagination
        total = len(matching_feeds)
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_feeds = matching_feeds[start_idx:end_idx]

        # Prepare sources summary
        sources_summary = [
            {
                "source": source_data["source"],
                "published_feeds_count": source_data["feed_count"],
                "percentage_of_total": round((source_data["feed_count"] / total) * 100, 2) if total > 0 else 0
            }
            for source_data in sources_map.values()
        ]
        
        # Sort sources by feed count (most popular first)
        sources_summary.sort(key=lambda x: x["published_feeds_count"], reverse=True)

        # Get content type breakdown for this category
        content_type_breakdown = {}
        for feed in matching_feeds:
            content_type_value = feed.get("content_type", "Unknown")
            if content_type_value not in content_type_breakdown:
                content_type_breakdown[content_type_value] = 0
            content_type_breakdown[content_type_value] += 1
        
        # Get skills breakdown
        skills_breakdown = {}
        for feed in matching_feeds:
            for skill in feed.get("skills", []):
                if skill not in skills_breakdown:
                    skills_breakdown[skill] = 0
                skills_breakdown[skill] += 1
        
        # Get tools breakdown
        tools_breakdown = {}
        for feed in matching_feeds:
            for tool in feed.get("tools", []):
                if tool not in tools_breakdown:
                    tools_breakdown[tool] = 0
                tools_breakdown[tool] += 1
        
        # Get roles breakdown
        roles_breakdown = {}
        for feed in matching_feeds:
            for role in feed.get("roles", []):
                if role not in roles_breakdown:
                    roles_breakdown[role] = 0
                roles_breakdown[role] += 1
        
        # Prepare applied filters
        applied_filters = {}
        if content_type:
            applied_filters["content_type"] = content_type
        if skills:
            applied_filters["skills"] = skills
        if tools:
            applied_filters["tools"] = tools
        if roles:
            applied_filters["roles"] = roles

        response_data = {
            "items": paginated_feeds,
            "category": {
                "id": category.id,
                "name": category.name,
                "description": category.description
            },
            "filters": {
                "applied": applied_filters,
                "available_counts": {
                    "content_types": content_type_breakdown,
                    "top_skills": dict(sorted(skills_breakdown.items(), key=lambda x: x[1], reverse=True)[:10]),
                    "top_tools": dict(sorted(tools_breakdown.items(), key=lambda x: x[1], reverse=True)[:10]),
                    "top_roles": dict(sorted(roles_breakdown.items(), key=lambda x: x[1], reverse=True)[:10])
                }
            },
            "statistics": {
                "total_feeds": total,
                "unique_sources": len(sources_map),
                "content_type_breakdown": content_type_breakdown,
                "sources_summary": sources_summary
            },
            "page": page,
            "limit": limit,
            "total": total,
            "has_more": end_idx < total
        }
        
        # Add debug info if requested
        if debug:
            response_data["debug"] = debug_info

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching feeds for category ID {category_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch feeds for category ID {category_id}"
        )

# @router.get("/feeds/category-debug/{category_id}")
# def debug_category_feeds(
#     category_id: int,
#     content_type: Optional[str] = Query(None, description="Filter by content type"),
#     db: Session = Depends(get_db)
# ):
#     """
#     Debug endpoint to see what's happening with category filtering.
#     """
#     try:
#         # Get category
#         category = db.query(Category).filter(Category.id == category_id).first()
#         if not category:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail=f"Category with ID {category_id} not found"
#             )
        
#         # Get all published feeds
#         published_feeds = db.query(PublishedFeed).options(
#             joinedload(PublishedFeed.feed)
#         ).filter(
#             PublishedFeed.is_active == True
#         ).all()
        
#         # Get all feeds (including unpublished)
#         all_feeds = db.query(Feed).all()
        
#         # Check feeds with the category
#         feeds_with_category = []
#         for feed in all_feeds:
#             feed_categories = feed.categories or []
#             if not isinstance(feed_categories, list):
#                 try:
#                     import json
#                     if isinstance(feed_categories, str):
#                         feed_categories = json.loads(feed_categories)
#                     else:
#                         feed_categories = []
#                 except:
#                     feed_categories = []
            
#             # Check for match
#             category_match = False
#             for cat in feed_categories:
#                 if isinstance(cat, str) and cat.lower() == category.name.lower():
#                     category_match = True
#                     break
            
#             if category_match:
#                 # Check if published
#                 is_published = any(pf.feed_id == feed.id for pf in published_feeds)
#                 feeds_with_category.append({
#                     "feed_id": feed.id,
#                     "title": feed.title,
#                     "content_type": feed.content_type.value if feed.content_type else None,
#                     "categories": feed_categories,
#                     "is_published": is_published,
#                     "published_feed_id": next((pf.id for pf in published_feeds if pf.feed_id == feed.id), None)
#                 })
        
#         # Check published feeds with the category
#         published_feeds_with_category = []
#         for pf in published_feeds:
#             if pf.feed:
#                 feed_categories = pf.feed.categories or []
#                 if not isinstance(feed_categories, list):
#                     try:
#                         import json
#                         if isinstance(feed_categories, str):
#                             feed_categories = json.loads(feed_categories)
#                         else:
#                             feed_categories = []
#                     except:
#                         feed_categories = []
                
#                 category_match = False
#                 for cat in feed_categories:
#                     if isinstance(cat, str) and cat.lower() == category.name.lower():
#                         category_match = True
#                         break
                
#                 if category_match:
#                     published_feeds_with_category.append({
#                         "published_feed_id": pf.id,
#                         "feed_id": pf.feed_id,
#                         "feed_title": pf.feed.title,
#                         "content_type": pf.feed.content_type.value if pf.feed.content_type else None,
#                         "categories": feed_categories,
#                         "published_at": pf.published_at
#                     })
        
#         return {
#             "category": {
#                 "id": category.id,
#                 "name": category.name,
#                 "description": category.description
#             },
#             "statistics": {
#                 "total_feeds": len(all_feeds),
#                 "total_published_feeds": len(published_feeds),
#                 "feeds_with_category": len(feeds_with_category),
#                 "published_feeds_with_category": len(published_feeds_with_category)
#             },
#             "feeds_with_category": feeds_with_category[:10],  # First 10 only
#             "published_feeds_with_category": published_feeds_with_category,
#             "debug_info": {
#                 "content_type_filter": content_type,
#                 "total_matching_published_feeds": len([
#                     pf for pf in published_feeds_with_category 
#                     if not content_type or (pf["content_type"] and pf["content_type"].lower() == content_type.lower())
#                 ])
#             }
#         }
        
#     except Exception as e:
#         logger.error(f"Error in category debug: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to debug category: {e}"
#         )
@router.post("/feeds/category-filter-by-id", response_model=Dict[str, Any])
def filter_feeds_within_category_by_id(
    request: CategoryFilterRequest,
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    db: Session = Depends(get_db)
):
    """
    Filter feeds within a specific category (by ID) by topics, sources, concepts, questions, or summary.
    """
    try:
        # Get category name from ID
        category = db.query(Category).filter(Category.id == int(request.category)).first()
        
        if not category:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Category with ID {request.category} not found"
            )

        # Use the existing enhanced filter but with category name
        enhanced_request = CategoryFilterRequest(
            category=category.name,
            filter_type=request.filter_type,
            user_id=request.user_id
        )
        
        return filter_feeds_within_category_enhanced(enhanced_request, page, limit, db)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error filtering category ID {request.category} by {request.filter_type}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to filter category ID {request.category} by {request.filter_type}"
        )

@router.get("/user/{user_id}/suggested-categories")
def get_suggested_categories(
    user_id: int,
    limit: int = Query(10, description="Number of suggestions to return"),
    db: Session = Depends(get_db)
):
    """
    Get suggested categories based on user's current interests and popular categories.
    """
    try:
        # Get user's current category IDs
        onboarding = db.query(UserOnboarding).filter(
            UserOnboarding.user_id == user_id
        ).first()
        
        current_category_ids = set(onboarding.domains_of_interest or []) if onboarding else set()

        # Get all active categories with feed counts
        all_categories = db.query(Category).filter(
            Category.is_active == True
        ).all()

        # Remove user's current categories from suggestions
        suggested_categories = [cat for cat in all_categories if cat.id not in current_category_ids]
        
        # Get category popularity (count of feeds per category)
        category_popularity = {}
        for category in suggested_categories:
            feed_count = db.query(PublishedFeed).join(Feed).filter(
                PublishedFeed.is_active == True,
                Feed.categories.cast(String).ilike(f'%"{category.name}"%')
            ).count()
            category_popularity[category.id] = feed_count

        # Sort by popularity and limit
        suggested_categories_sorted = sorted(
            suggested_categories, 
            key=lambda x: category_popularity.get(x.id, 0), 
            reverse=True
        )[:limit]

        return {
            "user_id": user_id,
            "suggested_categories": [
                {
                    "id": cat.id,
                    "name": cat.name,
                    "description": cat.description,
                    "popularity": category_popularity.get(cat.id, 0),
                    "feed_count": category_popularity.get(cat.id, 0)
                }
                for cat in suggested_categories_sorted
            ],
            "total_suggestions": len(suggested_categories_sorted),
            "current_categories_count": len(current_category_ids)
        }

    except Exception as e:
        logger.error(f"Error getting suggested categories for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get suggested categories"
        )



@router.get("/published-feeds/categories", response_model=Dict[str, Any])
def get_published_feed_categories(
    db: Session = Depends(get_db)
):
    """
    Get all unique categories from published feeds with their IDs and feed counts.
    """
    try:
        # Get all active published feeds with their feeds
        published_feeds = db.query(PublishedFeed).options(
            joinedload(PublishedFeed.feed)
        ).filter(
            PublishedFeed.is_active == True
        ).all()

        # Collect all unique categories from published feeds
        category_counts = {}
        
        for pf in published_feeds:
            if not pf.feed or not pf.feed.categories:
                continue
                
            feed_categories = pf.feed.categories
            if isinstance(feed_categories, list):
                for category_name in feed_categories:
                    if category_name not in category_counts:
                        category_counts[category_name] = 0
                    category_counts[category_name] += 1

        # Get category IDs and details from the Category table
        category_names = list(category_counts.keys())
        categories_with_details = []
        
        if category_names:
            # Find matching categories in the Category table
            existing_categories = db.query(Category).filter(
                Category.name.in_(category_names)
            ).all()
            
            # Create a mapping of category name to category object
            category_map = {cat.name: cat for cat in existing_categories}
            
            # Build response with category details
            for category_name, feed_count in category_counts.items():
                category_obj = category_map.get(category_name)
                
                if category_obj:
                    categories_with_details.append({
                        "id": category_obj.id,
                        "name": category_obj.name,
                        "description": category_obj.description,
                        "feed_count": feed_count,
                        "is_active": category_obj.is_active
                    })
                else:
                    # Category exists in feeds but not in Category table
                    categories_with_details.append({
                        "id": None,
                        "name": category_name,
                        "description": f"Auto-detected category: {category_name}",
                        "feed_count": feed_count,
                        "is_active": True
                    })

        # Sort by feed count (most popular first)
        categories_with_details.sort(key=lambda x: x["feed_count"], reverse=True)

        return {
            "categories": categories_with_details,
            "total_categories": len(categories_with_details),
            "total_published_feeds": len(published_feeds)
        }

    except Exception as e:
        logger.error(f"Error fetching published feed categories: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch published feed categories"
        )