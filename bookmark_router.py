from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.exc import IntegrityError
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging
import json

from database import get_db
from models import Bookmark, User, Feed, Blog, Slide, Transcript, Source, Category, SubCategory, PublishedFeed
from schemas import BookmarkCreate, BookmarkUpdate, BookmarkResponse, BookmarkListResponse, BookmarkCreateResponse

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


def format_feed_like_published_response(db: Session, feed: Feed, bookmark: "Bookmark" = None) -> Optional[Dict[str, Any]]:
    """
    Format feed response with EXACT same fields as /publish/feeds/by-category/{category_id} API.
    Fields: id, feed_id, admin_id, admin_name, published_at, is_active, feed_title, blog_title,
            feed_categories, content_type, skills, tools, roles, slides_count, slides, meta,
            category_name, subcategory_name, category_id, subcategory_id, category_display, source
    Plus bookmark-specific fields: bookmark_id, bookmark_notes, bookmark_tags, bookmark_created_at, bookmark_updated_at
    """
    try:
        if not feed:
            return None
        
        # Get published feed info if exists
        published_feed = feed.published_feed
        
        # Get category and subcategory names
        category_name = None
        subcategory_name = None
        if feed.category_id:
            category = db.query(Category).filter(Category.id == feed.category_id).first()
            if category:
                category_name = category.name
        
        if feed.subcategory_id:
            subcategory = db.query(SubCategory).filter(SubCategory.id == feed.subcategory_id).first()
            if subcategory:
                subcategory_name = subcategory.name
        
        # Process slides
        slides_data = []
        if feed.slides:
            sorted_slides = sorted(feed.slides, key=lambda x: x.order)
            for slide in sorted_slides:
                slides_data.append({
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
                })
        
        # Get enhanced metadata
        meta_data = _get_feed_metadata_for_bookmark(db, feed)
        
        # Safely get content_type with fallback and convert to display format
        feed_content_type = getattr(feed, 'content_type', 'BLOG')
        if feed_content_type == 'BLOG':
            feed_content_type = 'Blog'
        elif feed_content_type == 'WEBINAR':
            feed_content_type = 'Webinar'
        elif feed_content_type == 'PODCAST':
            feed_content_type = 'Podcast'
        elif feed_content_type == 'VIDEO':
            feed_content_type = 'Video'
        
        # Build the response with EXACT same fields as publish feeds by-category
        response = {
            "id": published_feed.id if published_feed else feed.id,
            "feed_id": feed.id,
            "admin_id": published_feed.admin_id if published_feed else None,
            "admin_name": published_feed.admin_name if published_feed else None,
            "published_at": published_feed.published_at.isoformat() if published_feed and published_feed.published_at else None,
            "is_active": published_feed.is_active if published_feed else False,
            "feed_title": feed.title,
            "blog_title": feed.blog.title if feed.blog else None,
            "feed_categories": feed.categories or [],
            "content_type": feed_content_type,
            "skills": getattr(feed, 'skills', []) or [],
            "tools": getattr(feed, 'tools', []) or [],
            "roles": getattr(feed, 'roles', []) or [],
            "slides_count": len(slides_data),
            "slides": slides_data,
            "meta": meta_data,
            "category_name": category_name,
            "subcategory_name": subcategory_name,
            "category_id": feed.category_id,
            "subcategory_id": feed.subcategory_id,
            "category_display": f"{category_name} {{ {subcategory_name} }}" if category_name and subcategory_name else category_name,
        }
        
        # Add bookmark-specific fields if bookmark is provided
        if bookmark:
            response["bookmark_id"] = bookmark.id
            response["bookmark_notes"] = bookmark.notes
            response["bookmark_tags"] = bookmark.tags
            response["bookmark_created_at"] = bookmark.created_at.isoformat() if bookmark.created_at else None
            response["bookmark_updated_at"] = bookmark.updated_at.isoformat() if bookmark.updated_at else None
        
        return response
        
    except Exception as e:
        logger.error(f"Error formatting feed {feed.id if feed else 'None'}: {e}")
        return None


def _get_feed_metadata_for_bookmark(db: Session, feed: Feed) -> Dict[str, Any]:
    """Get enhanced metadata for a feed in bookmark context."""
    try:
        if feed.source_type == "youtube":
            # Get the actual YouTube video ID from the Transcript table
            video_id = None
            author = "YouTube Creator"
            channel_name = None
            thumbnail_url = None
            
            if feed.transcript_id:
                transcript = db.query(Transcript).filter(Transcript.transcript_id == feed.transcript_id).first()
                if transcript:
                    if hasattr(transcript, 'video_id') and transcript.video_id:
                        video_id = transcript.video_id
                    if hasattr(transcript, 'channel_name') and transcript.channel_name:
                        author = transcript.channel_name
                        channel_name = transcript.channel_name
                    elif hasattr(transcript, 'author') and transcript.author:
                        author = transcript.author
                    if hasattr(transcript, 'thumbnail_url') and transcript.thumbnail_url:
                        thumbnail_url = transcript.thumbnail_url
            
            return {
                "title": feed.title,
                "original_title": feed.title,
                "author": author,
                "source_url": f"https://www.youtube.com/watch?v={video_id}" if video_id else "#",
                "source_type": "youtube",
                "video_id": video_id,
                "channel_name": channel_name,
                "thumbnail_url": thumbnail_url or (f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg" if video_id else None),
                "transcript_id": feed.transcript_id
            }
        else:
            # Blog source
            blog = feed.blog
            return {
                "title": feed.title,
                "original_title": blog.title if blog else "Unknown",
                "author": getattr(blog, 'author', 'Admin') if blog else 'Admin',
                "source_url": getattr(blog, 'url', '#') if blog else '#',
                "source_type": "blog",
                "website": getattr(blog, 'website', '') if blog else '',
                "thumbnail_url": getattr(blog, 'thumbnail_url', None) if blog else None
            }
    except Exception as e:
        logger.error(f"Error getting metadata for feed {feed.id}: {e}")
        return {
            "title": feed.title if feed else "Unknown",
            "source_type": feed.source_type if feed else "unknown"
        }


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

@bookmark_router.get("/")
def get_all_bookmarks(
    user_id: int = 1,  # In production, get from auth token
    content_type: Optional[str] = Query(None, description="Filter by content types (comma-separated: Webinar,Blog,Podcast,Video)"),
    skills: Optional[str] = Query(None, description="Filter by skills (comma-separated)"),
    tools: Optional[str] = Query(None, description="Filter by tools (comma-separated)"),
    roles: Optional[str] = Query(None, description="Filter by roles (comma-separated)"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    tags: Optional[str] = Query(None, description="Comma-separated tags to filter by"),
    include_slides: bool = Query(True, description="Include slides data"),
    include_source_info: bool = Query(True, description="Include source information"),
    db: Session = Depends(get_db)
):
    """Get all bookmarks for a user with pagination, filtering, and complete feed data matching publish feeds by-category format."""
    try:
        # Parse comma-separated strings into lists
        def parse_comma_separated(value: Optional[str]) -> List[str]:
            if not value:
                return []
            return [item.strip() for item in value.split(',') if item.strip()]
        
        content_types_list = parse_comma_separated(content_type)
        skills_list = parse_comma_separated(skills)
        tools_list = parse_comma_separated(tools)
        roles_list = parse_comma_separated(roles)
        tag_list = parse_comma_separated(tags)
        
        # Validate content_types if provided
        valid_content_types = ['WEBINAR', 'BLOG', 'PODCAST', 'VIDEO']
        content_types_upper = []
        
        for ct in content_types_list:
            ct_upper = ct.upper()
            if ct_upper not in valid_content_types:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid content_type: {ct}. Valid types: Webinar, Blog, Podcast, Video"
                )
            content_types_upper.append(ct_upper)
        
        # Build base query
        query = db.query(Bookmark).options(
            joinedload(Bookmark.feed).joinedload(Feed.slides),
            joinedload(Bookmark.feed).joinedload(Feed.blog),
            joinedload(Bookmark.feed).joinedload(Feed.published_feed)
        ).filter(Bookmark.user_id == user_id)
        
        # Filter by tags if provided
        if tag_list:
            query = query.filter(Bookmark.tags.op('&&')(tag_list))
        
        # Get all matching bookmarks for filtering
        all_bookmarks = query.order_by(Bookmark.created_at.desc()).all()
        
        # Apply filters and collect data for statistics
        matching_items = []
        sources_map = {}
        
        for bookmark in all_bookmarks:
            feed = bookmark.feed
            if not feed:
                continue
            
            # Apply content type filter
            if content_types_upper:
                feed_content_type = getattr(feed, 'content_type', None)
                if not feed_content_type:
                    continue
                
                feed_content_type_str = None
                if hasattr(feed_content_type, 'value'):
                    feed_content_type_str = feed_content_type.value
                elif isinstance(feed_content_type, str):
                    feed_content_type_str = feed_content_type
                else:
                    feed_content_type_str = str(feed_content_type)
                
                if not feed_content_type_str:
                    continue
                
                feed_content_type_normalized = feed_content_type_str.upper()
                if feed_content_type_normalized not in content_types_upper:
                    continue
            
            # Apply skills filter
            if skills_list:
                feed_skills = feed.skills or []
                if not isinstance(feed_skills, list):
                    try:
                        if isinstance(feed_skills, str):
                            feed_skills = json.loads(feed_skills)
                        else:
                            feed_skills = []
                    except:
                        feed_skills = []
                
                feed_skills_lower = [s.lower() for s in feed_skills if isinstance(s, str)]
                requested_skills_lower = [s.lower() for s in skills_list if isinstance(s, str)]
                
                if not any(skill in feed_skills_lower for skill in requested_skills_lower):
                    continue
            
            # Apply tools filter
            if tools_list:
                feed_tools = feed.tools or []
                if not isinstance(feed_tools, list):
                    try:
                        if isinstance(feed_tools, str):
                            feed_tools = json.loads(feed_tools)
                        else:
                            feed_tools = []
                    except:
                        feed_tools = []
                
                feed_tools_lower = [t.lower() for t in feed_tools if isinstance(t, str)]
                requested_tools_lower = [t.lower() for t in tools_list if isinstance(t, str)]
                
                if not any(tool in feed_tools_lower for tool in requested_tools_lower):
                    continue
            
            # Apply roles filter
            if roles_list:
                feed_roles = feed.roles or []
                if not isinstance(feed_roles, list):
                    try:
                        if isinstance(feed_roles, str):
                            feed_roles = json.loads(feed_roles)
                        else:
                            feed_roles = []
                    except:
                        feed_roles = []
                
                feed_roles_lower = [r.lower() for r in feed_roles if isinstance(r, str)]
                requested_roles_lower = [r.lower() for r in roles_list if isinstance(r, str)]
                
                if not any(role in feed_roles_lower for role in requested_roles_lower):
                    continue
            
            # Format feed response with EXACT same fields as publish feeds by-category API
            feed_data = format_feed_like_published_response(db, feed, bookmark)
            if not feed_data:
                continue
            
            # Get source info if requested
            source_info = None
            if include_source_info:
                source_info = _get_source_for_bookmark_feed(db, feed)
                feed_data["source"] = source_info
                
                if source_info and source_info.get("id"):
                    source_id = source_info["id"]
                    if source_id not in sources_map:
                        sources_map[source_id] = {
                            "source": source_info,
                            "feed_count": 0
                        }
                    sources_map[source_id]["feed_count"] += 1
            
            matching_items.append(feed_data)
        
        # Apply pagination
        total = len(matching_items)
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_items = matching_items[start_idx:end_idx]
        
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
        
        # Get content type breakdown
        content_type_breakdown = {}
        for feed_data in matching_items:
            content_type_value = feed_data.get("content_type", "Unknown")
            if content_type_value not in content_type_breakdown:
                content_type_breakdown[content_type_value] = 0
            content_type_breakdown[content_type_value] += 1
        
        # Get skills breakdown (now at top level, not in ai_generated_content)
        skills_breakdown = {}
        for feed_data in matching_items:
            feed_skills = feed_data.get("skills", []) or []
            for skill in feed_skills:
                if skill not in skills_breakdown:
                    skills_breakdown[skill] = 0
                skills_breakdown[skill] += 1
        
        # Get tools breakdown (now at top level, not in ai_generated_content)
        tools_breakdown = {}
        for feed_data in matching_items:
            feed_tools = feed_data.get("tools", []) or []
            for tool in feed_tools:
                if tool not in tools_breakdown:
                    tools_breakdown[tool] = 0
                tools_breakdown[tool] += 1
        
        # Get roles breakdown (now at top level, not in ai_generated_content)
        roles_breakdown = {}
        for feed_data in matching_items:
            feed_roles = feed_data.get("roles", []) or []
            for role in feed_roles:
                if role not in roles_breakdown:
                    roles_breakdown[role] = 0
                roles_breakdown[role] += 1
        
        # Prepare applied filters
        applied_filters = {}
        if content_types_list:
            applied_filters["content_types"] = content_types_list
        if skills_list:
            applied_filters["skills"] = skills_list
        if tools_list:
            applied_filters["tools"] = tools_list
        if roles_list:
            applied_filters["roles"] = roles_list
        if tag_list:
            applied_filters["tags"] = tag_list
        
        response_data = {
            "items": paginated_items,
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
        
        logger.info(f"Retrieved {len(paginated_items)} bookmarks for user {user_id} with complete feed data")
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting bookmarks for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get bookmarks"
        )


def _get_source_for_bookmark_feed(db: Session, feed: Feed) -> Optional[Dict[str, Any]]:
    """Get source information for a feed in bookmark context."""
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
                "published_feed_count": 1,
                "follower_count": 0,
                "is_active": True
            }
        else:
            return {
                "id": None,
                "name": "YouTube",
                "website": "https://www.youtube.com",
                "source_type": "youtube",
                "published_feed_count": 1,
                "follower_count": 0,
                "is_active": True
            }
            
    except Exception as e:
        logger.error(f"Error getting source for feed {feed.id}: {e}")
        return None



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
