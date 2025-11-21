from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import or_, String, func
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import json 
import os
import re
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from openai import OpenAI

from database import get_db
from models import Feed, Blog, Topic, Source, UserTopicFollow, UserSourceFollow, Transcript

router = APIRouter(prefix="/search", tags=["Feed Search"])

# Configure logging
logger = logging.getLogger(__name__)

# Initialize API clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# YouTube API client
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube_service = None

if YOUTUBE_API_KEY:
    try:
        youtube_service = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        logger.info("YouTube API client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize YouTube API client: {e}")
        youtube_service = None
else:
    logger.warning("YOUTUBE_API_KEY not found in environment variables")

# ------------------ Core Metadata Functions ------------------
import requests
from urllib.parse import urlparse
import base64

# Add these helper functions for images and favicons
def get_youtube_channel_thumbnail(channel_id: str) -> str:
    """Get YouTube channel thumbnail URL."""
    if not channel_id or not youtube_service:
        return None
    
    try:
        channel_response = youtube_service.channels().list(
            part="snippet",
            id=channel_id
        ).execute()
        
        if channel_response.get('items'):
            channel_snippet = channel_response['items'][0]['snippet']
            thumbnails = channel_snippet.get('thumbnails', {})
            # Get the highest quality thumbnail available
            if thumbnails.get('high'):
                return thumbnails['high']['url']
            elif thumbnails.get('medium'):
                return thumbnails['medium']['url']
            elif thumbnails.get('default'):
                return thumbnails['default']['url']
    except Exception as e:
        logger.error(f"Error fetching YouTube channel thumbnail: {e}")
    
    return None

def get_website_favicon(website_url: str) -> str:
    """Get website favicon URL."""
    try:
        if not website_url or website_url in ["Unknown", "#"]:
            return None
        
        # Try common favicon locations
        parsed_url = urlparse(website_url)
        base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        favicon_urls = [
            f"{base_domain}/favicon.ico",
            f"{base_domain}/favicon.png",
            f"{base_domain}/apple-touch-icon.png"
        ]
        
        # Test each URL to see which one exists
        for favicon_url in favicon_urls:
            try:
                response = requests.head(favicon_url, timeout=5)
                if response.status_code == 200:
                    return favicon_url
            except:
                continue
        
        # Fallback to Google favicon service
        domain = parsed_url.netloc
        return f"https://www.google.com/s2/favicons?domain={domain}&sz=64"
        
    except Exception as e:
        logger.error(f"Error fetching favicon for {website_url}: {e}")
        return None


def get_youtube_channel_info(video_id: str) -> Dict[str, str]:
    """Get channel information from YouTube API using video ID."""
    if not youtube_service or not video_id:
        return {"channel_name": "YouTube Creator", "channel_id": None}
    
    try:
        # Get video details to extract channel ID
        video_response = youtube_service.videos().list(
            part="snippet",
            id=video_id
        ).execute()
        
        if not video_response.get('items'):
            logger.warning(f"No video found for ID: {video_id}")
            return {"channel_name": "YouTube Creator", "channel_id": None}
        
        video_snippet = video_response['items'][0]['snippet']
        channel_id = video_snippet.get('channelId')
        channel_title = video_snippet.get('channelTitle', 'YouTube Creator')
        
        channel_thumbnail = None
        if channel_id:
            # Get detailed channel information with thumbnails
            channel_response = youtube_service.channels().list(
                part="snippet",
                id=channel_id
            ).execute()
            
            if channel_response.get('items'):
                channel_snippet = channel_response['items'][0]['snippet']
                channel_title = channel_snippet.get('title', channel_title)
                thumbnails = channel_snippet.get('thumbnails', {})
                
                # Get the highest quality thumbnail available
                if thumbnails.get('high'):
                    channel_thumbnail = thumbnails['high']['url']
                elif thumbnails.get('medium'):
                    channel_thumbnail = thumbnails['medium']['url']
                elif thumbnails.get('default'):
                    channel_thumbnail = thumbnails['default']['url']
        
        return {
            "channel_name": channel_title,
            "channel_id": channel_id,
            "channel_thumbnail": channel_thumbnail
        }
        
    except HttpError as e:
        logger.error(f"YouTube API error for video {video_id}: {e}")
        return {"channel_name": "YouTube Creator", "channel_id": None}
    except Exception as e:
        logger.error(f"Unexpected error fetching YouTube channel info for {video_id}: {e}")
        return {"channel_name": "YouTube Creator", "channel_id": None}


def extract_clean_source_name(website_url: str) -> str:
    """Extract clean source name using OpenAI for better naming."""
    try:
        if not website_url or website_url in ["Unknown", "#"]:
            return "Unknown Source"
        
        # Basic cleaning
        clean_url = website_url.replace("https://", "").replace("http://", "").split("/")[0]
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are a naming expert. Extract a clean, professional source name from a domain.
                    Return ONLY the name, no explanations.
                    Examples:
                    - "blog.hubspot.com" -> "HubSpot Blog"
                    - "news.ycombinator.com" -> "Hacker News" 
                    - "techcrunch.com" -> "TechCrunch"
                    - "medium.com" -> "Medium"
                    - "towardsdatascience.com" -> "Towards Data Science"
                    - If unsure, return the main domain name without TLD."""
                },
                {
                    "role": "user",
                    "content": f"Extract a clean source name from: {clean_url}"
                }
            ],
            temperature=0.1,
            max_tokens=50
        )
        
        source_name = response.choices[0].message.content.strip()
        source_name = source_name.replace('"', '').replace("'", "").strip()
        
        logger.info(f"Extracted source name: '{source_name}' from '{website_url}'")
        return source_name
        
    except Exception as e:
        logger.error(f"OpenAI source name extraction failed for {website_url}: {e}")
        # Fallback to basic domain extraction
        clean_url = website_url.replace("https://", "").replace("http://", "").split("/")[0]
        return clean_url

def get_feed_metadata(feed: Feed, db: Session) -> Dict[str, Any]:
    """Extract proper metadata for feeds including images and favicons."""
    if feed.source_type == "youtube":
        # Get the transcript to access YouTube-specific data
        transcript = db.query(Transcript).filter(Transcript.transcript_id == feed.transcript_id).first()
        
        if transcript:
            # Extract video ID from transcript data
            video_id = getattr(transcript, 'video_id', None)
            if not video_id:
                video_id = feed.transcript_id
            
            # Get original title from transcript
            original_title = transcript.title if transcript else feed.title
            
            # Get channel information from YouTube API
            channel_info = get_youtube_channel_info(video_id)
            channel_name = channel_info.get("channel_name", "YouTube Creator")
            channel_thumbnail = channel_info.get("channel_thumbnail")
            
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
                "channel_thumbnail": channel_thumbnail,
                "favicon": "https://www.google.com/s2/favicons?domain=youtube.com&sz=64"
            }
        else:
            # Fallback if transcript not found
            video_id = feed.transcript_id
            channel_info = get_youtube_channel_info(video_id)
            channel_name = channel_info.get("channel_name", "YouTube Creator")
            channel_thumbnail = channel_info.get("channel_thumbnail")
            
            return {
                "title": feed.title,
                "original_title": feed.title,
                "author": channel_name,
                "source_url": f"https://www.youtube.com/watch?v={video_id}",
                "source_type": "youtube",
                "channel_name": channel_name,
                "channel_id": channel_info.get("channel_id"),
                "video_id": video_id,
                "channel_thumbnail": channel_thumbnail,
                "favicon": "https://www.google.com/s2/favicons?domain=youtube.com&sz=64"
            }
    
    else:  # blog source type
        blog = feed.blog
        if blog:
            website_name = blog.website.replace("https://", "").replace("http://", "").split("/")[0]
            author = getattr(blog, 'author', 'Admin') or 'Admin'
            favicon = get_website_favicon(blog.website)
            
            return {
                "title": feed.title,
                "original_title": blog.title,
                "author": author,
                "source_url": blog.url,
                "source_type": "blog",
                "website_name": website_name,
                "website": blog.website,
                "favicon": favicon,
                "channel_thumbnail": None  # Not applicable for blogs
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
                "favicon": None,
                "channel_thumbnail": None
            }

# ------------------ Feed-Based Topic & Source Management ------------------

def create_topics_from_feed_categories(db: Session):
    """Create topics directly from feed categories."""
    try:
        # Get all ready feeds with categories
        feeds_with_categories = db.query(Feed).filter(
            Feed.categories.isnot(None),
            Feed.status == "ready"
        ).all()
        
        if not feeds_with_categories:
            logger.info("No feeds with categories found")
            return 0
        
        # Collect all unique categories from feeds
        all_categories = set()
        
        for feed in feeds_with_categories:
            if feed.categories:
                for category in feed.categories:
                    if category and category.strip() and category.lower() != "uncategorized":
                        all_categories.add(category)
        
        if not all_categories:
            logger.info("No valid categories found in feeds")
            return 0
        
        created_count = 0
        
        for category_name in all_categories:
            # Check if topic already exists
            existing_topic = db.query(Topic).filter(Topic.name == category_name).first()
            
            if not existing_topic:
                # Create new topic from category
                topic = Topic(
                    name=category_name,
                    description=f"Topic for {category_name} based on feed content",
                    is_active=True
                )
                db.add(topic)
                created_count += 1
                logger.info(f"Created topic from feed category: {category_name}")
        
        if created_count > 0:
            db.commit()
            logger.info(f"Topics created from feed categories: {created_count}")
        
        return created_count
        
    except Exception as e:
        logger.error(f"Error creating topics from feed categories: {e}")
        db.rollback()
        return 0

def create_sources_from_feeds(db: Session):
    """Create sources directly from feed metadata."""
    try:
        # Get all ready feeds
        ready_feeds = db.query(Feed).filter(Feed.status == "ready").all()
        
        if not ready_feeds:
            logger.info("No ready feeds found")
            return 0
        
        created_count = 0
        created_sources = set()  # Track sources to avoid duplicates
        
        for feed in ready_feeds:
            # Get feed metadata
            meta = get_feed_metadata(feed, db)
            source_type = meta.get("source_type", "blog")
            
            if source_type == "youtube":
                # Create source from YouTube channel
                channel_name = meta.get("channel_name")
                channel_id = meta.get("channel_id")
                
                if channel_name and channel_name != "YouTube Creator":
                    source_key = f"youtube:{channel_name}"
                    
                    if source_key not in created_sources:
                        existing_source = db.query(Source).filter(
                            Source.name == channel_name,
                            Source.source_type == "youtube"
                        ).first()
                        
                        if not existing_source:
                            website_url = f"https://www.youtube.com/channel/{channel_id}" if channel_id else "https://www.youtube.com"
                            
                            source = Source(
                                name=channel_name,
                                website=website_url,
                                source_type="youtube",
                                is_active=True
                            )
                            db.add(source)
                            created_count += 1
                            created_sources.add(source_key)
                            logger.info(f"Created YouTube source: {channel_name}")
            
            else:  # blog source type
                website = meta.get("website")
                
                if website and website not in ["Unknown", "#"]:
                    source_key = f"blog:{website}"
                    
                    if source_key not in created_sources:
                        existing_source = db.query(Source).filter(
                            Source.website == website,
                            Source.source_type == "blog"
                        ).first()
                        
                        if not existing_source:
                            source_name = extract_clean_source_name(website)
                            
                            source = Source(
                                name=source_name,
                                website=website,
                                source_type="blog",
                                is_active=True
                            )
                            db.add(source)
                            created_count += 1
                            created_sources.add(source_key)
                            logger.info(f"Created blog source: {source_name} from {website}")
        
        if created_count > 0:
            db.commit()
            logger.info(f"Sources created from feeds: {created_count}")
        
        return created_count
        
    except Exception as e:
        logger.error(f"Error creating sources from feeds: {e}")
        db.rollback()
        return 0

def initialize_feed_based_system(db: Session):
    """Initialize the complete topic and source system based only on feeds."""
    logger.info("Initializing feed-based topic and source system...")
    
    # Create topics from feed categories
    topics_count = create_topics_from_feed_categories(db)
    
    # Create sources from feed metadata
    sources_count = create_sources_from_feeds(db)
    
    logger.info(f"Feed-based system initialized: {topics_count} topics, {sources_count} sources")
    
    return {
        "topics_created": topics_count,
        "sources_created": sources_count
    }
def search_topics_only(query: str, search_query: str, page: int, limit: int, user_id: Optional[int], db: Session) -> Dict[str, Any]:
    """Search only topics with improved matching."""
    # First try exact match
    topics_query = db.query(Topic).filter(
        Topic.is_active == True,
        or_(
            Topic.name.ilike(search_query),
            Topic.description.ilike(search_query),
            Topic.name.ilike(f"%{query}%"),  # Broader match
            Topic.description.ilike(f"%{query}%")  # Broader match
        )
    )
    
    total_topics = topics_query.count()
    
    # If no exact matches, try fuzzy matching with all topics
    if total_topics == 0:
        all_topics = db.query(Topic).filter(Topic.is_active == True).all()
        matching_topics = []
        
        for topic in all_topics:
            # Simple fuzzy matching - check if query words appear in topic name/description
            query_words = query.lower().split()
            topic_text = f"{topic.name} {topic.description}".lower()
            
            matches = sum(1 for word in query_words if word in topic_text)
            if matches > 0:
                matching_topics.append((topic, matches))
        
        # Sort by match score
        matching_topics.sort(key=lambda x: x[1], reverse=True)
        topics = [topic for topic, score in matching_topics]
        total_topics = len(topics)
    else:
        topics = topics_query.offset((page - 1) * limit).limit(limit).all()
    
    topic_results = []
    for topic in topics:
        # Get accurate feed count
        all_feeds = db.query(Feed).filter(Feed.status == "ready", Feed.categories.isnot(None)).all()
        feed_count = 0
        for feed in all_feeds:
            if feed.categories and topic.name in feed.categories:
                feed_count += 1
        
        is_following = False
        if user_id:
            is_following = db.query(UserTopicFollow).filter(
                UserTopicFollow.user_id == user_id,
                UserTopicFollow.topic_id == topic.id
            ).first() is not None
        
        topic_results.append({
            "id": topic.id,
            "name": topic.name,
            "description": topic.description,
            "feed_count": feed_count,
            "follower_count": topic.follower_count,
            "is_following": is_following,
            "created_at": topic.created_at.isoformat() if topic.created_at else None
        })
    
    return {
        "topics": {
            "items": topic_results,
            "total": total_topics,
            "has_more": (page * limit) < total_topics
        }
    }

def search_sources_only(query: str, search_query: str, page: int, limit: int, user_id: Optional[int], db: Session) -> Dict[str, Any]:
    """Search only sources with improved matching."""
    sources_query = db.query(Source).filter(
        Source.is_active == True,
        or_(
            Source.name.ilike(search_query),
            Source.website.ilike(search_query),
            Source.source_type.ilike(search_query),
            Source.name.ilike(f"%{query}%"),  # Broader match
            Source.website.ilike(f"%{query}%")  # Broader match
        )
    )
    
    total_sources = sources_query.count()
    sources = sources_query.offset((page - 1) * limit).limit(limit).all()
    
    source_results = []
    for source in sources:
        # Get accurate feed count
        if source.source_type == "blog":
            feed_count = db.query(Feed).join(Blog).filter(
                Blog.website == source.website,
                Feed.status == "ready"
            ).count()
        else:
            all_feeds = db.query(Feed).filter(
                Feed.source_type == "youtube",
                Feed.status == "ready"
            ).all()
            feed_count = 0
            for feed in all_feeds:
                meta = get_feed_metadata(feed, db)
                if meta.get("channel_name") == source.name:
                    feed_count += 1
        
        is_following = False
        if user_id:
            is_following = db.query(UserSourceFollow).filter(
                UserSourceFollow.user_id == user_id,
                UserSourceFollow.source_id == source.id
            ).first() is not None
        
        # Get source image
        source_image = None
        if source.source_type == "youtube" and hasattr(source, 'external_id'):
            source_image = get_youtube_channel_thumbnail(source.external_id)
        elif source.source_type == "blog":
            source_image = get_website_favicon(source.website)
        
        source_results.append({
            "id": source.id,
            "name": source.name,
            "website": source.website,
            "source_type": source.source_type,
            "feed_count": feed_count,
            "follower_count": source.follower_count,
            "is_following": is_following,
            "source_image": source_image,
            "created_at": source.created_at.isoformat() if source.created_at else None
        })
    
    return {
        "sources": {
            "items": source_results,
            "total": total_sources,
            "has_more": (page * limit) < total_sources
        }
    }

def search_feeds_only(query: str, search_query: str, page: int, limit: int, user_id: Optional[int], db: Session) -> Dict[str, Any]:
    """Search only feeds."""
    feeds_query = db.query(Feed).options(joinedload(Feed.blog)).filter(
        Feed.status == "ready",
        or_(
            Feed.title.ilike(search_query),
            Feed.categories.cast(String).ilike(search_query),
            Feed.ai_generated_content.cast(String).ilike(search_query),
            Feed.title.ilike(f"%{query}%"),  # Broader match
        )
    ).order_by(Feed.created_at.desc())
    
    total_feeds = feeds_query.count()
    feeds = feeds_query.offset((page - 1) * limit).limit(limit).all()
    
    feed_results = []
    for feed in feeds:
        meta = get_feed_metadata(feed, db)
        
        feed_results.append({
            "id": feed.id,
            "title": feed.title,
            "categories": feed.categories,
            "source_type": feed.source_type,
            "slides_count": len(feed.slides),
            "meta": meta,
            "created_at": feed.created_at.isoformat() if feed.created_at else None,
            "ai_generated": bool(feed.ai_generated_content)
        })
    
    return {
        "feeds": {
            "items": feed_results,
            "total": total_feeds,
            "has_more": (page * limit) < total_feeds
        }
    }

def search_all_types(query: str, search_query: str, page: int, limit: int, user_id: Optional[int], db: Session) -> Dict[str, Any]:
    """Search all types with improved matching."""
    results = {}
    
    # Search topics
    topics_result = search_topics_only(query, search_query, page, limit, user_id, db)
    results.update(topics_result)
    
    # Search feeds
    feeds_result = search_feeds_only(query, search_query, page, limit, user_id, db)
    results.update(feeds_result)
    
    # Search sources
    sources_result = search_sources_only(query, search_query, page, limit, user_id, db)
    results.update(sources_result)
    
    return results

def should_show_fallback(results: Dict[str, Any]) -> bool:
    """Check if we should show fallback content."""
    total_items = 0
    for key in ['topics', 'feeds', 'sources']:
        if key in results and 'items' in results[key]:
            total_items += len(results[key]['items'])
    return total_items == 0

def get_popular_content(page: int, limit: int, user_id: Optional[int], db: Session) -> Dict[str, Any]:
    """Get popular content as fallback when no search results found."""
    # Get popular topics
    popular_topics = db.query(Topic).filter(
        Topic.is_active == True
    ).order_by(Topic.follower_count.desc()).limit(5).all()
    
    topic_results = []
    for topic in popular_topics:
        all_feeds = db.query(Feed).filter(Feed.status == "ready", Feed.categories.isnot(None)).all()
        feed_count = 0
        for feed in all_feeds:
            if feed.categories and topic.name in feed.categories:
                feed_count += 1
        
        is_following = False
        if user_id:
            is_following = db.query(UserTopicFollow).filter(
                UserTopicFollow.user_id == user_id,
                UserTopicFollow.topic_id == topic.id
            ).first() is not None
        
        topic_results.append({
            "id": topic.id,
            "name": topic.name,
            "description": topic.description,
            "feed_count": feed_count,
            "follower_count": topic.follower_count,
            "is_following": is_following,
            "created_at": topic.created_at.isoformat() if topic.created_at else None
        })
    
    # Get recent feeds
    recent_feeds = db.query(Feed).options(joinedload(Feed.blog)).filter(
        Feed.status == "ready"
    ).order_by(Feed.created_at.desc()).limit(limit).all()
    
    feed_results = []
    for feed in recent_feeds:
        meta = get_feed_metadata(feed, db)
        
        feed_results.append({
            "id": feed.id,
            "title": feed.title,
            "categories": feed.categories,
            "source_type": feed.source_type,
            "slides_count": len(feed.slides),
            "meta": meta,
            "created_at": feed.created_at.isoformat() if feed.created_at else None,
            "ai_generated": bool(feed.ai_generated_content)
        })
    
    return {
        "topics": {
            "items": topic_results,
            "total": len(topic_results),
            "has_more": False
        },
        "feeds": {
            "items": feed_results,
            "total": len(feed_results),
            "has_more": False
        },
        "sources": {
            "items": [],
            "total": 0,
            "has_more": False
        }
    }
# ------------------ Search Endpoints ------------------

@router.get("/", response_model=dict)
def search_feeds_and_topics(
    query: str,
    page: int = 1,
    limit: int = 20,
    search_type: str = "all",
    user_id: Optional[int] = None,
    auto_initialize: bool = True,
    db: Session = Depends(get_db)
):
    """Search across feeds, topics, and sources - never return empty results."""
    if not query or len(query.strip()) < 2:
        # If query is too short, return popular content instead of error
        return get_popular_content(page, limit, user_id, db)
    
    # Auto-initialize system if no topics/sources exist
    if auto_initialize:
        topics_count = db.query(Topic).filter(Topic.is_active == True).count()
        sources_count = db.query(Source).filter(Source.is_active == True).count()
        
        if topics_count == 0 or sources_count == 0:
            logger.info("Auto-initializing feed-based system...")
            initialize_feed_based_system(db)
    
    # Normalize search_type to lowercase
    search_type = search_type.lower()
    
    search_query = f"%{query.strip().lower()}%"
    results = {
        "query": query,
        "page": page,
        "limit": limit,
        "search_type": search_type
    }
    
    # Search based on type
    if search_type == "topics":
        results.update(search_topics_only(query, search_query, page, limit, user_id, db))
    elif search_type == "sources":
        results.update(search_sources_only(query, search_query, page, limit, user_id, db))
    elif search_type == "feeds":
        results.update(search_feeds_only(query, search_query, page, limit, user_id, db))
    else:
        results.update(search_all_types(query, search_query, page, limit, user_id, db))
    
    # If no results found, fall back to popular content
    if should_show_fallback(results):
        logger.info(f"No results found for '{query}', showing popular content as fallback")
        popular_results = get_popular_content(page, limit, user_id, db)
        results.update({
            "fallback_message": f"No results found for '{query}'. Showing popular content instead.",
            **popular_results
        })
    
    return results


@router.get("/topics/popular", response_model=dict)
def get_popular_topics(
    page: int = 1,
    limit: int = 20,
    user_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get popular topics based on actual feed count."""
    # Get topics with their feed counts
    topics_with_counts = []
    for topic in db.query(Topic).filter(Topic.is_active == True).all():
        feed_count = db.query(Feed).filter(
            Feed.categories.contains([topic.name]),
            Feed.status == "ready"
        ).count()
        
        if feed_count > 0:
            # Check if user follows this topic
            is_following = False
            if user_id:
                is_following = db.query(UserTopicFollow).filter(
                    UserTopicFollow.user_id == user_id,
                    UserTopicFollow.topic_id == topic.id
                ).first() is not None
            
            topics_with_counts.append({
                "topic": topic,
                "feed_count": feed_count,
                "is_following": is_following
            })
    
    # Sort by feed count (popularity)
    topics_with_counts.sort(key=lambda x: x["feed_count"], reverse=True)
    
    # Paginate
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated_topics = topics_with_counts[start_idx:end_idx]
    
    # Format response
    items = []
    for item in paginated_topics:
        topic = item["topic"]
        items.append({
            "id": topic.id,
            "name": topic.name,
            "description": topic.description,
            "feed_count": item["feed_count"],
            "follower_count": topic.follower_count,
            "is_following": item["is_following"],
            "created_at": topic.created_at.isoformat() if topic.created_at else None
        })
    
    total = len(topics_with_counts)
    has_more = (page * limit) < total
    
    return {
        "items": items,
        "page": page,
        "limit": limit,
        "total": total,
        "has_more": has_more
    }

@router.get("/user/feed", response_model=dict)
def get_personalized_feed(
    user_id: int,
    page: int = 1,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """Get personalized feed based on followed topics and sources."""
    # Get user's followed topics
    followed_topics = db.query(UserTopicFollow).filter(
        UserTopicFollow.user_id == user_id
    ).all()
    
    # Get user's followed sources
    followed_sources = db.query(UserSourceFollow).filter(
        UserSourceFollow.user_id == user_id
    ).all()
    
    # Build query for feeds from followed topics and sources
    feed_query = db.query(Feed).options(joinedload(Feed.blog)).filter(
        Feed.status == "ready"
    )
    
    # Add conditions for followed topics
    topic_conditions = []
    for topic_follow in followed_topics:
        topic = topic_follow.topic
        topic_conditions.append(Feed.categories.contains([topic.name]))
    
    # Add conditions for followed sources
    source_conditions = []
    for source_follow in followed_sources:
        source = source_follow.source
        if source.source_type == "blog":
            source_conditions.append(Blog.website == source.website)
    
    # Combine conditions
    conditions = []
    if topic_conditions:
        conditions.append(or_(*topic_conditions))
    if source_conditions:
        feed_query = feed_query.join(Blog)
        conditions.append(or_(*source_conditions))
    
    if conditions:
        feed_query = feed_query.filter(or_(*conditions))
    
    # Order by creation date (newest first)
    feed_query = feed_query.order_by(Feed.created_at.desc())
    
    # Paginate
    total = feed_query.count()
    feeds = feed_query.offset((page - 1) * limit).limit(limit).all()
    
    # Format response
    items = []
    for feed in feeds:
        meta = get_feed_metadata(feed, db)
        
        items.append({
            "id": feed.id,
            "title": feed.title,
            "categories": feed.categories,
            "source_type": feed.source_type,
            "slides_count": len(feed.slides),
            "meta": meta,
            "created_at": feed.created_at.isoformat() if feed.created_at else None,
            "ai_generated": bool(feed.ai_generated_content)
        })
    
    has_more = (page * limit) < total
    
    return {
        "items": items,
        "page": page,
        "limit": limit,
        "total": total,
        "has_more": has_more,
        "followed_topics_count": len(followed_topics),
        "followed_sources_count": len(followed_sources)
    }

# ------------------ Follow/Unfollow Endpoints ------------------

@router.post("/topics/{topic_name}/follow", response_model=dict)
def follow_topic(
    topic_name: str,
    user_id: int,
    db: Session = Depends(get_db)
):
    """Follow a topic."""
    topic = db.query(Topic).filter(Topic.name == topic_name).first()
    if not topic:
        # Create topic if it doesn't exist
        topic = Topic(name=topic_name, description=f"Topic for {topic_name}")
        db.add(topic)
        db.flush()
    
    # Check if already following
    existing_follow = db.query(UserTopicFollow).filter(
        UserTopicFollow.user_id == user_id,
        UserTopicFollow.topic_id == topic.id
    ).first()
    
    if existing_follow:
        raise HTTPException(status_code=400, detail="Already following this topic")
    
    # Create follow relationship
    follow = UserTopicFollow(user_id=user_id, topic_id=topic.id)
    db.add(follow)
    
    # Update follower count
    topic.follower_count += 1
    topic.updated_at = datetime.utcnow()
    
    db.commit()
    
    return {
        "message": f"Started following topic: {topic_name}",
        "topic_id": topic.id,
        "topic_name": topic.name,
        "follower_count": topic.follower_count,
        "is_following": True
    }

@router.post("/topics/{topic_name}/unfollow", response_model=dict)
def unfollow_topic(
    topic_name: str,
    user_id: int,
    db: Session = Depends(get_db)
):
    """Unfollow a topic."""
    topic = db.query(Topic).filter(Topic.name == topic_name).first()
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    
    follow = db.query(UserTopicFollow).filter(
        UserTopicFollow.user_id == user_id,
        UserTopicFollow.topic_id == topic.id
    ).first()
    
    if not follow:
        raise HTTPException(status_code=400, detail="Not following this topic")
    
    db.delete(follow)
    topic.follower_count = max(0, topic.follower_count - 1)
    topic.updated_at = datetime.utcnow()
    
    db.commit()
    
    return {
        "message": f"Stopped following topic: {topic_name}",
        "topic_id": topic.id,
        "topic_name": topic.name,
        "follower_count": topic.follower_count,
        "is_following": False
    }

@router.post("/sources/{source_id}/follow", response_model=dict)
def follow_source(
    source_id: int,
    user_id: int,
    db: Session = Depends(get_db)
):
    """Follow a source."""
    source = db.query(Source).filter(Source.id == source_id).first()
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    
    existing_follow = db.query(UserSourceFollow).filter(
        UserSourceFollow.user_id == user_id,
        UserSourceFollow.source_id == source_id
    ).first()
    
    if existing_follow:
        raise HTTPException(status_code=400, detail="Already following this source")
    
    follow = UserSourceFollow(user_id=user_id, source_id=source_id)
    db.add(follow)
    source.follower_count += 1
    source.updated_at = datetime.utcnow()
    
    db.commit()
    
    return {
        "message": f"Started following source: {source.name}",
        "source_id": source.id,
        "source_name": source.name,
        "follower_count": source.follower_count,
        "is_following": True
    }

@router.post("/sources/{source_id}/unfollow", response_model=dict)
def unfollow_source(
    source_id: int,
    user_id: int,
    db: Session = Depends(get_db)
):
    """Unfollow a source."""
    source = db.query(Source).filter(Source.id == source_id).first()
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    
    follow = db.query(UserSourceFollow).filter(
        UserSourceFollow.user_id == user_id,
        UserSourceFollow.source_id == source_id
    ).first()
    
    if not follow:
        raise HTTPException(status_code=400, detail="Not following this source")
    
    db.delete(follow)
    source.follower_count = max(0, source.follower_count - 1)
    source.updated_at = datetime.utcnow()
    
    db.commit()
    
    return {
        "message": f"Stopped following source: {source.name}",
        "source_id": source.id,
        "source_name": source.name,
        "follower_count": source.follower_count,
        "is_following": False
    }

# ------------------ Admin Endpoints ------------------

@router.post("/admin/initialize-system", response_model=dict)
def initialize_feed_system(db: Session = Depends(get_db)):
    """Initialize the complete feed-based topic and source system."""
    try:
        result = initialize_feed_based_system(db)
        
        return {
            "message": "Feed-based topic-source system initialized successfully",
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize system: {str(e)}")

@router.get("/admin/system-status", response_model=dict)
def get_system_status(db: Session = Depends(get_db)):
    """Get current status of feed-based topics and sources system."""
    topics_count = db.query(Topic).filter(Topic.is_active == True).count()
    sources_count = db.query(Source).filter(Source.is_active == True).count()
    feeds_count = db.query(Feed).filter(Feed.status == "ready").count()
    
    # Get topics with actual feed counts
    topics_with_counts = []
    for topic in db.query(Topic).filter(Topic.is_active == True).all():
        feed_count = db.query(Feed).filter(
            Feed.categories.contains([topic.name]),
            Feed.status == "ready"
        ).count()
        topics_with_counts.append({
            "name": topic.name,
            "feed_count": feed_count,
            "follower_count": topic.follower_count
        })
    
    # Get sources with actual feed counts
    sources_with_counts = []
    for source in db.query(Source).filter(Source.is_active == True).all():
        if source.source_type == "blog":
            feed_count = db.query(Feed).join(Blog).filter(
                Blog.website == source.website,
                Feed.status == "ready"
            ).count()
        else:
            feed_count = db.query(Feed).filter(
                Feed.source_type == "youtube",
                Feed.status == "ready"
            ).count()
        
        sources_with_counts.append({
            "name": source.name,
            "source_type": source.source_type,
            "feed_count": feed_count,
            "follower_count": source.follower_count
        })
    
    return {
        "system_status": "active",
        "topics_count": topics_count,
        "sources_count": sources_count,
        "feeds_count": feeds_count,
        "topics": topics_with_counts,
        "sources": sources_with_counts
    }




# ------------------ Feed Discovery by Topic & Source ------------------

# Update the get_feeds_by_topic function to fix category matching
@router.get("/topic/{topic_name}/feeds", response_model=dict)
def get_feeds_by_topic(
    topic_name: str,
    page: int = 1,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """Get all feeds with full metadata and slides for a specific topic."""
    # First, verify the topic exists
    topic = db.query(Topic).filter(
        Topic.name == topic_name,
        Topic.is_active == True
    ).first()
    
    if not topic:
        raise HTTPException(status_code=404, detail=f"Topic '{topic_name}' not found")
    
    # Query feeds that have this topic in their categories
    # Use string matching since JSON contains doesn't work well with array elements
    feeds_query = db.query(Feed).options(
        joinedload(Feed.blog),
        joinedload(Feed.slides)
    ).filter(
        Feed.status == "ready"
    ).order_by(Feed.created_at.desc())
    
    # Get all feeds and filter by category manually
    all_feeds = feeds_query.all()
    matching_feeds = []
    
    for feed in all_feeds:
        if feed.categories and topic_name in feed.categories:
            matching_feeds.append(feed)
    
    # Manual pagination
    total = len(matching_feeds)
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated_feeds = matching_feeds[start_idx:end_idx]
    
    # Format response with full feed data including slides
    items = []
    for feed in paginated_feeds:
        # Get proper metadata
        meta = get_feed_metadata(feed, db)
        
        # Get slides sorted by order
        sorted_slides = sorted(feed.slides, key=lambda x: x.order)
        
        items.append({
            "id": feed.id,
            "blog_id": feed.blog_id,
            "transcript_id": feed.transcript_id,
            "title": feed.title,
            "categories": feed.categories,
            "status": feed.status,
            "source_type": feed.source_type or "blog",
            "ai_generated_content": feed.ai_generated_content or {},
            "meta": meta,
            "slides": [
                {
                    "id": slide.id,
                    "order": slide.order,
                    "title": slide.title,
                    "body": slide.body,
                    "bullets": slide.bullets or [],
                    "background_color": slide.background_color,
                    "background_image_prompt": None,
                    "source_refs": slide.source_refs or [],
                    "render_markdown": bool(slide.render_markdown),
                    "created_at": slide.created_at.isoformat() if slide.created_at else None,
                    "updated_at": slide.updated_at.isoformat() if slide.updated_at else None
                } for slide in sorted_slides
            ],
            "slides_count": len(feed.slides),
            "created_at": feed.created_at.isoformat() if feed.created_at else None,
            "updated_at": feed.updated_at.isoformat() if feed.updated_at else None,
            "ai_generated": bool(feed.ai_generated_content)
        })
    
    has_more = (page * limit) < total
    
    return {
        "topic": {
            "id": topic.id,
            "name": topic.name,
            "description": topic.description,
            "follower_count": topic.follower_count
        },
        "items": items,
        "page": page,
        "limit": limit,
        "total": total,
        "has_more": has_more
    }



@router.get("/source/{source_id}/feeds", response_model=dict)
def get_feeds_by_source(
    source_id: int,
    page: int = 1,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """Get all feeds with full metadata and slides for a specific source."""
    # First, verify the source exists
    source = db.query(Source).filter(
        Source.id == source_id,
        Source.is_active == True
    ).first()
    
    if not source:
        raise HTTPException(status_code=404, detail=f"Source with ID {source_id} not found")
    
    # Build query based on source type
    if source.source_type == "blog":
        # For blog sources, get feeds from blogs with matching website
        feeds_query = db.query(Feed).options(
            joinedload(Feed.blog),
            joinedload(Feed.slides)
        ).join(Blog).filter(
            Blog.website == source.website,
            Feed.status == "ready"
        ).order_by(Feed.created_at.desc())
    
    else:  # youtube source
        # For YouTube sources, we need to match by channel name
        # This is more complex - we'll get all YouTube feeds and filter by channel
        feeds_query = db.query(Feed).options(
            joinedload(Feed.blog),
            joinedload(Feed.slides)
        ).filter(
            Feed.source_type == "youtube",
            Feed.status == "ready"
        ).order_by(Feed.created_at.desc())
        
        # We'll filter the results after fetching to match the channel
        all_feeds = feeds_query.all()
        matching_feeds = []
        
        for feed in all_feeds:
            meta = get_feed_metadata(feed, db)
            channel_name = meta.get("channel_name")
            if channel_name == source.name:
                matching_feeds.append(feed)
        
        # Manual pagination for YouTube feeds
        total = len(matching_feeds)
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_feeds = matching_feeds[start_idx:end_idx]
        
        # Format response for YouTube feeds
        items = []
        for feed in paginated_feeds:
            meta = get_feed_metadata(feed, db)
            sorted_slides = sorted(feed.slides, key=lambda x: x.order)
            
            items.append({
                "id": feed.id,
                "blog_id": feed.blog_id,
                "transcript_id": feed.transcript_id,
                "title": feed.title,
                "categories": feed.categories,
                "status": feed.status,
                "source_type": feed.source_type or "youtube",
                "ai_generated_content": feed.ai_generated_content or {},
                "meta": meta,
                "slides": [
                    {
                        "id": slide.id,
                        "order": slide.order,
                        "title": slide.title,
                        "body": slide.body,
                        "bullets": slide.bullets or [],
                        "background_color": slide.background_color,
                        "background_image_prompt": None,
                        "source_refs": slide.source_refs or [],
                        "render_markdown": bool(slide.render_markdown),
                        "created_at": slide.created_at.isoformat() if slide.created_at else None,
                        "updated_at": slide.updated_at.isoformat() if slide.updated_at else None
                    } for slide in sorted_slides
                ],
                "slides_count": len(feed.slides),
                "created_at": feed.created_at.isoformat() if feed.created_at else None,
                "updated_at": feed.updated_at.isoformat() if feed.updated_at else None,
                "ai_generated": bool(feed.ai_generated_content)
            })
        
        has_more = (page * limit) < total
        
        return {
            "source": {
                "id": source.id,
                "name": source.name,
                "website": source.website,
                "source_type": source.source_type,
                "follower_count": source.follower_count
            },
            "items": items,
            "page": page,
            "limit": limit,
            "total": total,
            "has_more": has_more
        }
    
    # For blog sources, continue with normal query
    total = feeds_query.count()
    feeds = feeds_query.offset((page - 1) * limit).limit(limit).all()
    
    # Format response for blog feeds
    items = []
    for feed in feeds:
        meta = get_feed_metadata(feed, db)
        sorted_slides = sorted(feed.slides, key=lambda x: x.order)
        
        items.append({
            "id": feed.id,
            "blog_id": feed.blog_id,
            "transcript_id": feed.transcript_id,
            "title": feed.title,
            "categories": feed.categories,
            "status": feed.status,
            "source_type": feed.source_type or "blog",
            "ai_generated_content": feed.ai_generated_content or {},
            "meta": meta,
            "slides": [
                {
                    "id": slide.id,
                    "order": slide.order,
                    "title": slide.title,
                    "body": slide.body,
                    "bullets": slide.bullets or [],
                    "background_color": slide.background_color,
                    "background_image_prompt": None,
                    "source_refs": slide.source_refs or [],
                    "render_markdown": bool(slide.render_markdown),
                    "created_at": slide.created_at.isoformat() if slide.created_at else None,
                    "updated_at": slide.updated_at.isoformat() if slide.updated_at else None
                } for slide in sorted_slides
            ],
            "slides_count": len(feed.slides),
            "created_at": feed.created_at.isoformat() if feed.created_at else None,
            "updated_at": feed.updated_at.isoformat() if feed.updated_at else None,
            "ai_generated": bool(feed.ai_generated_content)
        })
    
    has_more = (page * limit) < total
    
    return {
        "source": {
            "id": source.id,
            "name": source.name,
            "website": source.website,
            "source_type": source.source_type,
            "follower_count": source.follower_count
        },
        "items": items,
        "page": page,
        "limit": limit,
        "total": total,
        "has_more": has_more
    }

# Also update the summary endpoint
@router.get("/topic/{topic_name}/feeds/summary", response_model=dict)
def get_feeds_by_topic_summary(
    topic_name: str,
    page: int = 1,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """Get feed summaries (without slides) for a specific topic - faster for listing."""
    topic = db.query(Topic).filter(
        Topic.name == topic_name,
        Topic.is_active == True
    ).first()
    
    if not topic:
        raise HTTPException(status_code=404, detail=f"Topic '{topic_name}' not found")
    
    # Get all feeds and filter manually
    feeds_query = db.query(Feed).options(joinedload(Feed.blog)).filter(
        Feed.status == "ready"
    ).order_by(Feed.created_at.desc())
    
    all_feeds = feeds_query.all()
    matching_feeds = []
    
    for feed in all_feeds:
        if feed.categories and topic_name in feed.categories:
            matching_feeds.append(feed)
    
    total = len(matching_feeds)
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    feeds = matching_feeds[start_idx:end_idx]
    
    items = []
    for feed in feeds:
        meta = get_feed_metadata(feed, db)
        
        items.append({
            "id": feed.id,
            "title": feed.title,
            "categories": feed.categories,
            "source_type": feed.source_type,
            "slides_count": len(feed.slides),
            "meta": meta,
            "created_at": feed.created_at.isoformat() if feed.created_at else None,
            "ai_generated": bool(feed.ai_generated_content)
        })
    
    has_more = (page * limit) < total
    
    return {
        "topic": {
            "id": topic.id,
            "name": topic.name,
            "description": topic.description,
            "follower_count": topic.follower_count
        },
        "items": items,
        "page": page,
        "limit": limit,
        "total": total,
        "has_more": has_more
    }

# Add a debug endpoint to see what categories are actually in feeds
@router.get("/debug/feed-categories", response_model=dict)
def debug_feed_categories(db: Session = Depends(get_db)):
    """Debug endpoint to see what categories exist in feeds."""
    feeds = db.query(Feed).filter(
        Feed.categories.isnot(None),
        Feed.status == "ready"
    ).all()
    
    all_categories = {}
    
    for feed in feeds:
        if feed.categories:
            for category in feed.categories:
                if category not in all_categories:
                    all_categories[category] = []
                all_categories[category].append({
                    "feed_id": feed.id,
                    "feed_title": feed.title
                })
    
    return {
        "total_feeds_with_categories": len(feeds),
        "categories": all_categories
    }

@router.get("/source/{source_id}/feeds/summary", response_model=dict)
def get_feeds_by_source_summary(
    source_id: int,
    page: int = 1,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """Get feed summaries (without slides) for a specific source - faster for listing."""
    source = db.query(Source).filter(
        Source.id == source_id,
        Source.is_active == True
    ).first()
    
    if not source:
        raise HTTPException(status_code=404, detail=f"Source with ID {source_id} not found")
    
    if source.source_type == "blog":
        feeds_query = db.query(Feed).options(joinedload(Feed.blog)).join(Blog).filter(
            Blog.website == source.website,
            Feed.status == "ready"
        ).order_by(Feed.created_at.desc())
        
        total = feeds_query.count()
        feeds = feeds_query.offset((page - 1) * limit).limit(limit).all()
    
    else:  # youtube
        feeds_query = db.query(Feed).options(joinedload(Feed.blog)).filter(
            Feed.source_type == "youtube",
            Feed.status == "ready"
        ).order_by(Feed.created_at.desc())
        
        all_feeds = feeds_query.all()
        matching_feeds = []
        
        for feed in all_feeds:
            meta = get_feed_metadata(feed, db)
            if meta.get("channel_name") == source.name:
                matching_feeds.append(feed)
        
        total = len(matching_feeds)
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        feeds = matching_feeds[start_idx:end_idx]
    
    items = []
    for feed in feeds:
        meta = get_feed_metadata(feed, db)
        
        items.append({
            "id": feed.id,
            "title": feed.title,
            "categories": feed.categories,
            "source_type": feed.source_type,
            "slides_count": len(feed.slides),
            "meta": meta,
            "created_at": feed.created_at.isoformat() if feed.created_at else None,
            "ai_generated": bool(feed.ai_generated_content)
        })
    
    has_more = (page * limit) < total
    
    return {
        "source": {
            "id": source.id,
            "name": source.name,
            "website": source.website,
            "source_type": source.source_type,
            "follower_count": source.follower_count
        },
        "items": items,
        "page": page,
        "limit": limit,
        "total": total,
        "has_more": has_more
    }