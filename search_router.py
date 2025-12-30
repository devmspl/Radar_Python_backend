from fastapi import APIRouter, Depends, HTTPException,status
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
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = None
if openai_api_key:
    openai_client = OpenAI(api_key=openai_api_key)
else:
    logger.warning("OPENAI_API_KEY not found. AI features will be disabled.")

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

def generate_topic_description(topic_name: str, sample_feeds: List[Feed], db: Session) -> str:
    """Generate a meaningful topic description based on actual feed content."""
    try:
        if not sample_feeds:
            return f"Topic for {topic_name} based on feed content"
        
        # Extract titles and categories from sample feeds
        feed_samples = []
        for feed in sample_feeds[:5]:  # Use up to 5 feeds for context
            meta = get_feed_metadata(feed, db)
            feed_samples.append({
                "title": feed.title,
                "categories": feed.categories or [],
                "source_type": feed.source_type,
                "author": meta.get("author", "Unknown")
            })
        
        prompt = f"""Based on these feeds in the '{topic_name}' topic, create a concise 200-word description explaining what this topic is about.

Sample feeds:
{json.dumps(feed_samples, indent=2)}

Write a descriptive paragraph that:
1. Summarizes the main themes and subjects covered
2. Mentions the types of content (tutorials, news, guides, etc.)
3. Indicates the target audience (beginners, experts, etc.)
4. Highlights key technologies or areas of focus
5. Keeps it engaging and informative

Description:"""
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a content analyst who creates engaging topic descriptions based on feed content."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        description = response.choices[0].message.content.strip()
        return description
        
    except Exception as e:
        logger.error(f"Error generating topic description for {topic_name}: {e}")
        return f"Topic for {topic_name} based on feed content"


def extract_website_description(website_url: str) -> str:
    """Extract or generate a description for a website."""
    try:
        if not website_url or website_url in ["Unknown", "#"]:
            return "No description available"
        
        # Add protocol if missing for fetching
        fetch_url = website_url
        if not fetch_url.startswith(('http://', 'https://')):
            fetch_url = f"https://{fetch_url}"
        
        # First try to fetch the website and extract meta description
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(fetch_url, headers=headers, timeout=10)
            
            # Look for meta description tag
            if response.status_code == 200:
                content = response.text
                
                # Try to find meta description
                meta_desc_pattern = r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)["\'][^>]*>'
                match = re.search(meta_desc_pattern, content, re.IGNORECASE)
                if match:
                    description = match.group(1).strip()
                    if description and len(description) > 10:
                        return description[:300]  # Limit length
                
                # Try to find og:description
                og_desc_pattern = r'<meta[^>]*property=["\']og:description["\'][^>]*content=["\']([^"\']+)["\'][^>]*>'
                match = re.search(og_desc_pattern, content, re.IGNORECASE)
                if match:
                    description = match.group(1).strip()
                    if description and len(description) > 10:
                        return description[:300]
        
        except Exception as e:
            logger.debug(f"Could not fetch website for description extraction: {e}")
        
        # Fallback: Use OpenAI to generate a description based on the domain name
        clean_url = website_url.replace("https://", "").replace("http://", "").split("/")[0]
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are a content analyst. Generate a brief, professional description for a website based on its domain name.
                    Return ONLY a 1-2 sentence description, no explanations.
                    Examples:
                    - "blog.hubspot.com" -> "HubSpot Blog"
                    - "news.ycombinator.com" -> "Hacker News" 
                    - "techcrunch.com" -> "TechCrunch"
                    - "medium.com" -> "Medium"
                    - "towardsdatascience.com" -> "Towards Data Science"
                    - If unsure, create a generic description based on the domain name."""
                },
                {
                    "role": "user",
                    "content": f"Generate a brief description for website with domain: {clean_url}"
                }
            ],
            temperature=0.3,
            max_tokens=100
        )
        
        description = response.choices[0].message.content.strip()
        description = description.replace('"', '').replace("'", "").strip()
        
        return description[:300]  # Limit length
        
    except Exception as e:
        logger.error(f"Website description extraction failed for {website_url}: {e}")
        # Final fallback
        if website_url and website_url not in ["Unknown", "#"]:
            domain = website_url.replace("https://", "").replace("http://", "").split("/")[0]
            return f"Website at {domain} featuring content on various topics."
        return "No description available"

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
    """Get favicon URL for a website."""
    try:
        # Clean and format the URL
        if not website_url.startswith(('http://', 'https://')):
            website_url = f"https://{website_url}"
        
        # Extract domain
        from urllib.parse import urlparse
        parsed_url = urlparse(website_url)
        domain = parsed_url.netloc
        
        if not domain:
            # If parsing failed, try to extract domain manually
            domain = website_url.replace('https://', '').replace('http://', '').split('/')[0]
        
        # Common favicon locations
        favicon_urls = [
            f"https://{domain}/favicon.ico",
            f"https://{domain}/favicon.png",
            f"https://www.{domain}/favicon.ico",
            f"https://www.{domain}/favicon.png",
            f"https://{domain}/apple-touch-icon.png",
            f"https://www.{domain}/apple-touch-icon.png"
        ]
        
        # Try to check if favicon exists (optional)
        # For simplicity, we'll just return the most likely URL
        return favicon_urls[0]
        
    except Exception:
        # Fallback to a generic favicon
        return "https://www.google.com/s2/favicons?domain=" + (website_url.split('//')[-1].split('/')[0] if '//' in website_url else website_url)


def get_youtube_channel_info(video_id: str) -> Dict[str, Any]:
    """Get YouTube channel information using YouTube Data API (requests version)."""
    if not YOUTUBE_API_KEY:
        logger.warning("YouTube API key not found. Using fallback channel info.")
        return {
            "channel_name": "YouTube Creator",
            "channel_id": None,
            "available": False
        }
    
    try:
        # First, get video details to get channel ID
        video_url = f"https://www.googleapis.com/youtube/v3/videos"
        params = {
            "key": YOUTUBE_API_KEY,
            "id": video_id,
            "part": "snippet"
        }
        
        response = requests.get(video_url, params=params, timeout=10)
        if response.status_code != 200:
            logger.warning(f"Failed to get video info: {response.status_code}")
            return {
                "channel_name": "YouTube Creator",
                "channel_id": None,
                "available": False
            }
        
        video_data = response.json()
        if not video_data.get("items"):
            return {
                "channel_name": "YouTube Creator",
                "channel_id": None,
                "available": False
            }
        
        snippet = video_data["items"][0]["snippet"]
        channel_id = snippet.get("channelId")
        channel_title = snippet.get("channelTitle", "YouTube Creator")
        
        # Get channel details
        if channel_id:
            channel_url = f"https://www.googleapis.com/youtube/v3/channels"
            channel_params = {
                "key": YOUTUBE_API_KEY,
                "id": channel_id,
                "part": "snippet,statistics"
            }
            
            channel_response = requests.get(channel_url, params=channel_params, timeout=10)
            if channel_response.status_code == 200:
                channel_data = channel_response.json()
                if channel_data.get("items"):
                    channel_info = channel_data["items"][0]["snippet"]
                    stats = channel_data["items"][0].get("statistics", {})
                    
                    return {
                        "channel_name": channel_title,
                        "channel_id": channel_id,
                        "thumbnail": channel_info.get("thumbnails", {}).get("default", {}).get("url", ""),
                        "subscriber_count": stats.get("subscriberCount", "0"),
                        "video_count": stats.get("videoCount", "0"),
                        "available": True
                    }
        
        return {
            "channel_name": channel_title,
            "channel_id": channel_id,
            "available": True
        }
        
    except Exception as e:
        logger.error(f"Error fetching YouTube channel info: {e}")
        return {
            "channel_name": "YouTube Creator",
            "channel_id": None,
            "available": False
        }

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
    """Extract proper metadata for feeds including YouTube channel names, correct URLs, favicons, and topic descriptions."""
    # Initialize base metadata
    base_meta = {}
    concepts = []
    
    # Get concepts for this feed
    if hasattr(feed, 'concepts') and feed.concepts:
        for concept in feed.concepts:
            concepts.append({
                "id": concept.id,
                "name": concept.name,
                "description": concept.description
            })
    
    if feed.source_type == "youtube" and feed.transcript_id:
        # Get the transcript to access YouTube-specific data
        transcript = db.query(Transcript).filter(
            Transcript.transcript_id == feed.transcript_id
        ).first()
        
        if transcript:
            video_id = getattr(transcript, 'video_id', None)
            if not video_id:
                video_id = feed.transcript_id
            
            original_title = transcript.title if transcript else feed.title
            
            # Get channel information from YouTube API
            channel_info = get_youtube_channel_info(video_id)
            channel_name = channel_info.get("channel_name", "YouTube Creator")
            
            source_url = f"https://www.youtube.com/watch?v={video_id}"
            
            # Use YouTube favicon
            favicon = "https://www.youtube.com/s/desktop/12d6b690/img/favicon.ico"
            
            base_meta = {
                "title": feed.title,
                "original_title": original_title,
                "author": channel_name,
                "source_url": source_url,
                "source_type": "youtube",
                "channel_name": channel_name,
                "channel_id": channel_info.get("channel_id"),
                "video_id": video_id,
                "favicon": favicon,
                "channel_info": channel_info,
                "concepts": concepts
            }
    
    # For blogs
    elif feed.source_type == "blog" and feed.blog:
        blog = feed.blog
        website_name = blog.website.replace("https://", "").replace("http://", "").split("/")[0]
        author = getattr(blog, 'author', 'Admin') or 'Admin'
        
        # Get favicon URL
        favicon = get_website_favicon(blog.website)
        
        base_meta = {
            "title": feed.title,
            "original_title": blog.title,
            "author": author,
            "source_url": blog.url,
            "source_type": "blog",
            "website_name": website_name,
            "website": blog.website,
            "favicon": favicon,
            "concepts": concepts
        }
    
    else:
        # Fallback
        base_meta = {
            "title": feed.title,
            "original_title": feed.title,
            "author": "Unknown",
            "source_url": "#",
            "source_type": feed.source_type or "blog",
            "website_name": "Unknown",
            "website": "Unknown",
            "favicon": get_website_favicon("Unknown"),
            "concepts": concepts
        }
    
    # ADD TOPIC DESCRIPTIONS if feed has categories
    if feed.categories:
        topics = []
        for category_name in feed.categories:
            # Get topic from database
            topic = db.query(Topic).filter(
                Topic.name == category_name,
                Topic.is_active == True
            ).first()
            
            if topic:
                topics.append({
                    "name": topic.name,
                    "description": topic.description,
                    "id": topic.id
                })
            else:
                # If topic doesn't exist, create a basic one
                topics.append({
                    "name": category_name,
                    "description": f"Content related to {category_name}",
                    "id": None
                })
        
        base_meta["topics"] = topics
    
    return base_meta

def get_source_metadata(source: Source) -> Dict[str, Any]:
    """Get comprehensive metadata for a source."""
    if source.source_type == "youtube":
        # Extract channel ID from website URL
        channel_id = None
        if source.website and 'youtube.com/channel/' in source.website:
            # Extract from standard YouTube channel URL
            import re
            match = re.search(r'youtube\.com/channel/([a-zA-Z0-9_-]{24})', source.website)
            if match:
                channel_id = match.group(1)
        
        # Try to get channel ID from external_id attribute if not in URL
        if not channel_id:
            channel_id = getattr(source, 'external_id', None)
        
        channel_thumbnail = None
        if channel_id:
            # Get channel thumbnail using YouTube API
            try:
                channel_url = f"https://www.googleapis.com/youtube/v3/channels"
                params = {
                    "key": YOUTUBE_API_KEY,
                    "id": channel_id,
                    "part": "snippet"
                }
                
                response = requests.get(channel_url, params=params, timeout=10)
                if response.status_code == 200:
                    channel_data = response.json()
                    if channel_data.get("items"):
                        thumbnails = channel_data["items"][0]["snippet"].get("thumbnails", {})
                        if thumbnails.get('high'):
                            channel_thumbnail = thumbnails['high']['url']
                        elif thumbnails.get('medium'):
                            channel_thumbnail = thumbnails['medium']['url']
                        elif thumbnails.get('default'):
                            channel_thumbnail = thumbnails['default']['url']
            except Exception as e:
                logger.error(f"Error fetching channel thumbnail: {e}")
        
        return {
            "type": "youtube",
            "channel_id": channel_id,
            "channel_thumbnail": channel_thumbnail,
            "favicon": "https://www.youtube.com/s/desktop/12d6b690/img/favicon.ico",
            "website_url": source.website,
            "source_image": channel_thumbnail or "https://www.youtube.com/s/desktop/12d6b690/img/favicon.ico",
            "verified": getattr(source, 'verified', False),
            "subscriber_count": getattr(source, 'subscriber_count', None)
        }
    else:
        # For blog sources
        # Ensure the website has a protocol for favicon fetching
        website = source.website
        if not website.startswith(('http://', 'https://')):
            website = f"https://{website}"
        
        favicon = get_website_favicon(website)
        
        # Get domain for display
        domain = source.website.replace("https://", "").replace("http://", "").split("/")[0]
        
        return {
            "type": "blog",
            "favicon": favicon or f"https://www.google.com/s2/favicons?domain={domain}&sz=64",
            "website_url": website,
            "domain": domain,
            "source_image": favicon or f"https://www.google.com/s2/favicons?domain={domain}&sz=64",
            "verified": getattr(source, 'verified', False),
            "rss_feed": getattr(source, 'rss_feed', None),
            "last_crawled": getattr(source, 'last_crawled', None)
        }
# ------------------ Feed-Based Topic & Source Management ------------------

def create_topics_from_feed_categories(db: Session):
    """Create topics directly from feed categories with better descriptions."""
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
        category_feeds_map = {}
        
        for feed in feeds_with_categories:
            if feed.categories:
                for category in feed.categories:
                    if category and category.strip() and category.lower() != "uncategorized":
                        all_categories.add(category)
                        if category not in category_feeds_map:
                            category_feeds_map[category] = []
                        category_feeds_map[category].append(feed)
        
        if not all_categories:
            logger.info("No valid categories found in feeds")
            return 0
        
        created_count = 0
        
        for category_name in all_categories:
            # Check if topic already exists
            existing_topic = db.query(Topic).filter(Topic.name == category_name).first()
            
            if not existing_topic:
                # Get sample feeds for this category
                sample_feeds = category_feeds_map.get(category_name, [])[:5]
                
                # Generate better description
                description = generate_topic_description(category_name, sample_feeds, db)
                
                # Create new topic from category
                topic = Topic(
                    name=category_name,
                    description=description,
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
                channel_description = meta.get("channel_description", "")
                
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
                                description=channel_description[:500] if channel_description else "",
                                is_active=True
                            )
                            db.add(source)
                            created_count += 1
                            created_sources.add(source_key)
                            logger.info(f"Created YouTube source: {channel_name}")
            
            else:  # blog source type
                website = meta.get("website")
                website_description = meta.get("website_description", "")
                
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
                                description=website_description[:500] if website_description else "",
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

@router.post("/topics/{topic_id}/follow", response_model=dict)
def follow_topic(
    topic_id: int,
    user_id: int,
    db: Session = Depends(get_db)
):
    """Follow a topic by ID."""
    topic = db.query(Topic).filter(Topic.id == topic_id).first()
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    
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
        "message": f"Started following topic: {topic.name}",
        "topic_id": topic.id,
        "topic_name": topic.name,
        "follower_count": topic.follower_count,
        "is_following": True
    }

@router.post("/topics/{topic_id}/unfollow", response_model=dict)
def unfollow_topic(
    topic_id: int,
    user_id: int,
    db: Session = Depends(get_db)
):
    """Unfollow a topic by ID."""
    topic = db.query(Topic).filter(Topic.id == topic_id).first()
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
        "message": f"Stopped following topic: {topic.name}",
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
    try:
        # First, verify the topic exists
        topic = db.query(Topic).filter(
            Topic.name == topic_name,
            Topic.is_active == True
        ).first()
        
        if not topic:
            raise HTTPException(status_code=404, detail=f"Topic '{topic_name}' not found")
        
        # Use a more efficient approach - get all feeds first, then filter in Python
        # This avoids complex SQL queries with JSON arrays
        all_feeds = db.query(Feed).options(
            joinedload(Feed.blog),
            joinedload(Feed.slides)
        ).filter(
            Feed.status == "ready",
            Feed.categories.isnot(None)
        ).order_by(Feed.created_at.desc()).all()
        
        # Filter feeds that have this topic in their categories
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
            
            # Add topic description to feed metadata
            meta_with_topic = {
                **meta,
                "topic": {
                    "name": topic.name,
                    "description": topic.description
                }
            }
            
            items.append({
                "id": feed.id,
                "blog_id": feed.blog_id,
                "transcript_id": feed.transcript_id,
                "title": feed.title,
                "categories": feed.categories,
                "status": feed.status,
                "source_type": feed.source_type or "blog",
                "ai_generated_content": feed.ai_generated_content or {},
                "meta": meta_with_topic,  # Updated metadata with topic info
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
                "follower_count": topic.follower_count,
                "created_at": topic.created_at.isoformat() if topic.created_at else None,
                "updated_at": topic.updated_at.isoformat() if topic.updated_at else None
            },
            "items": items,
            "page": page,
            "limit": limit,
            "total": total,
            "has_more": has_more
        }
        
    except Exception as e:
        logger.error(f"Error in get_feeds_by_topic for topic '{topic_name}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/topic/{topic_name}/feeds/summary", response_model=dict)
def get_feeds_by_topic_summary(
    topic_name: str,
    page: int = 1,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """Get feed summaries (without slides) for a specific topic - faster for listing."""
    try:
        topic = db.query(Topic).filter(
            Topic.name == topic_name,
            Topic.is_active == True
        ).first()
        
        if not topic:
            raise HTTPException(status_code=404, detail=f"Topic '{topic_name}' not found")
        
        # Get all feeds and filter manually (more reliable than SQL JSON queries)
        all_feeds = db.query(Feed).options(joinedload(Feed.blog)).filter(
            Feed.status == "ready",
            Feed.categories.isnot(None)
        ).order_by(Feed.created_at.desc()).all()
        
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
            
            # Add topic description to feed metadata
            meta_with_topic = {
                **meta,
                "topic": {
                    "name": topic.name,
                    "description": topic.description
                }
            }
            
            items.append({
                "id": feed.id,
                "title": feed.title,
                "categories": feed.categories,
                "source_type": feed.source_type,
                "slides_count": len(feed.slides),
                "meta": meta_with_topic,  # Updated metadata with topic info
                "created_at": feed.created_at.isoformat() if feed.created_at else None,
                "ai_generated": bool(feed.ai_generated_content)
            })
        
        has_more = (page * limit) < total
        
        return {
            "topic": {
                "id": topic.id,
                "name": topic.name,
                "description": topic.description,
                "follower_count": topic.follower_count,
                "created_at": topic.created_at.isoformat() if topic.created_at else None,
                "updated_at": topic.updated_at.isoformat() if topic.updated_at else None
            },
            "items": items,
            "page": page,
            "limit": limit,
            "total": total,
            "has_more": has_more
        }
        
    except Exception as e:
        logger.error(f"Error in get_feeds_by_topic_summary for topic '{topic_name}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/source/{source_id}/feeds", response_model=dict)
def get_feeds_by_source(
    source_id: int,
    page: int = 1,
    limit: int = 20,
    include_unpublished: bool = False,  # New parameter to include unpublished feeds
    db: Session = Depends(get_db)
):
    """Get ALL feeds (published and unpublished) for a specific source."""
    # First, verify the source exists
    source = db.query(Source).filter(
        Source.id == source_id,
        Source.is_active == True
    ).first()
    
    if not source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Source with ID {source_id} not found"
        )
    
    # Build query based on source type
    if source.source_type == "blog":
        feeds_query = db.query(Feed).options(
            joinedload(Feed.blog),
            joinedload(Feed.slides),
            joinedload(Feed.category),  # Join category
            joinedload(Feed.subcategory)  # Join subcategory
        ).join(Blog).filter(
            Blog.website == source.website
        )
        
        if not include_unpublished:
            feeds_query = feeds_query.filter(Feed.status == "ready")
        
        total = feeds_query.count()
        feeds = feeds_query.order_by(Feed.created_at.desc()).offset((page - 1) * limit).limit(limit).all()
    
    else:  # youtube source
        # Get all YouTube feeds
        feeds_query = db.query(Feed).options(
            joinedload(Feed.blog),
            joinedload(Feed.slides),
            joinedload(Feed.category),  # Join category
            joinedload(Feed.subcategory)  # Join subcategory
        ).filter(
            Feed.source_type == "youtube"
        )
        
        if not include_unpublished:
            feeds_query = feeds_query.filter(Feed.status == "ready")
        
        # Get all feeds first for filtering
        all_feeds = feeds_query.order_by(Feed.created_at.desc()).all()
        
        # Filter to match channel name
        matching_feeds = []
        for feed in all_feeds:
            meta = get_feed_metadata(feed, db)
            if meta.get("channel_name") == source.name:
                matching_feeds.append(feed)
        
        # Manual pagination
        total = len(matching_feeds)
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        feeds = matching_feeds[start_idx:end_idx]
    
    # Format response with enhanced metadata
    items = []
    for feed in feeds:
        # Get category and subcategory names
        category_name = feed.category.name if feed.category else None
        subcategory_name = feed.subcategory.name if feed.subcategory else None
        
        # Get feed metadata
        meta = get_feed_metadata(feed, db)
        
        # Add category info to metadata
        meta_with_categories = {
            **meta,
            "category_name": category_name,
            "subcategory_name": subcategory_name,
            "category_id": feed.category_id,
            "subcategory_id": feed.subcategory_id,
            "category_display": f"{category_name} {{ {subcategory_name} }}" if category_name and subcategory_name else category_name,
        }
        
        # Format slides
        sorted_slides = sorted(feed.slides, key=lambda x: x.order) if feed.slides else []
        
        items.append({
            "id": feed.id,
            "title": feed.title,
            "categories": feed.categories,
            "status": feed.status,
            "source_type": feed.source_type or "blog",
            "meta": meta_with_categories,
            "slides_count": len(feed.slides),
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
            "category_name": category_name,
            "subcategory_name": subcategory_name,
            "category_id": feed.category_id,
            "subcategory_id": feed.subcategory_id,
            "category_display": f"{category_name} {{ {subcategory_name} }}" if category_name and subcategory_name else category_name,
            "created_at": feed.created_at.isoformat() if feed.created_at else None,
            "updated_at": feed.updated_at.isoformat() if feed.updated_at else None,
            "ai_generated": bool(feed.ai_generated_content)
        })
    
    has_more = (page * limit) < total
    
    # Get source metadata
    source_meta = get_source_metadata(source)
    
    # Build source response
    source_response = {
        "id": source.id,
        "name": source.name,
        "website": source.website,
        "source_type": source.source_type,
        "follower_count": getattr(source, 'follower_count', 0),
        "created_at": source.created_at.isoformat() if hasattr(source, 'created_at') and source.created_at else None,
        "updated_at": source.updated_at.isoformat() if hasattr(source, 'updated_at') and source.updated_at else None,
        "metadata": source_meta
    }
    
    # Add description if it exists
    if hasattr(source, 'description'):
        source_response["description"] = source.description
    
    return {
        "source": source_response,
        "items": items,
        "page": page,
        "limit": limit,
        "total": total,
        "has_more": has_more,
        "include_unpublished": include_unpublished
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
    
    # Get source metadata
    source_meta = get_source_metadata(source)
    
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
    
    # Build source response with safe attribute access
    source_response = {
        "id": source.id,
        "name": source.name,
        "website": source.website,
        "source_type": source.source_type,
        "follower_count": getattr(source, 'follower_count', 0),
        "created_at": source.created_at.isoformat() if hasattr(source, 'created_at') and source.created_at else None,
        "updated_at": source.updated_at.isoformat() if hasattr(source, 'updated_at') and source.updated_at else None,
        "metadata": source_meta
    }
    
    # Add description if it exists
    if hasattr(source, 'description'):
        source_response["description"] = source.description
    
    return {
        "source": source_response,
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

# @router.get("/source/{source_id}/feeds/summary", response_model=dict)
# def get_feeds_by_source_summary(
#     source_id: int,
#     page: int = 1,
#     limit: int = 20,
#     db: Session = Depends(get_db)
# ):
#     """Get feed summaries (without slides) for a specific source - faster for listing."""
#     source = db.query(Source).filter(
#         Source.id == source_id,
#         Source.is_active == True
#     ).first()
    
#     if not source:
#         raise HTTPException(status_code=404, detail=f"Source with ID {source_id} not found")
    
#     if source.source_type == "blog":
#         feeds_query = db.query(Feed).options(joinedload(Feed.blog)).join(Blog).filter(
#             Blog.website == source.website,
#             Feed.status == "ready"
#         ).order_by(Feed.created_at.desc())
        
#         total = feeds_query.count()
#         feeds = feeds_query.offset((page - 1) * limit).limit(limit).all()
    
#     else:  # youtube
#         feeds_query = db.query(Feed).options(joinedload(Feed.blog)).filter(
#             Feed.source_type == "youtube",
#             Feed.status == "ready"
#         ).order_by(Feed.created_at.desc())
        
#         all_feeds = feeds_query.all()
#         matching_feeds = []
        
#         for feed in all_feeds:
#             meta = get_feed_metadata(feed, db)
#             if meta.get("channel_name") == source.name:
#                 matching_feeds.append(feed)
        
#         total = len(matching_feeds)
#         start_idx = (page - 1) * limit
#         end_idx = start_idx + limit
#         feeds = matching_feeds[start_idx:end_idx]
    
#     items = []
#     for feed in feeds:
#         meta = get_feed_metadata(feed, db)
        
#         items.append({
#             "id": feed.id,
#             "title": feed.title,
#             "categories": feed.categories,
#             "source_type": feed.source_type,
#             "slides_count": len(feed.slides),
#             "meta": meta,
#             "created_at": feed.created_at.isoformat() if feed.created_at else None,
#             "ai_generated": bool(feed.ai_generated_content)
#         })
    
#     has_more = (page * limit) < total
    
#     return {
#         "source": {
#             "id": source.id,
#             "name": source.name,
#             "website": source.website,
#             "source_type": source.source_type,
#             "follower_count": source.follower_count
#         },
#         "items": items,
#         "page": page,
#         "limit": limit,
#         "total": total,
#         "has_more": has_more
#     }

@router.post("/topics/{topic_name}/update-description", response_model=dict)
def update_topic_description(
    topic_name: str,
    db: Session = Depends(get_db)
):
    """Update topic description based on current feed content."""
    topic = db.query(Topic).filter(Topic.name == topic_name).first()
    if not topic:
        raise HTTPException(status_code=404, detail=f"Topic '{topic_name}' not found")
    
    # Get feeds for this topic
    all_feeds = db.query(Feed).filter(
        Feed.status == "ready",
        Feed.categories.isnot(None)
    ).all()
    
    topic_feeds = []
    for feed in all_feeds:
        if feed.categories and topic_name in feed.categories:
            topic_feeds.append(feed)
    
    # Generate new description
    new_description = generate_topic_description(topic_name, topic_feeds, db)
    
    # Update topic
    topic.description = new_description
    topic.updated_at = datetime.utcnow()
    db.commit()
    
    return {
        "message": f"Topic description updated for {topic_name}",
        "topic": {
            "id": topic.id,
            "name": topic.name,
            "description": topic.description,
            "updated_at": topic.updated_at.isoformat()
        }
    }