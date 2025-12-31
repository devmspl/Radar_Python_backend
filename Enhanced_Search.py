# enhanced_search_router.py
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import or_, func, String, cast
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
import json  # Add this if not already present
from models import TranscriptJob, Transcript
from database import get_db
from dependencies import get_current_user
from models import (
    Feed, Blog, Transcript, Topic, Source, Concept, Domain, 
    ContentList, UserTopicFollow, UserSourceFollow, Bookmark,
    Category, SubCategory, FeedConcept, DomainConcept, Topic, User,
    UserOnboarding
)
from feed_router import get_feed_metadata
import os
import re
import requests

router = APIRouter(prefix="/search", tags=["Search"])
logger = logging.getLogger(__name__)

# ------------------ Helper Functions ------------------

def build_fuzzy_search_terms(query: str) -> List[str]:
    """
    Advanced fuzzy search with multiple strategies:
    1. Exact match
    2. Plural/singular variations
    3. Verb forms (ing, ed, er)
    4. Partial word matching (minimum 3 chars)
    5. Common typo tolerance
    6. Domain-specific synonyms
    
    Example: 'game' -> ['game', 'games', 'gaming', 'gamer', 'esport']
    """
    if not query or not query.strip():
        return []
    
    base_term = query.strip().lower()
    variations = set([base_term])
    
    # Strategy 1: Word variations (plural, verb forms)
    if len(base_term) >= 3:
        # Plural forms
        if not base_term.endswith('s'):
            variations.add(base_term + 's')
        if base_term.endswith('s') and len(base_term) > 3:
            variations.add(base_term[:-1])
        
        # Verb forms
        if not base_term.endswith('ing'):
            variations.add(base_term + 'ing')
            if base_term.endswith('e'):
                variations.add(base_term[:-1] + 'ing')  # code -> coding
        
        if base_term.endswith('ing') and len(base_term) > 4:
            variations.add(base_term[:-3])  # gaming -> game
            variations.add(base_term[:-3] + 'e')  # coding -> code
        
        # Agent forms
        if not base_term.endswith('er'):
            variations.add(base_term + 'er')
            if base_term.endswith('e'):
                variations.add(base_term + 'r')  # code -> coder
        
        # Past tense
        if not base_term.endswith('ed'):
            variations.add(base_term + 'ed')
            if base_term.endswith('e'):
                variations.add(base_term + 'd')  # code -> coded
    
    # Strategy 2: Partial matching (safer approach)
    # Only add prefixes if the term is reasonably long to avoid noise
    if len(base_term) >= 5:
        variations.add(base_term[:4])
    
    # Strategy 3: Common typo patterns
    typo_map = {
        'teh': 'the',
        'adn': 'and',
        'taht': 'that',
        'gamng': 'gaming',
        'codng': 'coding',
        'programing': 'programming',
        'devlopment': 'development'
    }
    
    if base_term in typo_map:
        variations.add(typo_map[base_term])
    
    # Strategy 4: Domain-specific synonyms
    synonym_map = {
        'game': ['gaming', 'esport', 'esports', 'gamer'],  # Removed 'play'
        'gaming': ['game', 'esport', 'esports', 'gamer'],
        'code': ['coding', 'programming', 'develop', 'software'],
        'coding': ['code', 'programming', 'develop', 'software'],
        'ai': ['artificial intelligence', 'machine learning', 'ml', 'deep learning'],
        'ml': ['machine learning', 'ai', 'artificial intelligence'],
        'web': ['website', 'internet', 'online', 'web development'],
        'mobile': ['app', 'smartphone', 'android', 'ios'],
        'data': ['database', 'analytics', 'big data', 'dataset'],
        'cloud': ['aws', 'azure', 'gcp', 'cloud computing'],
        'security': ['cybersecurity', 'infosec', 'cyber', 'secure'],
        'network': ['networking', 'internet', 'connectivity'],
        'design': ['ui', 'ux', 'graphic', 'creative'],
        'business': ['enterprise', 'corporate', 'commerce'],
        'finance': ['financial', 'money', 'banking', 'investment'],
        'health': ['healthcare', 'medical', 'wellness', 'fitness'],
        'education': ['learning', 'teaching', 'training', 'academic'],
        'tech': ['technology', 'technical', 'it'],
        'dev': ['development', 'developer', 'developing']
    }
    
    # Add synonyms for base term
    for key, synonyms in synonym_map.items():
        if base_term == key or base_term in synonyms:
            variations.update(synonyms)
            variations.add(key)
    
    # Strategy 5: Remove very short variations (less than 2 chars)
    variations = {v for v in variations if len(v) >= 2}
    
    return list(variations)

def extract_search_summary(feed: Feed) -> str:
    """Extract a summary from feed content for search results."""
    if feed.ai_generated_content and "summary" in feed.ai_generated_content:
        summary = feed.ai_generated_content["summary"]
        # Truncate to 200 characters
        if len(summary) > 200:
            return summary[:197] + "..."
        return summary
    
    # Fallback: use first slide body or empty string
    if feed.slides and len(feed.slides) > 0:
        first_slide = feed.slides[0]
        if len(first_slide.body) > 200:
            return first_slide.body[:197] + "..."
        return first_slide.body
    
    return ""

def format_feed_for_search(feed: Feed, db: Session, user_id: Optional[int] = None) -> Dict[str, Any]:
    """Format a feed for search results."""
    # Get metadata
    meta = get_feed_metadata(feed, db)
    
    # Check if bookmarked
    is_bookmarked = False
    if user_id:
        bookmark = db.query(Bookmark).filter(
            Bookmark.user_id == user_id,
            Bookmark.feed_id == feed.id
        ).first()
        is_bookmarked = bookmark is not None
    
    # Extract concepts
    concepts = []
    if hasattr(feed, 'concepts') and feed.concepts:
        concepts = [{"id": c.id, "name": c.name} for c in feed.concepts][:5]
    
    # Get source details
    source_info = {}
    if feed.source_type == "youtube":
        source_info = {
            "type": "youtube",
            "name": meta.get("channel_name", "YouTube"),
            "url": meta.get("source_url", "#")
        }
    elif feed.source_type == "blog" and feed.blog:
        source_info = {
            "type": "blog",
            "name": feed.blog.website,
            "url": feed.blog.website
        }
    
    return {
        "id": feed.id,
        "title": feed.title,
        "summary": extract_search_summary(feed),
        "content_type": feed.content_type.value if feed.content_type else "Video",
        "source_type": feed.source_type,
        "source_info": source_info,
        "categories": feed.categories or [],
        "concepts": concepts,
        "slides_count": len(feed.slides) if feed.slides else 0,
        "meta": meta,
        "is_bookmark": is_bookmarked,
        "created_at": feed.created_at.isoformat() if feed.created_at else None,
        "ai_generated": bool(feed.ai_generated_content)
    }

def auto_create_lists_from_youtube(db: Session):
    """Auto-create ContentList entries from existing YouTube feeds."""
    logger.info("Auto-creating lists from YouTube feeds...")
    
    # Instead of checking TranscriptJob, check existing feeds
    youtube_feeds = db.query(Feed).filter(
        Feed.source_type == "youtube",
        Feed.status == "ready"
    ).all()
    
    created_count = 0
    
    # Group by playlist
    playlist_groups = {}
    for feed in youtube_feeds:
        # Get transcript to find playlist_id
        if feed.transcript_id:
            transcript = db.query(Transcript).filter(
                Transcript.transcript_id == feed.transcript_id
            ).first()
            
            if transcript and transcript.playlist_id:
                playlist_id = transcript.playlist_id
                if playlist_id not in playlist_groups:
                    playlist_groups[playlist_id] = []
                playlist_groups[playlist_id].append(feed.id)
    
    # Create lists for playlists
    for playlist_id, feed_ids in playlist_groups.items():
        if len(feed_ids) >= 2:  # Only create if at least 2 feeds
            # Check if list already exists
            existing_list = db.query(ContentList).filter(
                ContentList.source_type == "youtube",
                ContentList.source_id == playlist_id
            ).first()
            
            if not existing_list:
                # Try to get playlist name
                playlist_name = f"YouTube Playlist {playlist_id}"
                
                # Create list
                new_list = ContentList(
                    name=playlist_name,
                    description=f"Auto-generated from YouTube playlist",
                    source_type="youtube",
                    source_id=playlist_id,
                    feed_ids=feed_ids,
                    is_active=True,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                db.add(new_list)
                created_count += 1
    
    # Group by channel
    channel_groups = {}
    for feed in youtube_feeds:
        meta = get_feed_metadata(feed, db)
        channel_id = meta.get("channel_id")
        channel_name = meta.get("channel_name")
        
        if channel_id and channel_name:
            if channel_id not in channel_groups:
                channel_groups[channel_id] = {
                    "name": f"{channel_name} Videos",
                    "channel_name": channel_name,
                    "feeds": []
                }
            channel_groups[channel_id]["feeds"].append(feed.id)
    
    # Create channel-based lists
    for channel_id, data in channel_groups.items():
        if len(data["feeds"]) >= 3:
            # Check if list exists
            existing_list = db.query(ContentList).filter(
                ContentList.source_type == "youtube",
                ContentList.source_id == f"channel_{channel_id}"
            ).first()
            
            if not existing_list:
                new_list = ContentList(
                    name=data["name"],
                    description=f"Videos from {data['channel_name']} YouTube channel",
                    source_type="youtube",
                    source_id=f"channel_{channel_id}",
                    feed_ids=data["feeds"],
                    is_active=True,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                db.add(new_list)
                created_count += 1
    
    try:
        db.commit()
        logger.info(f"Auto-created {created_count} lists from YouTube feeds")
        return created_count
    except Exception as e:
        db.rollback()
        logger.error(f"Error auto-creating lists: {str(e)}")
        return 0

# Make sure YOUTUBE_API_KEY is defined globally or passed to this function
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")


def format_list_details(content_list: ContentList, db: Session, user_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Format detailed information about a content list.
    """
    # Get feed details for statistics
    feed_ids = content_list.feed_ids or []
    feeds = []
    
    if feed_ids:
        feeds = db.query(Feed).options(
            joinedload(Feed.blog),
            joinedload(Feed.slides)
        ).filter(
            Feed.id.in_(feed_ids),
            Feed.status == "ready"
        ).all()
    
    # Calculate statistics
    total_feeds = len(feed_ids)
    active_feeds = len(feeds)
    
    # Extract unique topics from all feeds
    all_topics = set()
    for feed in feeds:
        if feed.categories:
            for category in feed.categories:
                all_topics.add(category)
    
    # Extract unique sources
    unique_sources = set()
    source_types = set()
    
    for feed in feeds:
        if feed.source_type:
            source_types.add(feed.source_type)
            
            if feed.source_type == "youtube":
                meta = get_feed_metadata(feed, db)
                channel_name = meta.get("channel_name")
                if channel_name:
                    unique_sources.add(f"YouTube: {channel_name}")
            elif feed.source_type == "blog" and feed.blog:
                unique_sources.add(f"Blog: {feed.blog.website}")
    
    # Calculate total slides
    total_slides = sum(len(feed.slides) for feed in feeds)
    
    # Get list type based on source
    list_type = "playlist" if content_list.source_type == "youtube" else "collection"
    if content_list.source_type == "blog":
        list_type = "blog_collection"
    elif content_list.source_type == "manual":
        list_type = "manual_collection"
    
    # Get thumbnail from first feed
    thumbnail_url = None
    if feeds:
        meta = get_feed_metadata(feeds[0], db)
        thumbnail_url = meta.get("thumbnail_url") or meta.get("favicon")
    
    # Get playlist/channel info for YouTube lists
    youtube_info = None
    if content_list.source_type == "youtube":
        # Extract playlist ID from source_id
        if content_list.source_id and content_list.source_id.startswith("PL"):
            # It's a playlist
            youtube_info = {
                "type": "playlist",
                "playlist_id": content_list.source_id,
                "url": f"https://www.youtube.com/playlist?list={content_list.source_id}"
            }
        elif content_list.source_id and content_list.source_id.startswith("channel_"):
            # It's a channel
            channel_id = content_list.source_id.replace("channel_", "")
            youtube_info = {
                "type": "channel",
                "channel_id": channel_id,
                "url": f"https://www.youtube.com/channel/{channel_id}"
            }
    
    # Get list creator info (if available)
    creator_info = None
    if hasattr(content_list, 'user_id') and content_list.user_id:
        creator = db.query(User).filter(User.id == content_list.user_id).first()
        if creator:
            creator_info = {
                "id": creator.id,
                "name": creator.full_name or creator.username,
                "email": creator.email
            }
    
    # Check if user has bookmarked any feeds in this list
    user_bookmarked_feeds = []
    if user_id:
        bookmarks = db.query(Bookmark).filter(
            Bookmark.user_id == user_id,
            Bookmark.feed_id.in_(feed_ids)
        ).all()
        user_bookmarked_feeds = [bm.feed_id for bm in bookmarks]
    
    return {
        "id": content_list.id,
        "name": content_list.name,
        "description": content_list.description,
        "source_type": content_list.source_type,
        "source_id": content_list.source_id,
        "list_type": list_type,
        "feed_ids": feed_ids,
        "is_active": content_list.is_active,
        "created_at": content_list.created_at.isoformat() if content_list.created_at else None,
        "updated_at": content_list.updated_at.isoformat() if content_list.updated_at else None,
        "statistics": {
            "total_feeds": total_feeds,
            "active_feeds": active_feeds,
            "total_slides": total_slides,
            "unique_topics": len(all_topics),
            "unique_sources": len(unique_sources),
            "source_types": list(source_types)
        },
        "top_topics": list(all_topics)[:10],  # Top 10 topics
        "top_sources": list(unique_sources)[:5],  # Top 5 sources
        "youtube_info": youtube_info,
        "creator_info": creator_info,
        "thumbnail_url": thumbnail_url,
        "user_stats": {
            "bookmarked_feeds": user_bookmarked_feeds,
            "bookmarked_count": len(user_bookmarked_feeds)
        } if user_id else None
    }


def get_source_metadata(source: Source, db: Session) -> Dict[str, Any]:
    """Get metadata for a source."""
    # Initialize metadata with basic info
    metadata = {
        "name": source.name,
        "website": source.website,
        "source_type": source.source_type,
        "description": getattr(source, 'description', ''),
        "favicon": None,
        "thumbnail": None,
        "channel_info": None,
        "top_feeds": []
    }
    
    # Get sample feeds for description and content analysis
    if source.source_type == "blog":
        sample_feeds = db.query(Feed).join(Blog).filter(
            Blog.website == source.website,
            Feed.status == "ready"
        ).order_by(Feed.created_at.desc()).limit(5).all()
    else:
        # For YouTube, get recent feeds from this source type
        sample_feeds = db.query(Feed).filter(
            Feed.source_type == "youtube",
            Feed.status == "ready"
        ).order_by(Feed.created_at.desc()).limit(5).all()
    
    # Generate description if empty
    if not metadata["description"] and sample_feeds:
        from collections import Counter
        all_topics = []
        for feed in sample_feeds:
            if feed.categories:
                all_topics.extend(feed.categories)
        
        if all_topics:
            topic_counter = Counter(all_topics)
            top_topics = topic_counter.most_common(3)
            
            if source.source_type == "youtube":
                metadata["description"] = f"YouTube channel covering {', '.join([topic for topic, _ in top_topics])}"
            else:
                metadata["description"] = f"Website covering {', '.join([topic for topic, _ in top_topics])}"
        else:
            metadata["description"] = f"YouTube channel {source.name}" if source.source_type == "youtube" else f"Website: {source.name}"
    
    # Handle YouTube-specific metadata
    if source.source_type == "youtube":
        # Extract Channel ID from the website URL
        channel_id = None
        
        # Pattern for standard channel URLs
        pattern = r'youtube\.com/channel/([a-zA-Z0-9_-]{24})'
        match = re.search(pattern, source.website)
        
        if match:
            channel_id = match.group(1)
            logger.info(f"Extracted channel ID: {channel_id} from URL: {source.website}")
        else:
            # Check if it's a custom URL format
            patterns = [
                r'youtube\.com/c/([a-zA-Z0-9_-]+)',
                r'youtube\.com/user/([a-zA-Z0-9_-]+)',
                r'youtube\.com/@([a-zA-Z0-9_-]+)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, source.website)
                if match:
                    # For custom URLs, we need to search for the channel
                    custom_name = match.group(1)
                    logger.info(f"Found custom YouTube URL: {custom_name}")
                    # You could implement search by custom name here
                    break
        
        # Call YouTube Data API with the extracted Channel ID
        if channel_id and YOUTUBE_API_KEY:
            try:
                api_url = "https://www.googleapis.com/youtube/v3/channels"
                params = {
                    'key': YOUTUBE_API_KEY,
                    'id': channel_id,
                    'part': 'snippet,statistics,brandingSettings'
                }
                
                logger.info(f"Calling YouTube API for channel ID: {channel_id}")
                response = requests.get(api_url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"YouTube API response: {data}")
                    
                    if data.get('items') and len(data['items']) > 0:
                        channel_data = data['items'][0]
                        snippet = channel_data.get('snippet', {})
                        stats = channel_data.get('statistics', {})
                        branding = channel_data.get('brandingSettings', {}).get('image', {})
                        
                        # Get the highest quality thumbnail
                        thumbnails = snippet.get('thumbnails', {})
                        thumbnail_url = None
                        for res in ['high', 'medium', 'default']:
                            if thumbnails.get(res):
                                thumbnail_url = thumbnails[res]['url']
                                break
                        
                        # Get banner image if available
                        banner_url = None
                        if branding.get('bannerExternalUrl'):
                            banner_url = branding['bannerExternalUrl']
                        
                        metadata.update({
                            "favicon": "https://www.youtube.com/s/desktop/12d6b690/img/favicon.ico",
                            "thumbnail": thumbnail_url,
                            "banner": banner_url,
                            "channel_info": {
                                "channel_id": channel_id,
                                "title": snippet.get('title', source.name),
                                "description": snippet.get('description', ''),
                                "published_at": snippet.get('publishedAt'),
                                "country": snippet.get('country'),
                                "subscriber_count": stats.get('subscriberCount'),
                                "video_count": stats.get('videoCount'),
                                "view_count": stats.get('viewCount'),
                                "available": True
                            }
                        })
                        logger.info(f"Successfully fetched channel info for {source.name}")
                    else:
                        logger.warning(f"No channel data found for ID: {channel_id}")
                        metadata["channel_info"] = {"available": False, "error": "Channel not found"}
                else:
                    logger.error(f"YouTube API error: {response.status_code} - {response.text}")
                    metadata["channel_info"] = {"available": False, "error": f"API error {response.status_code}"}
                    
            except Exception as e:
                logger.error(f"Failed to fetch YouTube channel info for {source.website}: {e}")
                metadata["channel_info"] = {"available": False, "error": str(e)}
        else:
            if not channel_id:
                logger.warning(f"Could not extract channel ID from URL: {source.website}")
            if not YOUTUBE_API_KEY:
                logger.warning("YOUTUBE_API_KEY not configured")
            metadata["channel_info"] = {"available": False, "error": "Missing channel ID or API key"}
    
    # Handle blog sources
    elif source.source_type == "blog":
        # Get website favicon
        try:
            from urllib.parse import urlparse
            website_url = source.website
            if not website_url.startswith(('http://', 'https://')):
                website_url = f"https://{website_url}"
            
            parsed_url = urlparse(website_url)
            domain = parsed_url.netloc or parsed_url.path.split('/')[0]
            metadata["favicon"] = f"https://www.google.com/s2/favicons?domain={domain}&sz=64"
        except Exception as e:
            logger.error(f"Failed to generate favicon URL for {source.website}: {e}")
            metadata["favicon"] = None
    
    # Get top feeds for display (for both YouTube and blog)
    for feed in sample_feeds[:3]:  # Only first 3
        try:
            # Get feed metadata
            feed_meta = get_feed_metadata(feed, db)
            
            # Extract summary
            summary = ""
            if feed.ai_generated_content and "summary" in feed.ai_generated_content:
                summary = feed.ai_generated_content["summary"]
                if len(summary) > 100:
                    summary = summary[:97] + "..."
            elif feed.slides and len(feed.slides) > 0:
                summary = feed.slides[0].body
                if len(summary) > 100:
                    summary = summary[:97] + "..."
            
            # Get thumbnail
            thumbnail = None
            if source.source_type == "youtube":
                thumbnail = feed_meta.get("thumbnail_url") or feed_meta.get("channel_thumbnail")
            else:
                thumbnail = feed_meta.get("favicon")
            
            metadata["top_feeds"].append({
                "id": feed.id,
                "title": feed.title,
                "summary": summary,
                "thumbnail": thumbnail
            })
        except Exception as e:
            logger.error(f"Error processing feed {feed.id} for source metadata: {e}")
    
    return metadata
# ------------------ Tab 1: Content Search ------------------

@router.get("/content", response_model=Dict[str, Any])
def search_content(
    query: Optional[str] = Query(None, description="Search query"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Results per page"),
    content_type: Optional[str] = Query(None, description="Filter by content type"),
    source_type: Optional[str] = Query(None, description="Filter by source type"),
    domain: Optional[str] = Query(None, description="Filter by domain"),
    topic: Optional[str] = Query(None, description="Filter by topic"),
    concept: Optional[str] = Query(None, description="Filter by concept"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Advanced content search with intelligent scoring and fuzzy matching.
    
    Features:
    - NO QUERY: Returns feeds from user's selected categories (latest first)
    - WITH QUERY: Smart scoring based on:
        * Title match (highest weight)
        * AI summary match
        * Categories/topics match
        * Subcategory match
        * Source followed by user
        * Bookmarked by user
        * Freshness (recent content)
    - Case-insensitive fuzzy search
    - Results sorted by relevance score
    """
    user_id = current_user.id
    
    # Step 1: Get user's onboarding data
    user_onboarding = db.query(UserOnboarding).filter(
        UserOnboarding.user_id == user_id
    ).first()
    
    if not user_onboarding:
        raise HTTPException(
            status_code=404,
            detail="User onboarding data not found. Please complete onboarding first."
        )
    
    user_category_ids = user_onboarding.domains_of_interest or []
    
    if not user_category_ids or len(user_category_ids) == 0:
        return {
            "tab": "content",
            "query": query,
            "items": [],
            "page": page,
            "limit": limit,
            "total": 0,
            "has_more": False,
            "message": "No categories selected in onboarding. Please select categories first."
        }
    
    # Step 2: Get user's personalization data
    followed_sources = db.query(UserSourceFollow).filter(
        UserSourceFollow.user_id == user_id
    ).all()
    followed_source_ids = [fs.source_id for fs in followed_sources]
    
    bookmarked_feeds = db.query(Bookmark).filter(
        Bookmark.user_id == user_id
    ).all()
    bookmarked_feed_ids = [bf.feed_id for bf in bookmarked_feeds]
    
    # Step 3: Base query - Get ALL feeds from user's categories
    base_query = db.query(Feed).options(
        joinedload(Feed.blog),
        joinedload(Feed.slides),
        joinedload(Feed.category),
        joinedload(Feed.subcategory),
        joinedload(Feed.concepts)
    ).filter(
        Feed.category_id.in_(user_category_ids),
        Feed.status == "ready"
    )
    
    # Apply hard filters (these are mandatory)
    if content_type:
        base_query = base_query.filter(Feed.content_type == content_type)
    
    if source_type:
        base_query = base_query.filter(Feed.source_type == source_type)
    
    if topic and topic.strip():
        base_query = base_query.filter(Feed.categories.contains([topic.strip()]))
    
    if concept and concept.strip():
        concept_obj = db.query(Concept).filter(
            Concept.name.ilike(f"%{concept}%")
        ).first()
        if concept_obj:
            feed_ids = db.query(FeedConcept.feed_id).filter(
                FeedConcept.concept_id == concept_obj.id
            ).all()
            if feed_ids:
                feed_id_list = [fid[0] for fid in feed_ids]
                base_query = base_query.filter(Feed.id.in_(feed_id_list))
    
    if domain and domain.strip():
        domain_obj = db.query(Domain).filter(
            Domain.name.ilike(f"%{domain}%")
        ).first()
        if domain_obj:
            domain_concepts = db.query(DomainConcept).filter(
                DomainConcept.domain_id == domain_obj.id
            ).all()
            concept_ids = [dc.concept_id for dc in domain_concepts]
            
            feed_ids = db.query(FeedConcept.feed_id).filter(
                FeedConcept.concept_id.in_(concept_ids)
            ).distinct().all()
            
            if feed_ids:
                feed_id_list = [fid[0] for fid in feed_ids]
                base_query = base_query.filter(Feed.id.in_(feed_id_list))
    
    # Step 4: Get all matching feeds
    all_feeds = base_query.all()
    
    # Step 5: Score and filter based on query
    scored_feeds = []
    
    if not query or not query.strip():
        # NO QUERY: Just return latest feeds
        for feed in all_feeds:
            scored_feeds.append((feed, 0.0))  # No scoring needed
    else:
        # WITH QUERY: Apply fuzzy search and scoring
        search_variations = build_fuzzy_search_terms(query)
        
        for feed in all_feeds:
            score = 0.0
            matched = False
            
            # Prepare searchable text
            title = (feed.title or "").lower()
            categories_text = " ".join(feed.categories or []).lower()
            subcategory_name = feed.subcategory.name.lower() if feed.subcategory else ""
            
            # AI summary
            ai_summary = ""
            if feed.ai_generated_content and isinstance(feed.ai_generated_content, dict):
                ai_summary = (feed.ai_generated_content.get("summary", "") or "").lower()
            
            # Check each search variation
            for variation in search_variations:
                variation_lower = variation.lower()
                
                # Title match (50 points - highest priority)
                if variation_lower in title:
                    score += 50.0
                    matched = True
                    if title.startswith(variation_lower):
                        score += 10.0  # Bonus for title starting with query
                
                # AI Summary match (30 points)
                if variation_lower in ai_summary:
                    score += 30.0
                    matched = True
                
                # Categories/Topics match (25 points)
                if variation_lower in categories_text:
                    score += 25.0
                    matched = True
                
                # Subcategory match (20 points)
                if variation_lower in subcategory_name:
                    score += 20.0
                    matched = True
            
            # Only include if matched
            if not matched:
                continue
            
            # Personalization bonuses
            # Bookmarked (15 points)
            if feed.id in bookmarked_feed_ids:
                score += 15.0
            
            # Source followed (10 points)
            if feed.source_type == "blog" and feed.blog:
                source = db.query(Source).filter(
                    Source.source_type == "blog",
                    Source.website == feed.blog.website
                ).first()
                if source and source.id in followed_source_ids:
                    score += 10.0
            
            # Freshness bonus (up to 10 points)
            if feed.created_at:
                days_old = (datetime.utcnow() - feed.created_at).days
                if days_old < 7:
                    score += 10.0
                elif days_old < 30:
                    score += 5.0
                elif days_old < 90:
                    score += 2.0
            
            scored_feeds.append((feed, round(score, 2)))
    
    # Fallback: If query is very short (2 chars) and no matches, return latest feeds
    if query and query.strip() and len(query.strip()) <= 2 and len(scored_feeds) == 0:
        for feed in all_feeds[:limit * 2]:  # Get double the limit for better results
            scored_feeds.append((feed, 0.0))
    
    # Step 6: Sort by score (highest first) or date (if no query)
    if query and query.strip():
        scored_feeds.sort(key=lambda x: x[1], reverse=True)
    else:
        scored_feeds.sort(key=lambda x: x[0].created_at, reverse=True)
    
    # Step 7: Paginate
    total = len(scored_feeds)
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated_feeds = scored_feeds[start_idx:end_idx]
    
    # Step 8: Format response
    items = []
    for feed, score in paginated_feeds:
        feed_data = format_feed_for_search(feed, db, user_id)
        if query and query.strip():
            feed_data["relevance_score"] = score
        items.append(feed_data)
    
    return {
        "tab": "content",
        "query": query,
        "items": items,
        "page": page,
        "limit": limit,
        "total": total,
        "has_more": (page * limit) < total,
        "user_selected_categories": user_category_ids,
        "filters": {
            "content_type": content_type,
            "source_type": source_type,
            "domain": domain,
            "topic": topic,
            "concept": concept
        }
    }


# ------------------ Tab 2: Lists Search ------------------


@router.get("/lists", response_model=Dict[str, Any])
def search_lists(
    query: Optional[str] = Query(None, description="Search query"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Results per page"),
    source_type: Optional[str] = Query(None, description="Filter by source type: youtube, blog, or manual"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    COMPREHENSIVE SEARCH PIPELINE for playlists.
    Scans: Playlist Name/Desc, Feed Titles, Summaries, Category Names, Subcategory Names, 
    Topics, and Channel/Source Names.
    """
    user_id = current_user.id
    
    # 1. Fetch User Context for Prioritization
    user_onboarding = db.query(UserOnboarding).filter(UserOnboarding.user_id == user_id).first()
    user_category_ids = user_onboarding.domains_of_interest or [] if user_onboarding else []
    bookmarked_fids = {b.feed_id for b in db.query(Bookmark).filter(Bookmark.user_id == user_id).all()}

    # 2. Fetch All Lists
    lists_query = db.query(ContentList).filter(ContentList.is_active == True)
    if source_type:
        lists_query = lists_query.filter(ContentList.source_type == source_type)
    
    all_lists = lists_query.all()
    if not all_lists and db.query(ContentList).count() == 0:
        auto_create_lists_from_youtube(db)
        all_lists = lists_query.all()

    # 3. GLOBAL METADATA PIPELINE (Batch Fetch)
    all_fids = set()
    for cl in all_lists:
        for fid in (cl.feed_ids or []):
            try: all_fids.add(int(fid))
            except: continue
    
    feed_data_map = {}
    if all_fids:
        # Transcript has 'title' (Original Title), Blog has 'website'
        feeds_data = db.query(
            Feed.id, Feed.title, Feed.category_id, Feed.categories, Feed.ai_generated_content,
            Feed.source_type, Category.name.label("cat_name"), SubCategory.name.label("sub_name"),
            Transcript.title.label("orig_title"), Blog.website
        ).outerjoin(Category, Feed.category_id == Category.id)\
         .outerjoin(SubCategory, Feed.subcategory_id == SubCategory.id)\
         .outerjoin(Transcript, Feed.transcript_id == Transcript.transcript_id)\
         .outerjoin(Blog, Feed.blog_id == Blog.id)\
         .filter(Feed.id.in_(list(all_fids))).all()
        
        for f in feeds_data:
            summary = (f.ai_generated_content or {}).get("summary", "")
            source_name = f.website if f.source_type == "blog" else ""
            
            feed_data_map[f.id] = {
                "title": f.title or "",
                "summary": summary or "",
                "cat_name": f.cat_name or "",
                "sub_name": f.sub_name or "",
                "topics": " ".join(f.categories or []),
                "source_name": source_name or "",
                "orig_title": f.orig_title or "",
                "cat_id": f.category_id
            }

    # 4. Relational Search Engineering
    scored_items = []
    q_lower = query.strip().lower() if query else None
    
    for cl in all_lists:
        search_buffer_parts = [str(cl.name or ""), str(cl.description or ""), str(cl.source_id or "")]
        
        playlist_fids = []
        for fid in (cl.feed_ids or []):
            try: playlist_fids.append(int(fid))
            except: continue

        list_cat_ids = set()
        for fid in playlist_fids:
            if fid in feed_data_map:
                fd = feed_data_map[fid]
                search_buffer_parts.extend([
                    fd["title"], fd["summary"], fd["cat_name"], 
                    fd["sub_name"], fd["topics"], fd["source_name"], fd["orig_title"]
                ])
                if fd["cat_id"]: list_cat_ids.add(fd["cat_id"])

        search_vector = " ".join(search_buffer_parts).lower()
        
        s_score = 0.0
        is_match = False
        reasons = set()

        # CASE 1: Empty Query - Default Prioritization
        if not q_lower:
            s_score = float(len(playlist_fids)) * 1.0
            is_match = True
        # CASE 2: Pipeline Search
        else:
            # Tier 1: Exact Name Match
            if q_lower in (cl.name or "").lower():
                s_score += 500.0; is_match = True; reasons.add("Playlist Title Match")
            
            # Tier 2: Deep Vector Match (Categories, Sources, Content)
            if q_lower in search_vector:
                s_score += 150.0; is_match = True; reasons.add("Content/Source Match")
            
            # Tier 3: Individual Check for Category/Source specifically to boost
            for fid in playlist_fids:
                if fid in feed_data_map:
                    fd = feed_data_map[fid]
                    if q_lower in fd["cat_name"].lower() or q_lower in fd["sub_name"].lower():
                        s_score += 100.0; reasons.add("Category Match")
                    if q_lower in fd["source_name"].lower():
                        s_score += 120.0; reasons.add("Channel/Source Match")

        if is_match:
            # User Personalization Bonus
            cat_bonus = sum(100 for cid in list_cat_ids if cid in user_category_ids)
            s_score += min(cat_bonus, 500) # Cap onboarding bonus
            if cat_bonus > 0: reasons.add("Recommended based on your interests")
            
            has_bookmark = any(fid in bookmarked_fids for fid in playlist_fids)
            if has_bookmark: s_score += 50.0; reasons.add("Contains your bookmarks")

            scored_items.append({
                "list": cl, "score": s_score, "fcount": len(playlist_fids),
                "cats": list(set(fd["cat_name"] for fid in playlist_fids if fid in feed_data_map and feed_data_map[fid]["cat_name"])),
                "is_bookmark": has_bookmark, "reasons": list(reasons)
            })

    # 5. Result Construction
    scored_items.sort(key=lambda x: x["score"], reverse=True)
    total = len(scored_items)
    paginated = scored_items[(page - 1) * limit : page * limit]

    items_list = []
    for item in paginated:
        cl = item["list"]
        samples = db.query(Feed).filter(Feed.id.in_(cl.feed_ids[:3] or [])).all()
        sample_feeds = []
        for f in samples:
            meta = get_feed_metadata(f, db)
            summary = (f.ai_generated_content or {}).get("summary", "")
            sample_feeds.append({"id": f.id, "title": f.title, "summary": summary[:120], "meta": meta})

        items_list.append({
            "id": cl.id,
            "name": cl.name,
            "description": cl.description,
            "source_type": cl.source_type,
            "source_id": cl.source_id,
            "list_type": "channel" if (cl.source_id or "").startswith("channel_") or (cl.source_id or "").startswith("UU") else "playlist",
            "feed_count": item["fcount"],
            "categories": item["cats"],
            "is_bookmark": item["is_bookmark"],
            "sample_feeds": sample_feeds,
            "thumbnail_url": sample_feeds[0]["meta"].get("thumbnail_url") if sample_feeds else None,
            "relevance_score": item["score"],
            "match_reasons": item["reasons"]
        })

    return {
        "tab": "lists", "query": query, "items": items_list,
        "page": page, "limit": limit, "total": total, "has_more": (page * limit) < total,
        "user_selected_categories": user_category_ids
    }


# ------------------ Tab 3: Topics Search ------------------

@router.get("/topics", response_model=Dict[str, Any])
def search_topics(
    query: Optional[str] = Query(None, description="Search query"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Results per page"),
    domain: Optional[str] = Query(None, description="Filter by domain"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Search topics extracted from content via LLM.
    
    This is for Tab 3: Topics
    """
    user_id = current_user.id
    
    # Step 1: Get user's onboarding data to find selected categories
    user_onboarding = db.query(UserOnboarding).filter(
        UserOnboarding.user_id == user_id
    ).first()
    
    if not user_onboarding:
        raise HTTPException(
            status_code=404, detail="User onboarding data not found. Please complete onboarding first."
        )
    
    # Get user's domains of interest (category IDs)
    user_category_ids = user_onboarding.domains_of_interest
    
    if not user_category_ids or len(user_category_ids) == 0:
        return {
            "tab": "topics",
            "query": query,
            "items": [],
            "page": page,
            "limit": limit,
            "total": 0,
            "has_more": False,
            "message": "No categories selected in onboarding. Please select categories first."
        }
    
    # Step 2 & 3: Get feeds matching user categories and search query
    query_obj = db.query(Feed).filter(
        Feed.category_id.in_(user_category_ids),
        Feed.status == "ready",
        Feed.categories.isnot(None)
    )
    
    if query and query.strip():
        search_term = f"%{query.strip().lower()}%"
        query_obj = query_obj.filter(cast(Feed.categories, String).ilike(search_term))
        
    # Step 4: Get all relevant feeds to extract topics
    all_feeds = query_obj.all()
    
    # Step 5: Extract unique topics and calculate counts
    topics_dict = {}
    
    # Get followed topics for status
    followed_topics = db.query(UserTopicFollow).filter(UserTopicFollow.user_id == user_id).all()
    followed_topic_ids = [ft.topic_id for ft in followed_topics]
    
    for feed in all_feeds:
        if feed.categories:
            for category_name in feed.categories:
                if category_name not in topics_dict:
                    # Filter by search query if present
                    if query and query.strip() and query.strip().lower() not in category_name.lower():
                        continue
                        
                    # Count feeds for this topic within user categories
                    feed_count = db.query(func.count(Feed.id)).filter(
                        Feed.categories.contains([category_name]),
                        Feed.category_id.in_(user_category_ids),
                        Feed.status == "ready"
                    ).scalar() or 0
                    
                    if feed_count == 0:
                        continue
                        
                    # Get topic metadata
                    topic = db.query(Topic).filter(Topic.name == category_name).first()
                    topic_id = topic.id if topic else None
                    follower_count = topic.follower_count if topic else 0
                    description = topic.description if topic else f"Content related to {category_name}"
                    
                    # Get unique sources for this topic within user categories
                    topic_feeds_sample = db.query(Feed).filter(
                        Feed.categories.contains([category_name]),
                        Feed.category_id.in_(user_category_ids),
                        Feed.status == "ready"
                    ).limit(10).all()
                    
                    unique_sources = set()
                    for t_feed in topic_feeds_sample:
                        if t_feed.source_type == "youtube":
                            meta = get_feed_metadata(t_feed, db)
                            if meta.get("channel_name"): unique_sources.add(meta["channel_name"])
                        elif t_feed.source_type == "blog" and t_feed.blog:
                            unique_sources.add(t_feed.blog.website)
                    
                    topics_dict[category_name] = {
                        "id": topic_id,
                        "name": category_name,
                        "description": description,
                        "feed_count": feed_count,
                        "follower_count": follower_count,
                        "is_following": topic_id in followed_topic_ids if topic_id else False,
                        "sources": list(unique_sources)[:5]
                    }
    
    # Step 6: Sort by feed_count (descending)
    topic_list = list(topics_dict.values())
    topic_list.sort(key=lambda x: x["feed_count"], reverse=True)
    
    # Step 7: Paginate results
    total = len(topic_list)
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated_items = topic_list[start_idx:end_idx]
    
    return {
        "tab": "topics",
        "query": query,
        "items": paginated_items,
        "page": page,
        "limit": limit,
        "total": total,
        "has_more": (page * limit) < total,
        "user_selected_categories": user_category_ids
    }


# ------------------ Tab 4: Sources Search ------------------

@router.get("/sources", response_model=Dict[str, Any])
def search_sources(
    query: Optional[str] = Query(None, description="Search query"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Results per page"),
    source_type: Optional[str] = Query(None, description="Filter by source type"),
    domain: Optional[str] = Query(None, description="Filter by domain"),
    topic: Optional[str] = Query(None, description="Filter by topic"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Search sources (websites, YouTube channels, authors).
    
    Logic:
    - When query is EMPTY: Return sources from feeds in user's selected categories and subcategories
    - When query has VALUE: 
        1. First search Source table directly
        2. If not found, search feeds and extract sources from matched feeds
    
    This is for Tab 4: Sources
    """
    user_id = current_user.id
    
    # Step 1: Get user's onboarding data to find selected categories
    user_onboarding = db.query(UserOnboarding).filter(
        UserOnboarding.user_id == user_id
    ).first()
    
    if not user_onboarding:
        raise HTTPException(
            status_code=404,
            detail="User onboarding data not found. Please complete onboarding first."
        )
    
    # Get user's domains of interest (category IDs)
    user_category_ids = user_onboarding.domains_of_interest
    
    if not user_category_ids or len(user_category_ids) == 0:
        return {
            "tab": "sources",
            "query": query,
            "items": [],
            "page": page,
            "limit": limit,
            "total": 0,
            "has_more": False,
            "message": "No categories selected in onboarding. Please select categories first."
        }
    
    # Get all followed sources for status checking later
    followed_sources = db.query(UserSourceFollow).filter(UserSourceFollow.user_id == user_id).all()
    followed_source_ids = [fs.source_id for fs in followed_sources]
    
    # Step 2: Get all subcategories linked to user's selected categories
    user_subcategories = db.query(SubCategory).filter(
        SubCategory.category_id.in_(user_category_ids)
    ).all()
    user_subcategory_ids = [sc.id for sc in user_subcategories]
    
    # Dictionary to store unique sources
    sources_dict = {}
    
    # === CASE 1: Query is EMPTY ===
    if not query or not query.strip():
        # Get all feeds from user's selected categories and subcategories
        feeds_query = db.query(Feed).filter(
            Feed.category_id.in_(user_category_ids),
            Feed.status == "ready"
        )
        
        # Also filter by subcategory if available
        if user_subcategory_ids:
            feeds_query = feeds_query.filter(
                or_(
                    Feed.subcategory_id.in_(user_subcategory_ids),
                    Feed.subcategory_id.is_(None)  # Include feeds without subcategory
                )
            )
        
        # Apply source_type filter if provided
        if source_type:
            feeds_query = feeds_query.filter(Feed.source_type == source_type)
        
        all_feeds = feeds_query.all()
        
        # Extract unique sources from feeds
        for feed in all_feeds:
            source_key = None
            source_obj = None
            
            if feed.source_type == "blog" and feed.blog:
                source_key = f"blog_{feed.blog.website}"
                # Find or create source object
                source_obj = db.query(Source).filter(
                    Source.source_type == "blog",
                    Source.website == feed.blog.website
                ).first()
                
            elif feed.source_type == "youtube":
                # Extract channel info from metadata
                meta = get_feed_metadata(feed, db)
                channel_name = meta.get("channel_name")
                source_url = meta.get("source_url")
                
                if channel_name:
                    source_key = f"youtube_{channel_name}"
                    # Find or create source object
                    source_obj = db.query(Source).filter(
                        Source.source_type == "youtube",
                        Source.name == channel_name
                    ).first()
            
            # If source found and not already in dictionary, add it
            if source_key and source_key not in sources_dict:
                if source_obj:
                    sources_dict[source_key] = {
                        "source": source_obj,
                        "feeds": []
                    }
                else:
                    # Create temporary source object for sources not in Source table
                    if feed.source_type == "blog" and feed.blog:
                        sources_dict[source_key] = {
                            "source": type('obj', (object,), {
                                'id': None,
                                'name': feed.blog.website,
                                'website': feed.blog.website,
                                'source_type': 'blog',
                                'description': '',
                                'follower_count': 0,
                                'created_at': None,
                                'updated_at': None
                            })(),
                            "feeds": []
                        }
                    elif feed.source_type == "youtube":
                        meta = get_feed_metadata(feed, db)
                        channel_name = meta.get("channel_name", "Unknown Channel")
                        sources_dict[source_key] = {
                            "source": type('obj', (object,), {
                                'id': None,
                                'name': channel_name,
                                'website': meta.get("source_url", ""),
                                'source_type': 'youtube',
                                'description': '',
                                'follower_count': 0,
                                'created_at': None,
                                'updated_at': None
                            })(),
                            "feeds": []
                        }
            
            # Add feed to this source's feed list
            if source_key and source_key in sources_dict:
                sources_dict[source_key]["feeds"].append(feed)
    
    # === CASE 2: Query has VALUE ===
    else:
        search_term = f"%{query.strip().lower()}%"
        
        # Step 2a: First search directly in Source table
        direct_sources_query = db.query(Source).filter(
            Source.is_active == True,
            or_(
                Source.name.ilike(search_term),
                Source.website.ilike(search_term)
            )
        )
        
        if source_type:
            direct_sources_query = direct_sources_query.filter(Source.source_type == source_type)
        
        direct_sources = direct_sources_query.all()
        
        # If sources found directly, use them
        if direct_sources:
            for source_obj in direct_sources:
                source_key = f"{source_obj.source_type}_{source_obj.name}"
                
                # Get feeds for this source within user's categories
                if source_obj.source_type == "blog":
                    feeds = db.query(Feed).join(Blog).filter(
                        Blog.website == source_obj.website,
                        Feed.category_id.in_(user_category_ids),
                        Feed.status == "ready"
                    ).all()
                else:
                    feeds = db.query(Feed).filter(
                        Feed.source_type == "youtube",
                        Feed.category_id.in_(user_category_ids),
                        Feed.status == "ready"
                    ).all()
                
                # Only include sources that have feeds in user's categories
                if feeds:
                    sources_dict[source_key] = {
                        "source": source_obj,
                        "feeds": feeds
                    }
        
        # Step 2b: If no direct sources found, search in feeds
        else:
            # Search for feeds matching the query
            feeds_query = db.query(Feed).filter(
                Feed.category_id.in_(user_category_ids),
                Feed.status == "ready",
                or_(
                    Feed.title.ilike(search_term),
                    cast(Feed.categories, String).ilike(search_term)
                )
            )
            
            if source_type:
                feeds_query = feeds_query.filter(Feed.source_type == source_type)
            
            matched_feeds = feeds_query.all()
            
            # Extract sources from matched feeds
            for feed in matched_feeds:
                source_key = None
                source_obj = None
                
                if feed.source_type == "blog" and feed.blog:
                    source_key = f"blog_{feed.blog.website}"
                    source_obj = db.query(Source).filter(
                        Source.source_type == "blog",
                        Source.website == feed.blog.website
                    ).first()
                    
                elif feed.source_type == "youtube":
                    meta = get_feed_metadata(feed, db)
                    channel_name = meta.get("channel_name")
                    
                    if channel_name:
                        source_key = f"youtube_{channel_name}"
                        source_obj = db.query(Source).filter(
                            Source.source_type == "youtube",
                            Source.name == channel_name
                        ).first()
                
                # Add source to dictionary
                if source_key and source_key not in sources_dict:
                    if source_obj:
                        sources_dict[source_key] = {
                            "source": source_obj,
                            "feeds": []
                        }
                    else:
                        # Create temporary source object
                        if feed.source_type == "blog" and feed.blog:
                            sources_dict[source_key] = {
                                "source": type('obj', (object,), {
                                    'id': None,
                                    'name': feed.blog.website,
                                    'website': feed.blog.website,
                                    'source_type': 'blog',
                                    'description': '',
                                    'follower_count': 0,
                                    'created_at': None,
                                    'updated_at': None
                                })(),
                                "feeds": []
                            }
                        elif feed.source_type == "youtube":
                            meta = get_feed_metadata(feed, db)
                            channel_name = meta.get("channel_name", "Unknown Channel")
                            sources_dict[source_key] = {
                                "source": type('obj', (object,), {
                                    'id': None,
                                    'name': channel_name,
                                    'website': meta.get("source_url", ""),
                                    'source_type': 'youtube',
                                    'description': '',
                                    'follower_count': 0,
                                    'created_at': None,
                                    'updated_at': None
                                })(),
                                "feeds": []
                            }
                
                # Add feed to source's feed list
                if source_key and source_key in sources_dict:
                    sources_dict[source_key]["feeds"].append(feed)
    
    # Step 3: Build response items from sources_dict
    items_list = []
    
    for source_key, source_data in sources_dict.items():
        source = source_data["source"]
        feeds = source_data["feeds"]
        
        # Calculate feed count
        feed_count = len(feeds)
        
        # Skip if no feeds (shouldn't happen based on logic above)
        if feed_count == 0:
            continue
        
        # Extract top topics from feeds
        topic_counts = {}
        for feed in feeds:
            if feed.categories:
                for cat in feed.categories:
                    topic_counts[cat] = topic_counts.get(cat, 0) + 1
        
        top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Get source metadata
        metadata = {}
        if source.id:  # Only get metadata for real sources
            metadata = get_source_metadata(source, db)
        
        source_description = getattr(source, 'description', '')
        if not source_description and metadata.get("description"):
            source_description = metadata["description"]
        
        items_list.append({
            "id": source.id,
            "name": source.name,
            "website": source.website,
            "source_type": source.source_type,
            "description": source_description,
            "feed_count": feed_count,
            "follower_count": source.follower_count if hasattr(source, 'follower_count') else 0,
            "top_topics": [topic for topic, count in top_topics],
            "is_following": source.id in followed_source_ids if source.id else False,
            "created_at": source.created_at.isoformat() if source.created_at else None,
            "updated_at": source.updated_at.isoformat() if source.updated_at else None,
            "metadata": metadata
        })
    
    # Apply topic filter if present
    if topic and topic.strip():
        items_list = [item for item in items_list if topic.strip() in item["top_topics"]]
    
    # Step 4: Sort by feed_count (descending)
    items_list.sort(key=lambda x: x["feed_count"], reverse=True)
    
    # Step 5: Paginate results
    total = len(items_list)
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated_items = items_list[start_idx:end_idx]
    
    return {
        "tab": "sources",
        "query": query,
        "items": paginated_items,
        "page": page,
        "limit": limit,
        "total": total,
        "has_more": (page * limit) < total,
        "user_selected_categories": user_category_ids
    }


# ------------------ Tab 5: Concepts Search ------------------

@router.get("/concepts", response_model=Dict[str, Any])
def search_concepts(
    query: Optional[str] = Query(None, description="Search query"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Results per page"),
    domain: Optional[str] = Query(None, description="Filter by domain"),
    topic: Optional[str] = Query(None, description="Filter by topic"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Search concepts extracted from content via LLM.
    
    This is for Tab 5: Concepts
    """
    user_id = current_user.id
    
    # Step 1: Get user's onboarding data to find selected categories
    user_onboarding = db.query(UserOnboarding).filter(
        UserOnboarding.user_id == user_id
    ).first()
    
    if not user_onboarding:
        raise HTTPException(
            status_code=404,
            detail="User onboarding data not found. Please complete onboarding first."
        )
    
    # Step 1: Get user's onboarding data
    user_onboarding = db.query(UserOnboarding).filter(UserOnboarding.user_id == user_id).first()
    if not user_onboarding:
        raise HTTPException(status_code=404, detail="User onboarding data not found.")
    user_category_ids = user_onboarding.domains_of_interest or []

    if not user_category_ids and not query:
        return {"tab": "concepts", "query": query, "items": [], "page": page, "limit": limit, "total": 0, "has_more": False}

    concepts_dict = {}

    # === CASE 1: Empty Query - Show concepts from user categories ===
    if not query or not query.strip():
        # Get common concepts from user categories
        relevant_concepts = db.query(Concept).join(FeedConcept).join(Feed).filter(
            Concept.is_active == True,
            Feed.category_id.in_(user_category_ids),
            Feed.status == "ready"
        ).distinct().all()
        
        for concept in relevant_concepts:
            concepts_dict[concept.id] = {
                "concept": concept,
                "priority": 100,
                "in_user_categories": True
            }

    # === CASE 2: Query has VALUE - Deep Relational Search ===
    else:
        query_clean = query.strip().lower()
        search_term = f"%{query_clean}%"
        
        # A. Direct Concept matches (Name, Description, Domain)
        direct_matches = db.query(Concept).join(DomainConcept, isouter=True).join(Domain, isouter=True).filter(
            Concept.is_active == True,
            or_(
                Concept.name.ilike(search_term),
                Concept.description.ilike(search_term),
                Domain.name.ilike(search_term)
            )
        ).distinct().all()
        
        for concept in direct_matches:
            score = 150
            if concept.name.lower().startswith(query_clean):
                score = 300 
            elif query_clean in concept.name.lower():
                score = 250
            concepts_dict[concept.id] = {"concept": concept, "priority": score, "match_type": "direct"}
            
        # B. Relational Matching via Content (Feeds, Categories, SubCategories)
        # We find concepts related to anything matching the query
        relational_results = db.query(Concept).join(FeedConcept).join(Feed).join(Category, isouter=True).join(SubCategory, isouter=True).filter(
            Concept.is_active == True,
            or_(
                Feed.title.ilike(search_term),
                Feed.categories.contains([query_clean]),
                Category.name.ilike(search_term),
                SubCategory.name.ilike(search_term)
            )
        ).distinct().all()
        
        for concept in relational_results:
            if concept.id not in concepts_dict:
                concepts_dict[concept.id] = {
                    "concept": concept,
                    "priority": 100, # Base relational score
                    "match_type": "relational"
                }
            else:
                # If already exists, boost it for having content relevance too
                concepts_dict[concept.id]["priority"] += 50

        # Boost concepts based on user categories and global feed volume
        for concept_id, data in concepts_dict.items():
            # Bonus 1: User Onboarding Match
            has_user_cat_feeds = db.query(FeedConcept).join(Feed).filter(
                FeedConcept.concept_id == concept_id,
                Feed.category_id.in_(user_category_ids),
                Feed.status == "ready"
            ).first() is not None
            
            if has_user_cat_feeds:
                data["priority"] += 50
                data["in_user_categories"] = True
            else:
                data["in_user_categories"] = False

    # Step 3: Build response items & Calculate feed counts
    items_list = []
    for concept_id, data in concepts_dict.items():
        concept = data["concept"]
        
        # Base filters for feeds
        feed_filters = [
            FeedConcept.concept_id == concept.id,
            Feed.status == "ready"
        ]
        
        # Apply topic filter if provided
        if topic and topic.strip():
            feed_filters.append(Feed.categories.contains([topic.strip()]))
            
        # 1. Total feed count (Matching search criteria + global context)
        total_feed_count = db.query(func.count(Feed.id)).join(FeedConcept).filter(*feed_filters).scalar() or 0
        
        # 2. User relevant feeds (Within user's selected categories)
        user_relevant_filters = feed_filters + [Feed.category_id.in_(user_category_ids)]
        relevant_feeds_query = db.query(Feed).join(FeedConcept).filter(*user_relevant_filters).order_by(Feed.created_at.desc())
        
        user_feeds = relevant_feeds_query.limit(3).all()
        user_feed_count = relevant_feeds_query.count()

        # Skip if no feeds at all matching filters (e.g. if topic filter active and no matches)
        if total_feed_count == 0:
            continue

        # Format feeds for JSON serialization (Avoids Internal Server Error)
        formatted_feeds = []
        for f in user_feeds:
            # Safely get summary
            summary = ""
            if f.ai_generated_content and isinstance(f.ai_generated_content, dict):
                summary = f.ai_generated_content.get("summary", "")[:150] + "..."
            
            # Safely get thumbnail
            thumbnail = None
            if f.source_type == "blog" and f.blog:
                thumbnail = f"https://www.google.com/s2/favicons?domain={f.blog.website}&sz=64"
            
            formatted_feeds.append({
                "id": f.id,
                "title": f.title,
                "summary": summary,
                "thumbnail": thumbnail,
                "source_type": f.source_type,
                "created_at": f.created_at.isoformat() if f.created_at else None
            })

        # Get domains
        concept_domains = [{"id": d.id, "name": d.name} for d in concept.domains]
        domain_ids = [d.id for d in concept.domains]
        
        # Related Concepts Logic
        related_concepts_list = concept.related_concepts[:5] if concept.related_concepts else []
        
        # DYNAMIC FALLBACK: If no related concepts in DB, find concepts in same domains
        if not related_concepts_list and domain_ids:
            shared_domain_concepts = db.query(Concept.name).join(DomainConcept).filter(
                DomainConcept.domain_id.in_(domain_ids),
                Concept.id != concept.id,
                Concept.is_active == True
            ).limit(5).all()
            related_concepts_list = [c[0] for c in shared_domain_concepts]
        
        items_list.append({
            "id": concept.id,
            "name": concept.name,
            "description": concept.description,
            "feed_count": total_feed_count,
            "priority": data["priority"],
            "in_user_categories": data["in_user_categories"],
            "popularity_score": concept.popularity_score,
            "related_concepts": related_concepts_list,
            "domains": concept_domains,
            "created_at": concept.created_at.isoformat() if concept.created_at else None
        })

    # Step 4: Sort by Priority FIRST, then by feed_count
    items_list.sort(key=lambda x: (-x["priority"], -x["feed_count"]))

    # Step 5: Paginate
    total = len(items_list)
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated_items = items_list[start_idx:end_idx]

    return {
        "tab": "concepts",
        "query": query,
        "items": paginated_items,
        "page": page,
        "limit": limit,
        "total": total,
        "has_more": (page * limit) < total,
        "user_selected_categories": user_category_ids
    }


# ------------------ Unified Search Endpoint ------------------

@router.get("/unified", response_model=Dict[str, Any])
def unified_search(
    query: str = Query(..., min_length=2, description="Search query"),
    tabs: str = Query("all", description="Tabs to search: all or comma-separated list"),
    page: int = Query(1, ge=1, description="Page number"),
    limit_per_tab: int = Query(5, ge=1, le=20, description="Results per tab"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Unified search across all 5 tabs.
    
    Returns results from multiple tabs in one response.
    """
    user_id = current_user.id
    
    # Step 1: Get user's onboarding data to find selected categories
    user_onboarding = db.query(UserOnboarding).filter(
        UserOnboarding.user_id == user_id
    ).first()
    
    if not user_onboarding:
        raise HTTPException(
            status_code=404,
            detail="User onboarding data not found. Please complete onboarding first."
        )
    
    # Get user's domains of interest (category IDs)
    user_category_ids = user_onboarding.domains_of_interest
    
    if not user_category_ids or len(user_category_ids) == 0:
        return {
            "query": query,
            "tabs": tabs,
            "results": {},
            "message": "No categories selected in onboarding. Please select categories first."
        }

    if not query or len(query.strip()) < 2:
        return {
            "query": query,
            "tabs": tabs,
            "results": {},
            "message": "Query too short"
        }
    
    # Parse tabs parameter
    if tabs.lower() == "all":
        tabs_list = ["content", "lists", "topics", "sources", "concepts"]
    else:
        tabs_list = [tab.strip().lower() for tab in tabs.split(",")]
    
    results = {}
    
    # Search each requested tab
    if "content" in tabs_list:
        content_results = search_content(query, 1, limit_per_tab, current_user=current_user, db=db)
        results["content"] = {
            "items": content_results["items"][:limit_per_tab],
            "total": content_results["total"],
            "has_more": content_results["has_more"]
        }
    
    if "lists" in tabs_list:
        lists_results = search_lists(query, 1, limit_per_tab, current_user=current_user, db=db)
        results["lists"] = {
            "items": lists_results["items"][:limit_per_tab],
            "total": lists_results["total"],
            "has_more": lists_results["has_more"]
        }
    
    if "topics" in tabs_list:
        topics_results = search_topics(query, 1, limit_per_tab, current_user=current_user, db=db)
        results["topics"] = {
            "items": topics_results["items"][:limit_per_tab],
            "total": topics_results["total"],
            "has_more": topics_results["has_more"]
        }
    
    if "sources" in tabs_list:
        sources_results = search_sources(query, 1, limit_per_tab, current_user=current_user, db=db)
        results["sources"] = {
            "items": sources_results["items"][:limit_per_tab],
            "total": sources_results["total"],
            "has_more": sources_results["has_more"]
        }
    
    if "concepts" in tabs_list:
        concepts_results = search_concepts(query, 1, limit_per_tab, current_user=current_user, db=db)
        results["concepts"] = {
            "items": concepts_results["items"][:limit_per_tab],
            "total": concepts_results["total"],
            "has_more": concepts_results["has_more"]
        }
    
    return {
        "query": query,
        "tabs": tabs_list,
        "results": results,
        "page": page,
        "limit_per_tab": limit_per_tab
    }


@router.get("/concept/{concept_id}", response_model=Dict[str, Any])
def get_concept_by_id(
    concept_id: int,
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Results per page"),
    include_slides: bool = Query(True, description="Include full slide content"),
    user_id: Optional[int] = Query(None, description="User ID for personalization"),
    db: Session = Depends(get_db)
):
    """
    Get concept by ID with all associated feeds including slides and full metadata.
    """
    # Get the concept
    concept = db.query(Concept).filter(
        Concept.id == concept_id,
        Concept.is_active == True
    ).first()
    
    if not concept:
        raise HTTPException(status_code=404, detail="Concept not found")
    
    # Get all feeds associated with this concept
    feed_ids_query = db.query(FeedConcept.feed_id).filter(
        FeedConcept.concept_id == concept_id
    )
    
    # Get feeds with all relationships including slides
    query_obj = db.query(Feed).options(
        joinedload(Feed.blog),
        joinedload(Feed.slides),
        joinedload(Feed.category),
        joinedload(Feed.subcategory),
        joinedload(Feed.concepts),
        joinedload(Feed.published_feed)
    ).filter(
        Feed.id.in_(feed_ids_query),
        Feed.status == "ready"
    )
    
    # Count and paginate
    total = query_obj.count()
    query_obj = query_obj.order_by(Feed.created_at.desc())
    feeds = query_obj.offset((page - 1) * limit).limit(limit).all()
    
    # Format feeds with full content including slides
    formatted_feeds = []
    for feed in feeds:
        # Get metadata
        meta = get_feed_metadata(feed, db)
        
        # Check if bookmarked
        is_bookmarked = False
        if user_id:
            bookmark = db.query(Bookmark).filter(
                Bookmark.user_id == user_id,
                Bookmark.feed_id == feed.id
            ).first()
            is_bookmarked = bookmark is not None
        
        # Get other concepts
        feed_concepts = []
        if hasattr(feed, 'concepts') and feed.concepts:
            for c in feed.concepts:
                if c.id != concept_id:  # Exclude the current concept
                    feed_concepts.append({"id": c.id, "name": c.name})
        
        # Get source info
        source_info = {}
        if feed.source_type == "youtube":
            source_info = {
                "type": "youtube",
                "name": meta.get("channel_name", "YouTube"),
                "url": meta.get("source_url", "#"),
                "channel_id": meta.get("channel_id"),
                "channel_thumbnail": meta.get("channel_thumbnail")
            }
        elif feed.source_type == "blog" and feed.blog:
            source_info = {
                "type": "blog",
                "name": feed.blog.website,
                "url": feed.blog.website,
                "author": getattr(feed.blog, 'author', 'Unknown'),
                "favicon": meta.get("favicon")
            }
        
        # Check if published
        is_published = feed.published_feed is not None
        
        # Get category and subcategory
        category_name = feed.category.name if feed.category else None
        subcategory_name = feed.subcategory.name if feed.subcategory else None
        
        # Extract summary from AI content
        summary = ""
        if feed.ai_generated_content and "summary" in feed.ai_generated_content:
            summary = feed.ai_generated_content["summary"]
        
        # Get key points
        key_points = []
        if feed.ai_generated_content and "key_points" in feed.ai_generated_content:
            key_points = feed.ai_generated_content["key_points"]
        
        # Get conclusion
        conclusion = ""
        if feed.ai_generated_content and "conclusion" in feed.ai_generated_content:
            conclusion = feed.ai_generated_content["conclusion"]
        
        # Prepare slides data
        slides_data = []
        if include_slides and feed.slides:
            sorted_slides = sorted(feed.slides, key=lambda x: x.order)
            for slide in sorted_slides:
                slide_data = {
                    "id": slide.id,
                    "order": slide.order,
                    "title": slide.title,
                    "body": slide.body,
                    "bullets": slide.bullets or [],
                    "background_color": slide.background_color or "#FFFFFF",
                    "source_refs": slide.source_refs or [],
                    "render_markdown": bool(slide.render_markdown),
                    "created_at": slide.created_at.isoformat() if slide.created_at else None,
                    "updated_at": slide.updated_at.isoformat() if slide.updated_at else None
                }
                slides_data.append(slide_data)
        
        # Format the complete feed
        feed_data = {
            "id": feed.id,
            "title": feed.title,
            "summary": summary,
            "key_points": key_points,
            "conclusion": conclusion,
            "content_type": feed.content_type.value if feed.content_type else "Video",
            "source_type": feed.source_type,
            "source_info": source_info,
            "categories": feed.categories or [],
            "concepts": feed_concepts,  # Other concepts (excluding current)
            "skills": getattr(feed, 'skills', []) or [],
            "tools": getattr(feed, 'tools', []) or [],
            "roles": getattr(feed, 'roles', []) or [],
            "status": feed.status,
            "is_published": is_published,
            "is_bookmarked": is_bookmarked,
            "ai_generated": bool(feed.ai_generated_content),
            "meta": meta,
            "category": {
                "id": feed.category_id,
                "name": category_name
            } if feed.category_id else None,
            "subcategory": {
                "id": feed.subcategory_id,
                "name": subcategory_name
            } if feed.subcategory_id else None,
            "slides": slides_data if include_slides else [],
            "slides_count": len(feed.slides) if feed.slides else 0,
            "created_at": feed.created_at.isoformat() if feed.created_at else None,
            "updated_at": feed.updated_at.isoformat() if feed.updated_at else None,
            "published_at": feed.published_feed.published_at.isoformat() if is_published and feed.published_feed else None
        }
        
        # Add source-specific IDs
        if feed.source_type == "blog":
            feed_data["blog_id"] = feed.blog_id
            if feed.blog:
                feed_data["website"] = feed.blog.website
                feed_data["blog_url"] = feed.blog.url
        elif feed.source_type == "youtube":
            feed_data["transcript_id"] = feed.transcript_id
            # Try to get video_id
            if feed.transcript_id:
                transcript = db.query(Transcript).filter(
                    Transcript.transcript_id == feed.transcript_id
                ).first()
                if transcript:
                    feed_data["video_id"] = transcript.video_id
                    feed_data["youtube_url"] = f"https://www.youtube.com/watch?v={transcript.video_id}"
        
        formatted_feeds.append(feed_data)
    
    # Get related concepts
    related_concepts = []
    if concept.related_concepts:
        related_concept_names = concept.related_concepts[:5]
        related_concept_objs = db.query(Concept).filter(
            Concept.name.in_(related_concept_names)
        ).all()
        related_concepts = [
            {
                "id": c.id,
                "name": c.name,
                "description": c.description,
                "feed_count": len(c.feeds) if hasattr(c, 'feeds') else 0,
                "popularity_score": c.popularity_score
            }
            for c in related_concept_objs
        ]
    
    # Get domains for this concept
    domains = []
    domain_concepts = db.query(DomainConcept).filter(
        DomainConcept.concept_id == concept_id
    ).all()
    
    for dc in domain_concepts:
        domain = db.query(Domain).filter(Domain.id == dc.domain_id).first()
        if domain:
            domains.append({
                "id": domain.id,
                "name": domain.name,
                "description": domain.description,
                "relevance_score": dc.relevance_score
            })
    
    # Get concept statistics from all feeds (not just paginated ones)
    all_feeds_for_concept = db.query(Feed).filter(
        Feed.id.in_(feed_ids_query),
        Feed.status == "ready"
    ).all()
    
    feed_count = len(all_feeds_for_concept)
    unique_topics = set()
    unique_sources = set()
    total_slides = 0
    content_types = {}
    
    for feed in all_feeds_for_concept:
        # Collect unique topics from categories
        if feed.categories:
            for category in feed.categories:
                unique_topics.add(category)
        
        # Collect unique sources
        if feed.source_type == "youtube":
            meta = get_feed_metadata(feed, db)
            channel_name = meta.get("channel_name")
            if channel_name:
                unique_sources.add(f"YouTube: {channel_name}")
        elif feed.source_type == "blog" and feed.blog:
            unique_sources.add(f"Blog: {feed.blog.website}")
        
        # Count slides
        if feed.slides:
            total_slides += len(feed.slides)
        
        # Count content types
        content_type = feed.content_type.value if feed.content_type else "Unknown"
        content_types[content_type] = content_types.get(content_type, 0) + 1
    
    # Get most common topic
    topic_counts = {}
    for feed in all_feeds_for_concept:
        if feed.categories:
            for category in feed.categories:
                topic_counts[category] = topic_counts.get(category, 0) + 1
    
    most_common_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return {
        "concept": {
            "id": concept.id,
            "name": concept.name,
            "description": concept.description,
            "created_at": concept.created_at.isoformat() if concept.created_at else None,
            "updated_at": concept.updated_at.isoformat() if concept.updated_at else None,
            "popularity_score": concept.popularity_score,
            "is_active": concept.is_active
        },
        "statistics": {
            "feed_count": feed_count,
            "unique_topics": len(unique_topics),
            "unique_sources": len(unique_sources),
            "total_slides": total_slides,
            "average_slides_per_feed": round(total_slides / feed_count, 2) if feed_count > 0 else 0,
            "content_type_distribution": [
                {"type": ct, "count": count} for ct, count in content_types.items()
            ],
            "top_topics": [{"topic": topic, "count": count} for topic, count in most_common_topics],
            "top_sources": list(unique_sources)[:10]
        },
        "domains": domains,
        "related_concepts": related_concepts,
        "feeds": {
            "items": formatted_feeds,
            "page": page,
            "limit": limit,
            "total": total,
            "has_more": (page * limit) < total,
            "include_slides": include_slides
        }
    }

@router.get("/user/follows", response_model=Dict[str, Any])
def get_user_follows(
    topic_id: Optional[int] = Query(None, description="Check if user follows this specific topic"),
    source_id: Optional[int] = Query(None, description="Check if user follows this specific source"),
    include_counts: bool = Query(True, description="Include follower and feed counts"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get user's follow data for topics and sources.
    
    Without parameters: Returns all topics and sources the user follows
    With topic_id or source_id: Returns specific follow status
    
    Uses token authentication via get_current_user dependency
    """
    try:
        user_id = current_user.id
        
        response_data = {
            "user_id": user_id,
            "username": current_user.full_name,
            "email": current_user.email,
            "total_following": 0
        }
        
        # CASE 1: Check specific topic follow status
        if topic_id is not None:
            topic = db.query(Topic).filter(Topic.id == topic_id, Topic.is_active == True).first()
            if not topic:
                raise HTTPException(status_code=404, detail=f"Topic with ID {topic_id} not found")
            
            # Check if user follows this topic
            follow_record = db.query(UserTopicFollow).filter(
                UserTopicFollow.user_id == user_id,
                UserTopicFollow.topic_id == topic_id
            ).first()
            
            is_following = follow_record is not None
            
            # Get topic details
            topic_details = {
                "id": topic.id,
                "name": topic.name,
                "description": topic.description,
                "is_following": is_following,
                "followed_since": follow_record.created_at.isoformat() if follow_record else None
            }
            
            if include_counts:
                # Get feed count for this topic
                feed_count = db.query(Feed).filter(
                    Feed.status == "ready",
                    Feed.categories.contains([topic.name])
                ).count()
                
                topic_details.update({
                    "feed_count": feed_count,
                    "follower_count": topic.follower_count
                })
            
            return {
                **response_data,
                "specific_follow": {
                    "type": "topic",
                    "data": topic_details
                }
            }
        
        # CASE 2: Check specific source follow status
        if source_id is not None:
            source = db.query(Source).filter(Source.id == source_id, Source.is_active == True).first()
            if not source:
                raise HTTPException(status_code=404, detail=f"Source with ID {source_id} not found")
            
            # Check if user follows this source
            follow_record = db.query(UserSourceFollow).filter(
                UserSourceFollow.user_id == user_id,
                UserSourceFollow.source_id == source_id
            ).first()
            
            is_following = follow_record is not None
            
            # Get source details
            source_details = {
                "id": source.id,
                "name": source.name,
                "website": source.website,
                "source_type": source.source_type,
                "is_following": is_following,
                "followed_since": follow_record.created_at.isoformat() if follow_record else None
            }
            
            if include_counts:
                # Get feed count for this source
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
                
                source_details.update({
                    "feed_count": feed_count,
                    "follower_count": source.follower_count
                })
            
            return {
                **response_data,
                "specific_follow": {
                    "type": "source",
                    "data": source_details
                }
            }
        
        # CASE 3: No specific ID - Return all follows
        
        # Get all topics user follows
        topic_follows = db.query(UserTopicFollow).filter(
            UserTopicFollow.user_id == user_id
        ).all()
        
        followed_topics = []
        for follow in topic_follows:
            topic = db.query(Topic).filter(Topic.id == follow.topic_id).first()
            if topic and topic.is_active:
                topic_data = {
                    "id": topic.id,
                    "name": topic.name,
                    "description": topic.description,
                    "followed_since": follow.created_at.isoformat() if follow.created_at else None,
                    "follower_count": topic.follower_count
                }
                
                if include_counts:
                    # Get feed count for this topic
                    feed_count = db.query(Feed).filter(
                        Feed.status == "ready",
                        Feed.categories.contains([topic.name])
                    ).count()
                    topic_data["feed_count"] = feed_count
                
                followed_topics.append(topic_data)
        
        # Get all sources user follows
        source_follows = db.query(UserSourceFollow).filter(
            UserSourceFollow.user_id == user_id
        ).all()
        
        followed_sources = []
        for follow in source_follows:
            source = db.query(Source).filter(Source.id == follow.source_id).first()
            if source and source.is_active:
                source_data = {
                    "id": source.id,
                    "name": source.name,
                    "website": source.website,
                    "source_type": source.source_type,
                    "followed_since": follow.created_at.isoformat() if follow.created_at else None,
                    "follower_count": source.follower_count
                }
                
                if include_counts:
                    # Get feed count for this source
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
                    
                    source_data["feed_count"] = feed_count
                
                followed_sources.append(source_data)
        
        # Sort by followed date (most recent first)
        followed_topics.sort(key=lambda x: x["followed_since"] or "", reverse=True)
        followed_sources.sort(key=lambda x: x["followed_since"] or "", reverse=True)
        
        response_data.update({
            "topics": {
                "count": len(followed_topics),
                "items": followed_topics
            },
            "sources": {
                "count": len(followed_sources),
                "items": followed_sources
            },
            "total_following": len(followed_topics) + len(followed_sources)
        })
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching user follows: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch follow data: {str(e)}")


@router.get("/lists/{list_id}/feeds", response_model=Dict[str, Any])
def get_list_feeds_with_slides(
    list_id: int,
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Results per page"),
    include_slides: bool = Query(True, description="Include full slide content"),
    sort_by: str = Query("created_at", description="Sort by: created_at, title, or slides_count"),
    sort_order: str = Query("desc", description="Sort order: asc or desc"),
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all feeds in a list with full content including slides.
    
    Returns complete feed data with all slides for each feed in the list.
    """
    # Get the content list
    content_list = db.query(ContentList).filter(
        ContentList.id == list_id,
        ContentList.is_active == True
    ).first()
    
    if not content_list:
        raise HTTPException(status_code=404, detail=f"List with ID {list_id} not found")
    
    # Get user_id for personalization
    user_id = current_user.id if current_user else None
    
    # Get all feed IDs in this list
    feed_ids = content_list.feed_ids or []
    
    if not feed_ids:
        return {
            "list": {
                "id": content_list.id,
                "name": content_list.name,
                "description": content_list.description,
                "source_type": content_list.source_type,
                "list_type": "playlist" if content_list.source_type == "youtube" else "collection",
                "feed_count": 0
            },
            "feeds": {
                "items": [],
                "page": page,
                "limit": limit,
                "total": 0,
                "has_more": False
            }
        }
    
    # Query for feeds in this list with all relationships
    query = db.query(Feed).options(
        joinedload(Feed.blog),
        joinedload(Feed.slides),
        joinedload(Feed.category),
        joinedload(Feed.subcategory),
        joinedload(Feed.concepts),
        joinedload(Feed.published_feed)
    ).filter(
        Feed.id.in_(feed_ids),
        Feed.status == "ready"
    )
    
    # Apply sorting
    if sort_by == "title":
        if sort_order.lower() == "asc":
            query = query.order_by(Feed.title.asc())
        else:
            query = query.order_by(Feed.title.desc())
    elif sort_by == "slides_count":
        # Sort by number of slides
        query = query.order_by(func.array_length(Feed.slide_ids, 1).desc() if sort_order.lower() == "desc" 
                             else func.array_length(Feed.slide_ids, 1).asc())
    else:  # Default: sort by created_at
        if sort_order.lower() == "asc":
            query = query.order_by(Feed.created_at.asc())
        else:
            query = query.order_by(Feed.created_at.desc())
    
    # Count and paginate
    total = query.count()
    feeds = query.offset((page - 1) * limit).limit(limit).all()
    
    # Format feeds with full content
    formatted_feeds = []
    for feed in feeds:
        # Get metadata
        meta = get_feed_metadata(feed, db)
        
        # Check if bookmarked
        is_bookmarked = False
        if user_id:
            bookmark = db.query(Bookmark).filter(
                Bookmark.user_id == user_id,
                Bookmark.feed_id == feed.id
            ).first()
            is_bookmarked = bookmark is not None
        
        # Get concepts
        concepts = []
        if hasattr(feed, 'concepts') and feed.concepts:
            concepts = [{"id": c.id, "name": c.name} for c in feed.concepts][:5]
        
        # Get source info
        source_info = {}
        if feed.source_type == "youtube":
            source_info = {
                "type": "youtube",
                "name": meta.get("channel_name", "YouTube"),
                "url": meta.get("source_url", "#"),
                "channel_id": meta.get("channel_id"),
                "channel_thumbnail": meta.get("channel_thumbnail")
            }
        elif feed.source_type == "blog" and feed.blog:
            source_info = {
                "type": "blog",
                "name": feed.blog.website,
                "url": feed.blog.website,
                "author": getattr(feed.blog, 'author', 'Unknown'),
                "favicon": meta.get("favicon")
            }
        
        # Check if published
        is_published = feed.published_feed is not None
        
        # Get category and subcategory
        category_name = feed.category.name if feed.category else None
        subcategory_name = feed.subcategory.name if feed.subcategory else None
        
        # Extract summary from AI content
        summary = ""
        if feed.ai_generated_content and "summary" in feed.ai_generated_content:
            summary = feed.ai_generated_content["summary"]
        
        # Get key points
        key_points = []
        if feed.ai_generated_content and "key_points" in feed.ai_generated_content:
            key_points = feed.ai_generated_content["key_points"]
        
        # Get conclusion
        conclusion = ""
        if feed.ai_generated_content and "conclusion" in feed.ai_generated_content:
            conclusion = feed.ai_generated_content["conclusion"]
        
        # Prepare slides data
        slides_data = []
        if include_slides and feed.slides:
            sorted_slides = sorted(feed.slides, key=lambda x: x.order)
            for slide in sorted_slides:
                slide_data = {
                    "id": slide.id,
                    "order": slide.order,
                    "title": slide.title,
                    "body": slide.body,
                    "bullets": slide.bullets or [],
                    "background_color": slide.background_color or "#FFFFFF",
                    # "background_image_prompt": slide.background_image_prompt,
                    "source_refs": slide.source_refs or [],
                    "render_markdown": bool(slide.render_markdown),
                    "created_at": slide.created_at.isoformat() if slide.created_at else None,
                    "updated_at": slide.updated_at.isoformat() if slide.updated_at else None
                }
                slides_data.append(slide_data)
        
        # Format the complete feed
        feed_data = {
            "id": feed.id,
            "title": feed.title,
            "summary": summary,
            "key_points": key_points,
            "conclusion": conclusion,
            "content_type": feed.content_type.value if feed.content_type else "Video",
            "source_type": feed.source_type,
            "source_info": source_info,
            "categories": feed.categories or [],
            "concepts": concepts,
            "skills": getattr(feed, 'skills', []) or [],
            "tools": getattr(feed, 'tools', []) or [],
            "roles": getattr(feed, 'roles', []) or [],
            "status": feed.status,
            "is_published": is_published,
            "is_bookmarked": is_bookmarked,
            "ai_generated": bool(feed.ai_generated_content),
            "meta": meta,
            "category": {
                "id": feed.category_id,
                "name": category_name
            } if feed.category_id else None,
            "subcategory": {
                "id": feed.subcategory_id,
                "name": subcategory_name
            } if feed.subcategory_id else None,
            "slides": slides_data if include_slides else [],
            "slides_count": len(feed.slides) if feed.slides else 0,
            "created_at": feed.created_at.isoformat() if feed.created_at else None,
            "updated_at": feed.updated_at.isoformat() if feed.updated_at else None,
            "published_at": feed.published_feed.published_at.isoformat() if is_published and feed.published_feed else None
        }
        
        # Add source-specific IDs
        if feed.source_type == "blog":
            feed_data["blog_id"] = feed.blog_id
            if feed.blog:
                feed_data["website"] = feed.blog.website
                feed_data["blog_url"] = feed.blog.url
        elif feed.source_type == "youtube":
            feed_data["transcript_id"] = feed.transcript_id
            # Try to get video_id
            if feed.transcript_id:
                transcript = db.query(Transcript).filter(
                    Transcript.transcript_id == feed.transcript_id
                ).first()
                if transcript:
                    feed_data["video_id"] = transcript.video_id
                    feed_data["youtube_url"] = f"https://www.youtube.com/watch?v={transcript.video_id}"
        
        formatted_feeds.append(feed_data)
    
    # Format list details
    list_details = {
        "id": content_list.id,
        "name": content_list.name,
        "description": content_list.description,
        "source_type": content_list.source_type,
        "source_id": content_list.source_id,
        "list_type": "playlist" if content_list.source_type == "youtube" else "collection",
        "feed_count": len(feed_ids),
        "created_at": content_list.created_at.isoformat() if content_list.created_at else None,
        "updated_at": content_list.updated_at.isoformat() if content_list.updated_at else None
    }
    
    # Add YouTube info for YouTube playlists
    if content_list.source_type == "youtube" and content_list.source_id:
        if content_list.source_id.startswith("PL"):
            list_details["youtube_info"] = {
                "type": "playlist",
                "playlist_id": content_list.source_id,
                "url": f"https://www.youtube.com/playlist?list={content_list.source_id}"
            }
        elif content_list.source_id.startswith("channel_"):
            channel_id = content_list.source_id.replace("channel_", "")
            list_details["youtube_info"] = {
                "type": "channel",
                "channel_id": channel_id,
                "url": f"https://www.youtube.com/channel/{channel_id}"
            }
    
    return {
        "list": list_details,
        "feeds": {
            "items": formatted_feeds,
            "page": page,
            "limit": limit,
            "total": total,
            "has_more": (page * limit) < total,
            "sort_by": sort_by,
            "sort_order": sort_order,
            "include_slides": include_slides
        }
    }


# ------------------ User Subcategories Search API ------------------

@router.get("/user-subcategories", response_model=Dict[str, Any])
def search_user_subcategories(
    query: Optional[str] = Query(None, description="Search query to match subcategory name or description"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Results per page"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Search subcategories with smart prioritization.
    
    Logic:
    - NO QUERY: Returns subcategories from user's selected categories only
    - WITH QUERY: 
        1. First shows matching subcategories from user's selected categories
        2. Then shows matching subcategories from OTHER categories
        3. Searches in both name AND description fields
    
    Authentication Required: Yes (Bearer Token)
    """
    user_id = current_user.id
    
    # Step 1: Get user's onboarding data
    user_onboarding = db.query(UserOnboarding).filter(
        UserOnboarding.user_id == user_id
    ).first()
    
    if not user_onboarding:
        raise HTTPException(
            status_code=404,
            detail="User onboarding data not found. Please complete onboarding first."
        )
    
    user_category_ids = user_onboarding.domains_of_interest or []
    
    if not user_category_ids or len(user_category_ids) == 0:
        return {
            "tab": "user-subcategories",
            "query": query,
            "items": [],
            "page": page,
            "limit": limit,
            "total": 0,
            "has_more": False,
            "message": "No categories selected in onboarding. Please select categories first."
        }
    
    # Step 2: Build subcategory list based on query presence
    subcategory_list = []
    
    if not query or not query.strip():
        # NO QUERY: Show only user's selected categories' subcategories
        user_subcategories = db.query(SubCategory).filter(
            SubCategory.category_id.in_(user_category_ids),
            SubCategory.is_active == True
        ).all()
        
        for subcategory in user_subcategories:
            feed_count = db.query(func.count(Feed.id)).filter(
                Feed.subcategory_id == subcategory.id,
                Feed.status == "ready"
            ).scalar() or 0
            
            category = db.query(Category).filter(Category.id == subcategory.category_id).first()
            
            subcategory_list.append({
                "id": subcategory.id,
                "uuid": subcategory.uuid,
                "name": subcategory.name,
                "description": subcategory.description,
                "category_id": subcategory.category_id,
                "category_name": category.name if category else "Unknown",
                "feed_count": feed_count,
                "is_active": subcategory.is_active,
                "is_from_user_categories": True,
                "created_at": subcategory.created_at.isoformat() if subcategory.created_at else None,
                "updated_at": subcategory.updated_at.isoformat() if subcategory.updated_at else None
            })
    else:
        # WITH QUERY: Prioritize user's categories, then show others
        # Generate fuzzy search variations
        search_variations = build_fuzzy_search_terms(query)
        
        # Build OR conditions for all variations
        search_conditions = []
        for variation in search_variations:
            search_term = f"%{variation}%"
            search_conditions.append(SubCategory.name.ilike(search_term))
            search_conditions.append(SubCategory.description.ilike(search_term))
        
        # Part 1: Get matching subcategories from USER'S selected categories
        user_matches = db.query(SubCategory).filter(
            SubCategory.category_id.in_(user_category_ids),
            SubCategory.is_active == True,
            or_(*search_conditions)
        ).all()
        
        for subcategory in user_matches:
            feed_count = db.query(func.count(Feed.id)).filter(
                Feed.subcategory_id == subcategory.id,
                Feed.status == "ready"
            ).scalar() or 0
            
            category = db.query(Category).filter(Category.id == subcategory.category_id).first()
            
            subcategory_list.append({
                "id": subcategory.id,
                "uuid": subcategory.uuid,
                "name": subcategory.name,
                "description": subcategory.description,
                "category_id": subcategory.category_id,
                "category_name": category.name if category else "Unknown",
                "feed_count": feed_count,
                "is_active": subcategory.is_active,
                "is_from_user_categories": True,
                "created_at": subcategory.created_at.isoformat() if subcategory.created_at else None,
                "updated_at": subcategory.updated_at.isoformat() if subcategory.updated_at else None
            })
        
        # Part 2: Get matching subcategories from OTHER categories
        other_matches = db.query(SubCategory).filter(
            ~SubCategory.category_id.in_(user_category_ids),
            SubCategory.is_active == True,
            or_(*search_conditions)
        ).all()
        
        for subcategory in other_matches:
            feed_count = db.query(func.count(Feed.id)).filter(
                Feed.subcategory_id == subcategory.id,
                Feed.status == "ready"
            ).scalar() or 0
            
            category = db.query(Category).filter(Category.id == subcategory.category_id).first()
            
            subcategory_list.append({
                "id": subcategory.id,
                "uuid": subcategory.uuid,
                "name": subcategory.name,
                "description": subcategory.description,
                "category_id": subcategory.category_id,
                "category_name": category.name if category else "Unknown",
                "feed_count": feed_count,
                "is_active": subcategory.is_active,
                "is_from_user_categories": False,
                "created_at": subcategory.created_at.isoformat() if subcategory.created_at else None,
                "updated_at": subcategory.updated_at.isoformat() if subcategory.updated_at else None
            })
    
    # Step 3: Sort - User's categories first (by feed_count), then others (by feed_count)
    subcategory_list.sort(key=lambda x: (not x["is_from_user_categories"], -x["feed_count"]))
    
    # Step 4: Paginate
    total = len(subcategory_list)
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated_items = subcategory_list[start_idx:end_idx]
    
    return {
        "tab": "user-subcategories",
        "query": query,
        "items": paginated_items,
        "page": page,
        "limit": limit,
        "total": total,
        "has_more": (page * limit) < total,
        "user_selected_categories": user_category_ids
    }