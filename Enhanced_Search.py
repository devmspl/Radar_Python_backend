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
from models import (
    Feed, Blog, Transcript, Topic, Source, Concept, Domain, 
    ContentList, UserTopicFollow, UserSourceFollow, Bookmark,
    Category, SubCategory, FeedConcept, DomainConcept,Topic
)
from feed_router import get_feed_metadata
import os
import re
import requests




router = APIRouter(prefix="/search", tags=["Search"])
logger = logging.getLogger(__name__)

# ------------------ Helper Functions ------------------

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
        "is_bookmarked": is_bookmarked,
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
    query: str = Query(..., min_length=2, description="Search query"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Results per page"),
    content_type: Optional[str] = Query(None, description="Filter by content type"),
    source_type: Optional[str] = Query(None, description="Filter by source type"),
    domain: Optional[str] = Query(None, description="Filter by domain"),
    topic: Optional[str] = Query(None, description="Filter by topic"),
    concept: Optional[str] = Query(None, description="Filter by concept"),
    user_id: Optional[int] = Query(None, description="User ID for personalization"),
    db: Session = Depends(get_db)
):
    """
    Search content feeds (blogs and videos) with AI-generated summaries.
    
    This is for Tab 1: Content (Summary/Highlights)
    """
    # Base query
    query_obj = db.query(Feed).options(
        joinedload(Feed.blog),
        joinedload(Feed.slides),
        joinedload(Feed.category),
        joinedload(Feed.subcategory),
        joinedload(Feed.concepts)
    ).filter(Feed.status == "ready")
    
    # Text search
    if query:
        search_term = f"%{query.lower()}%"
        query_obj = query_obj.filter(
            or_(
                Feed.title.ilike(search_term),
                cast(Feed.categories, String).ilike(search_term),
                cast(Feed.ai_generated_content, String).ilike(search_term)
            )
        )
    
    # Filters
    if content_type:
        query_obj = query_obj.filter(Feed.content_type == content_type)
    
    if source_type:
        query_obj = query_obj.filter(Feed.source_type == source_type)
    
    if topic and topic.strip():
        query_obj = query_obj.filter(Feed.categories.contains([topic.strip()]))
    
    if concept and concept.strip():
        # Search in concepts
        concept_obj = db.query(Concept).filter(
            Concept.name.ilike(f"%{concept}%")
        ).first()
        if concept_obj:
            feed_ids = db.query(FeedConcept.feed_id).filter(
                FeedConcept.concept_id == concept_obj.id
            ).all()
            if feed_ids:
                feed_id_list = [fid[0] for fid in feed_ids]
                query_obj = query_obj.filter(Feed.id.in_(feed_id_list))
    
    if domain and domain.strip():
        # Search in domains
        domain_obj = db.query(Domain).filter(
            Domain.name.ilike(f"%{domain}%")
        ).first()
        if domain_obj:
            # Get concepts in this domain
            domain_concepts = db.query(DomainConcept).filter(
                DomainConcept.domain_id == domain_obj.id
            ).all()
            concept_ids = [dc.concept_id for dc in domain_concepts]
            
            # Get feeds with these concepts
            feed_ids = db.query(FeedConcept.feed_id).filter(
                FeedConcept.concept_id.in_(concept_ids)
            ).distinct().all()
            
            if feed_ids:
                feed_id_list = [fid[0] for fid in feed_ids]
                query_obj = query_obj.filter(Feed.id.in_(feed_id_list))
    
    # Count and paginate
    total = query_obj.count()
    query_obj = query_obj.order_by(Feed.created_at.desc())
    feeds = query_obj.offset((page - 1) * limit).limit(limit).all()
    
    # Format results
    items = []
    for feed in feeds:
        items.append(format_feed_for_search(feed, db, user_id))
    
    return {
        "tab": "content",
        "query": query,
        "items": items,
        "page": page,
        "limit": limit,
        "total": total,
        "has_more": (page * limit) < total,
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
    user_id: Optional[int] = Query(None, description="User ID"),
    db: Session = Depends(get_db)
):
    """
    Search content lists (playlists or curated collections).
    
    This is for Tab 2: Lists
    Auto-generated from YouTube playlists or created manually
    """
    # Try to auto-create lists from existing YouTube playlists if no lists exist
    all_lists = db.query(ContentList).count()
    if all_lists == 0:
        # Auto-create lists from YouTube playlists in transcripts
        logger.info("No lists found, auto-creating from YouTube playlists...")
        auto_create_lists_from_youtube(db)  # This will now work with the import
    
    # Base query
    query_obj = db.query(ContentList).filter(ContentList.is_active == True)
    
    if query:
        search_term = f"%{query.lower()}%"
        query_obj = query_obj.filter(
            or_(
                ContentList.name.ilike(search_term),
                ContentList.description.ilike(search_term),
                ContentList.source_id.ilike(search_term)
            )
        )
    
    if source_type:
        query_obj = query_obj.filter(ContentList.source_type == source_type)
    
    # Count and paginate
    total = query_obj.count()
    query_obj = query_obj.order_by(ContentList.created_at.desc())
    content_lists = query_obj.offset((page - 1) * limit).limit(limit).all()
    
    # Format results
    items = []
    for content_list in content_lists:
        # Get feed details
        feed_details = []
        all_feeds = []
        
        if content_list.feed_ids and len(content_list.feed_ids) > 0:
            # Get feeds with their relationships
            feeds = db.query(Feed).options(
                joinedload(Feed.blog),
                joinedload(Feed.slides),
                joinedload(Feed.category),
                joinedload(Feed.subcategory)
            ).filter(
                Feed.id.in_(content_list.feed_ids),
                Feed.status == "ready"
            ).all()
            
            all_feeds = feeds
            
            # Format sample feeds (first 3)
            for feed in feeds[:3]:
                meta = get_feed_metadata(feed, db)
                
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
                
                # Get source info
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
                
                feed_details.append({
                    "id": feed.id,
                    "title": feed.title,
                    "summary": summary,
                    "content_type": feed.content_type.value if feed.content_type else "Video",
                    "source_type": feed.source_type,
                    "source_info": source_info,
                    "meta": meta
                })
        
        # Extract topics from all feeds
        all_topics = set()
        for feed in all_feeds:
            if feed.categories:
                for category in feed.categories:
                    all_topics.add(category)
        
        # Get list type based on source
        list_type = "playlist" if content_list.source_type == "youtube" else "collection"
        if content_list.source_type == "blog":
            list_type = "blog_collection"
        
        # Get thumbnail from first feed
        thumbnail_url = None
        if all_feeds:
            meta = get_feed_metadata(all_feeds[0], db)
            thumbnail_url = meta.get("thumbnail_url")
        
        items.append({
            "id": content_list.id,
            "name": content_list.name,
            "description": content_list.description,
            "source_type": content_list.source_type,
            "source_id": content_list.source_id,
            "list_type": list_type,
            "feed_count": len(content_list.feed_ids) if content_list.feed_ids else 0,
            "sample_feeds": feed_details,
            "topics": list(all_topics)[:10],
            "thumbnail_url": thumbnail_url,
            "created_at": content_list.created_at.isoformat() if content_list.created_at else None,
            "updated_at": content_list.updated_at.isoformat() if content_list.updated_at else None
        })
    
    return {
        "tab": "lists",
        "query": query,
        "items": items,
        "page": page,
        "limit": limit,
        "total": total,
        "has_more": (page * limit) < total,
        "filters": {
            "source_type": source_type
        }
    }


# ------------------ Tab 3: Topics Search ------------------

@router.get("/topics", response_model=Dict[str, Any])
def search_topics(
    query: Optional[str] = Query(None, description="Search query"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Results per page"),
    domain: Optional[str] = Query(None, description="Filter by domain"),
    user_id: Optional[int] = Query(None, description="User ID"),
    db: Session = Depends(get_db)
):
    """
    Search topics extracted from content via LLM.
    
    This is for Tab 3: Topics
    Topics can be followed via topic_id
    """
    # Get all unique topics from feed categories
    all_feeds = db.query(Feed).filter(
        Feed.status == "ready",
        Feed.categories.isnot(None)
    ).all()
    
    # Extract unique topics with counts
    topics_dict = {}
    for feed in all_feeds:
        if feed.categories:
            for category_name in feed.categories:
                if category_name not in topics_dict:
                    # Count feeds for this topic
                    feed_count = db.query(Feed).filter(
                        Feed.categories.contains([category_name]),
                        Feed.status == "ready"
                    ).count()
                    
                    # Get or create Topic in database
                    topic = db.query(Topic).filter(
                        Topic.name == category_name
                    ).first()
                    
                    # If topic doesn't exist in Topic table, create it
                    if not topic:
                        topic = Topic(
                            name=category_name,
                            description=f"Content related to {category_name}",
                            follower_count=0,
                            is_active=True,
                            created_at=datetime.utcnow(),
                            updated_at=datetime.utcnow()
                        )
                        db.add(topic)
                        db.commit()
                        db.refresh(topic)
                    
                    # Get follower count from Topic table
                    follower_count = topic.follower_count
                    
                    # Get unique sources for this topic
                    topic_feeds = db.query(Feed).filter(
                        Feed.categories.contains([category_name]),
                        Feed.status == "ready"
                    ).all()
                    
                    unique_sources = set()
                    for topic_feed in topic_feeds:
                        if topic_feed.source_type == "youtube":
                            meta = get_feed_metadata(topic_feed, db)
                            channel_name = meta.get("channel_name")
                            if channel_name:
                                unique_sources.add(channel_name)
                        elif topic_feed.source_type == "blog" and topic_feed.blog:
                            unique_sources.add(topic_feed.blog.website)
                    
                    # Check if current user is following this topic
                    is_following = False
                    if user_id and topic:
                        user_follow = db.query(UserTopicFollow).filter(
                            UserTopicFollow.user_id == user_id,
                            UserTopicFollow.topic_id == topic.id
                        ).first()
                        is_following = user_follow is not None
                    
                    topics_dict[category_name] = {
                        "id": topic.id,
                        "name": category_name,
                        "description": topic.description,
                        "feed_count": feed_count,
                        "follower_count": follower_count,
                        "source_count": len(unique_sources),
                        "is_following": is_following,
                        "popularity": feed_count + (follower_count * 2),  # Weight followers more
                        "sources": list(unique_sources)[:5]  # Top 5 sources
                    }
    
    # Convert to list and filter
    topics_list = list(topics_dict.values())
    
    if query:
        search_term = query.lower()
        topics_list = [t for t in topics_list if search_term in t["name"].lower()]
    
    if domain and domain.strip():
        # Filter by domain via concepts
        domain_concepts = db.query(DomainConcept).join(Concept).join(Domain).filter(
            Domain.name.ilike(f"%{domain}%")
        ).all()
        
        if domain_concepts:
            concept_ids = [dc.concept_id for dc in domain_concepts]
            # Get feeds with these concepts
            feed_ids = db.query(FeedConcept.feed_id).filter(
                FeedConcept.concept_id.in_(concept_ids)
            ).distinct().all()
            
            if feed_ids:
                feed_id_list = [fid[0] for fid in feed_ids]
                # Filter topics that appear in these feeds
                filtered_topics = []
                for topic in topics_list:
                    # Check if any feed with this topic is in our filtered list
                    feeds_with_topic = db.query(Feed).filter(
                        Feed.categories.contains([topic["name"]]),
                        Feed.id.in_(feed_id_list)
                    ).count()
                    
                    if feeds_with_topic > 0:
                        filtered_topics.append(topic)
                
                topics_list = filtered_topics
    
    # Sort and paginate
    topics_list.sort(key=lambda x: x["popularity"], reverse=True)
    total = len(topics_list)
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated_topics = topics_list[start_idx:end_idx]
    
    return {
        "tab": "topics",
        "query": query,
        "items": paginated_topics,
        "page": page,
        "limit": limit,
        "total": total,
        "has_more": (page * limit) < total,
        "filters": {
            "domain": domain
        }
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
    user_id: Optional[int] = Query(None, description="User ID for follow status"),
    db: Session = Depends(get_db)
):
    """
    Search sources (websites, YouTube channels, authors).
    
    This is for Tab 4: Sources
    Sources can be followed
    """
    query_obj = db.query(Source).filter(Source.is_active == True)
    
    if query:
        search_term = f"%{query.lower()}%"
        query_obj = query_obj.filter(
            or_(
                Source.name.ilike(search_term),
                Source.website.ilike(search_term)
            )
        )
    
    if source_type:
        query_obj = query_obj.filter(Source.source_type == source_type)
    
    # Count and paginate
    total = query_obj.count()
    query_obj = query_obj.order_by(Source.created_at.desc())
    sources = query_obj.offset((page - 1) * limit).limit(limit).all()
    
    # Check follow status
    followed_source_ids = []
    if user_id:
        followed_sources = db.query(UserSourceFollow).filter(
            UserSourceFollow.user_id == user_id
        ).all()
        followed_source_ids = [fs.source_id for fs in followed_sources]
    
    # Format results
    items = []
    for source in sources:
        # Get feed count
        if source.source_type == "blog":
            feed_count = db.query(Feed).join(Blog).filter(
                Blog.website == source.website
            ).count()
        else:
            feed_count = db.query(Feed).filter(
                Feed.source_type == "youtube"
            ).count()
        
        # Get top topics for this source
        if source.source_type == "blog":
            source_feeds = db.query(Feed).join(Blog).filter(
                Blog.website == source.website
            ).all()
        else:
            source_feeds = db.query(Feed).filter(
                Feed.source_type == "youtube"
            ).all()
        
        topic_counts = {}
        for feed in source_feeds:
            if feed.categories:
                for category in feed.categories:
                    topic_counts[category] = topic_counts.get(category, 0) + 1
        
        top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Get source metadata
        metadata = get_source_metadata(source, db)
        
        # Use metadata description if source description is empty
        source_description = getattr(source, 'description', '')
        if not source_description and metadata.get("description"):
            source_description = metadata["description"]
        
        items.append({
            "id": source.id,
            "name": source.name,
            "website": source.website,
            "source_type": source.source_type,
            "description": source_description,
            "feed_count": feed_count,
            "follower_count": source.follower_count,
            "top_topics": [topic for topic, count in top_topics],
            "is_following": source.id in followed_source_ids,
            "created_at": source.created_at.isoformat() if source.created_at else None,
            "updated_at": source.updated_at.isoformat() if source.updated_at else None,
            "metadata": metadata
        })
    
    # Apply additional filters
    filtered_items = items
    
    if domain and domain.strip():
        # Filter by domain
        filtered_items = []
        for item in items:
            # Check if source has content in this domain
            source_feeds = db.query(Feed).filter(Feed.source_type == item["source_type"])
            
            if item["source_type"] == "blog":
                source_feeds = source_feeds.join(Blog).filter(Blog.website == item["website"])
            
            feed_ids = [f.id for f in source_feeds.all()]
            
            # Check if any feed has concepts in this domain
            domain_obj = db.query(Domain).filter(
                Domain.name.ilike(f"%{domain}%")
            ).first()
            
            if domain_obj:
                domain_concepts = db.query(DomainConcept).filter(
                    DomainConcept.domain_id == domain_obj.id
                ).all()
                concept_ids = [dc.concept_id for dc in domain_concepts]
                
                # Check if any feed has these concepts
                has_domain = db.query(FeedConcept).filter(
                    FeedConcept.feed_id.in_(feed_ids),
                    FeedConcept.concept_id.in_(concept_ids)
                ).first() is not None
                
                if has_domain:
                    filtered_items.append(item)
    
    if topic and topic.strip():
        # Filter by topic
        filtered_items = [item for item in filtered_items if topic in item["top_topics"]]
    
    # Sort by feed count (most feeds first)
    filtered_items.sort(key=lambda x: x["feed_count"], reverse=True)
    total = len(filtered_items)
    
    # Return only paginated items
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated_items = filtered_items[start_idx:end_idx]
    
    return {
        "tab": "sources",
        "query": query,
        "items": paginated_items,
        "page": page,
        "limit": limit,
        "total": total,
        "has_more": (page * limit) < total,
        "filters": {
            "source_type": source_type,
            "domain": domain,
            "topic": topic
        }
    }

# ------------------ Tab 5: Concepts Search ------------------

@router.get("/concepts", response_model=Dict[str, Any])
def search_concepts(
    query: Optional[str] = Query(None, description="Search query"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Results per page"),
    domain: Optional[str] = Query(None, description="Filter by domain"),
    topic: Optional[str] = Query(None, description="Filter by topic"),
    user_id: Optional[int] = Query(None, description="User ID"),
    db: Session = Depends(get_db)
):
    """
    Search concepts extracted from content via LLM.
    
    This is for Tab 5: Concepts
    """
    query_obj = db.query(Concept).filter(Concept.is_active == True)
    
    if query:
        search_term = f"%{query.lower()}%"
        query_obj = query_obj.filter(Concept.name.ilike(search_term))
    
    if domain and domain.strip():
        # Filter by domain
        domain_obj = db.query(Domain).filter(
            Domain.name.ilike(f"%{domain}%")
        ).first()
        
        if domain_obj:
            domain_concepts = db.query(DomainConcept).filter(
                DomainConcept.domain_id == domain_obj.id
            ).all()
            concept_ids = [dc.concept_id for dc in domain_concepts]
            query_obj = query_obj.filter(Concept.id.in_(concept_ids))
    
    # Get all concepts
    all_concepts = query_obj.all()
    
    # Apply topic filter
    filtered_concepts = []
    for concept in all_concepts:
        # Check if this concept appears in feeds with the specified topic
        if topic and topic.strip():
            concept_feeds = concept.feeds
            has_topic = False
            
            for feed in concept_feeds:
                if feed.categories and topic in feed.categories:
                    has_topic = True
                    break
            
            if not has_topic:
                continue
        
        # Calculate feed count
        feed_count = len(concept.feeds)
        
        # Get domains
        domains = [{"id": d.id, "name": d.name} for d in concept.domains]
        
        # Get related concepts
        related_concepts = []
        if concept.related_concepts:
            related_objs = db.query(Concept).filter(
                Concept.name.in_(concept.related_concepts[:5])
            ).all()
            related_concepts = [c.name for c in related_objs]
        
        filtered_concepts.append({
            "id": concept.id,
            "name": concept.name,
            "description": concept.description,
            "feed_count": feed_count,
            "popularity_score": concept.popularity_score,
            "related_concepts": related_concepts,
            "domains": domains,
            "created_at": concept.created_at.isoformat() if concept.created_at else None
        })
    
    # Sort and paginate
    filtered_concepts.sort(key=lambda x: x["feed_count"], reverse=True)
    total = len(filtered_concepts)
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated_items = filtered_concepts[start_idx:end_idx]
    
    return {
        "tab": "concepts",
        "query": query,
        "items": paginated_items,
        "page": page,
        "limit": limit,
        "total": total,
        "has_more": (page * limit) < total,
        "filters": {
            "domain": domain,
            "topic": topic
        }
    }

# ------------------ Unified Search Endpoint ------------------

@router.get("/unified", response_model=Dict[str, Any])
def unified_search(
    query: str = Query(..., min_length=2, description="Search query"),
    tabs: str = Query("all", description="Tabs to search: all or comma-separated list"),
    page: int = Query(1, ge=1, description="Page number"),
    limit_per_tab: int = Query(5, ge=1, le=20, description="Results per tab"),
    user_id: Optional[int] = Query(None, description="User ID for personalization"),
    db: Session = Depends(get_db)
):
    """
    Unified search across all 5 tabs.
    
    Returns results from multiple tabs in one response.
    """
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
        content_results = search_content(query, 1, limit_per_tab, user_id=user_id, db=db)
        results["content"] = {
            "items": content_results["items"][:limit_per_tab],
            "total": content_results["total"],
            "has_more": content_results["has_more"]
        }
    
    if "lists" in tabs_list:
        lists_results = search_lists(query, 1, limit_per_tab, user_id=user_id, db=db)
        results["lists"] = {
            "items": lists_results["items"][:limit_per_tab],
            "total": lists_results["total"],
            "has_more": lists_results["has_more"]
        }
    
    if "topics" in tabs_list:
        topics_results = search_topics(query, 1, limit_per_tab, user_id=user_id, db=db)
        results["topics"] = {
            "items": topics_results["items"][:limit_per_tab],
            "total": topics_results["total"],
            "has_more": topics_results["has_more"]
        }
    
    if "sources" in tabs_list:
        sources_results = search_sources(query, 1, limit_per_tab, user_id=user_id, db=db)
        results["sources"] = {
            "items": sources_results["items"][:limit_per_tab],
            "total": sources_results["total"],
            "has_more": sources_results["has_more"]
        }
    
    if "concepts" in tabs_list:
        concepts_results = search_concepts(query, 1, limit_per_tab, user_id=user_id, db=db)
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
    user_id: Optional[int] = Query(None, description="User ID for personalization"),
    db: Session = Depends(get_db)
):
    """
    Get concept by ID with all associated feeds.
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
    
    # Get feeds with their relationships
    query_obj = db.query(Feed).options(
        joinedload(Feed.blog),
        joinedload(Feed.slides),
        joinedload(Feed.category),
        joinedload(Feed.subcategory),
        joinedload(Feed.concepts)
    ).filter(
        Feed.id.in_(feed_ids_query),
        Feed.status == "ready"
    )
    
    # Count and paginate
    total = query_obj.count()
    query_obj = query_obj.order_by(Feed.created_at.desc())
    feeds = query_obj.offset((page - 1) * limit).limit(limit).all()
    
    # Format feeds
    formatted_feeds = []
    for feed in feeds:
        formatted_feeds.append(format_feed_for_search(feed, db, user_id))
    
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
                "feed_count": len(c.feeds) if hasattr(c, 'feeds') else 0
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
    
    # Get concept statistics
    feed_count = total
    unique_topics = set()
    unique_sources = set()
    
    for feed in feeds:
        # Collect unique topics from categories
        if feed.categories:
            for category in feed.categories:
                unique_topics.add(category)
        
        # Collect unique sources
        if feed.source_type == "youtube":
            meta = get_feed_metadata(feed, db)
            channel_name = meta.get("channel_name")
            if channel_name:
                unique_sources.add(channel_name)
        elif feed.source_type == "blog" and feed.blog:
            unique_sources.add(feed.blog.website)
    
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
            "top_topics": list(unique_topics)[:10],
            "top_sources": list(unique_sources)[:10]
        },
        "domains": domains,
        "related_concepts": related_concepts,
        "feeds": {
            "items": formatted_feeds,
            "page": page,
            "limit": limit,
            "total": total,
            "has_more": (page * limit) < total
        }
    }