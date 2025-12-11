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
    Category, SubCategory, FeedConcept, DomainConcept
)
from feed_router import get_feed_metadata

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
@router.get("/debug/youtube-playlists")
def debug_youtube_playlists(db: Session = Depends(get_db)):
    """Debug endpoint to check YouTube playlist jobs."""
    
    playlist_jobs = db.query(TranscriptJob).filter(
        TranscriptJob.type == "playlist"
    ).all()
    
    job_info = []
    for job in playlist_jobs:
        transcript_count = db.query(Transcript).filter(
            Transcript.job_id == job.id
        ).count()
        
        # Parse playlists
        playlists = []
        if job.playlists:
            try:
                playlists = json.loads(job.playlists)
            except json.JSONDecodeError:
                pass
        
        job_info.append({
            "job_id": job.job_id,
            "url": job.url,
            "content_name": job.content_name,
            "status": job.status.value,
            "transcript_count": transcript_count,
            "playlists": playlists,
            "created_at": job.created_at
        })
    
    return {
        "total_playlist_jobs": len(playlist_jobs),
        "completed_jobs": len([j for j in playlist_jobs if j.status.value == "completed"]),
        "jobs": job_info
    }


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

@router.get("/debug/lists")
def debug_lists(
    db: Session = Depends(get_db)
):
    """
    Debug endpoint to check list data.
    """
    lists = db.query(ContentList).all()
    
    debug_info = []
    for clist in lists:
        debug_info.append({
            "id": clist.id,
            "name": clist.name,
            "description": clist.description,
            "source_type": clist.source_type,
            "is_active": clist.is_active,
            "feed_ids": clist.feed_ids if hasattr(clist, 'feed_ids') else None,
            "feed_count": len(clist.feed_ids) if clist.feed_ids else 0,
            "created_at": clist.created_at
        })
    
    return {
        "total_lists": len(lists),
        "lists": debug_info
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
    Topics cannot be followed or bookmarked
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
            for category in feed.categories:
                if category not in topics_dict:
                    # Count feeds for this topic
                    feed_count = db.query(Feed).filter(
                        Feed.categories.contains([category]),
                        Feed.status == "ready"
                    ).count()
                    
                    topics_dict[category] = {
                        "name": category,
                        "feed_count": feed_count,
                        "popularity": feed_count
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
    
    # Get all sources
    all_sources = query_obj.all()
    
    # Check follow status
    followed_source_ids = []
    if user_id:
        followed_sources = db.query(UserSourceFollow).filter(
            UserSourceFollow.user_id == user_id
        ).all()
        followed_source_ids = [fs.source_id for fs in followed_sources]
    
    # Format results
    items = []
    for source in all_sources:
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
        
        items.append({
            "id": source.id,
            "name": source.name,
            "website": source.website,
            "source_type": source.source_type,
            "feed_count": feed_count,
            "follower_count": source.follower_count,
            "top_topics": [topic for topic, count in top_topics],
            "is_following": source.id in followed_source_ids,
            "created_at": source.created_at.isoformat() if source.created_at else None
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
    
    # Sort and paginate
    filtered_items.sort(key=lambda x: x["feed_count"], reverse=True)
    total = len(filtered_items)
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