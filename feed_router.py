from fastapi import APIRouter, Depends, HTTPException, Response, BackgroundTasks
from sqlalchemy.orm import Session, joinedload
from datetime import datetime
import os
import re
import logging
import json
from typing import List, Optional, Dict, Any
from database import get_db
from models import Blog, Category, Feed, Slide, Transcript, TranscriptJob, Source,PublishedFeed
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from schemas import FeedRequest, DeleteSlideRequest, YouTubeFeedRequest
from sqlalchemy import or_, String
from enum import Enum

router = APIRouter(prefix="/get", tags=["Feeds"])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ContentType(str, Enum):
    WEBINAR = "Webinar"
    BLOG = "Blog"
    PODCAST = "Podcast"
    VIDEO = "Video"

# ------------------ AI Categorization Function ------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def categorize_content_with_openai(content: str, admin_categories: list) -> tuple:
    """Categorize content using OpenAI and extract skills, tools, roles."""
    try:
        truncated_content = content[:4000] + "..." if len(content) > 4000 else content
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": """You are a content analysis assistant. Analyze the content and:
                    1. Categorize it into the provided categories
                    2. Extract relevant skills mentioned
                    3. Extract tools/technologies mentioned  
                    4. Extract relevant job roles
                    5. Determine content type (Blog, Video, Podcast, Webinar)
                    
                    Return JSON with this structure:
                    {
                        "categories": ["category1", "category2"],
                        "skills": ["skill1", "skill2"],
                        "tools": ["tool1", "tool2"],
                        "roles": ["role1", "role2"],
                        "content_type": "Blog/Video/Podcast/Webinar"
                    }"""
                },
                {
                    "role": "user",
                    "content": f"Available categories: {', '.join(admin_categories)}.\n\nContent:\n{truncated_content}\n\nReturn JSON with categories, skills, tools, roles, and content_type."
                }
            ],
            temperature=0.1,
            max_tokens=500,
            response_format={ "type": "json_object" }
        )
        
        analysis = json.loads(response.choices[0].message.content.strip())
        
        # Validate categories
        categories = analysis.get("categories", [])
        matched_categories = []
        for item in categories:
            for category in admin_categories:
                if (item.lower() == category.lower() or 
                    item.lower() in category.lower() or 
                    category.lower() in item.lower()):
                    matched_categories.append(category)
                    break
        
        # Remove duplicates and limit
        seen = set()
        unique_categories = []
        for cat in matched_categories:
            if cat not in seen:
                seen.add(cat)
                unique_categories.append(cat)
        
        if not unique_categories:
            raise HTTPException(status_code=500, detail="OpenAI failed to categorize content")
        
        # Extract other metadata
        skills = list(set(analysis.get("skills", [])))[:10]
        tools = list(set(analysis.get("tools", [])))[:10]
        roles = list(set(analysis.get("roles", [])))[:10]
        
        # Determine content type
        content_type_str = analysis.get("content_type", "Blog")
        content_type = ContentType.BLOG  # default
        
        if "video" in content_type_str.lower() or "youtube" in content_type_str.lower():
            content_type = ContentType.VIDEO
        elif "podcast" in content_type_str.lower():
            content_type = ContentType.PODCAST
        elif "webinar" in content_type_str.lower():
            content_type = ContentType.WEBINAR
        
        return unique_categories[:3], skills, tools, roles, content_type
        
    except Exception as e:
        logger.error(f"OpenAI categorization error: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI categorization failed: {str(e)}")

# ------------------ AI Content Generation Functions ------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_feed_content_with_ai(title: str, content: str, categories: List[str], content_type: str = "blog") -> Dict[str, Any]:
    """Generate engaging feed content using AI."""
    try:
        truncated_content = content[:6000] + "..." if len(content) > 6000 else content
        
        system_prompt = {
            "blog": """You are a content summarization and presentation expert. Create an engaging, structured feed from blog content. 
            
            Return JSON with the following structure:
            {
                "title": "Engaging title that captures the essence",
                "summary": "2-3 paragraph comprehensive summary of the main content",
                "key_points": ["Point 1", "Point 2", "Point 3", "Point 4", "Point 5"],
                "conclusion": "1-2 paragraph concluding thoughts or key takeaways"
            }""",
            
            "transcript": """You are a video content summarization expert. Create an engaging, structured feed from video transcript content.
            
            Return JSON with the following structure:
            {
                "title": "Engaging title that captures the video's essence", 
                "summary": "2-3 paragraph summary of the video's main content and message",
                "key_points": ["Key insight 1", "Key insight 2", "Key insight 3", "Key insight 4", "Key insight 5"],
                "conclusion": "1-2 paragraph conclusion with main takeaways and actionable advice"
            }"""
        }
        
        user_prompt = f"""
        Original Title: {title}
        Content Type: {content_type}
        Categories: {', '.join(categories)}
        
        Content:
        {truncated_content}
        
        Please generate engaging, structured feed content in the specified JSON format.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": system_prompt.get(content_type, system_prompt["blog"])
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.7,
            max_tokens=2000,
            response_format={ "type": "json_object" }
        )
        
        content = response.choices[0].message.content
        ai_content = json.loads(content)
        
        # Validate required fields
        required_fields = ["title", "summary", "key_points", "conclusion"]
        for field in required_fields:
            if field not in ai_content:
                raise HTTPException(status_code=500, detail=f"OpenAI response missing required field: {field}")
        
        return ai_content
        
    except Exception as e:
        logger.error(f"OpenAI feed generation error: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI feed generation failed: {str(e)}")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_slide_with_ai(slide_type: str, context: str, categories: List[str], content_type: str = "blog", previous_slides: List[Dict] = None) -> Dict[str, Any]:
    """Generate a specific slide type using AI."""
    try:
        system_prompt = {
            "role": "system", 
            "content": f"""You are a presentation design expert. Create engaging, concise slides.
            Return ONLY valid JSON with this structure:
            {{
                "title": "Engaging slide title",
                "body": "Substantive body content that summarizes key points",
                "bullets": ["Bullet point 1", "Bullet point 2", "Bullet point 3"]
            }}"""
        }
        
        messages = [
            system_prompt,
            {
                "role": "user",
                "content": f"Context: {context}\nCategories: {', '.join(categories)}\nGenerate engaging slide content as valid JSON with title, body, and bullets array."
            }
        ]
        
        if previous_slides:
            previous_titles = [f"Slide {s.get('order')}: {s.get('title', '')}" for s in previous_slides[-2:]]
            messages[1]["content"] += f"\nPrevious slides (avoid repetition): {', '.join(previous_titles)}"
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=1200,
            response_format={ "type": "json_object" }
        )
        
        content = response.choices[0].message.content
        slide_data = json.loads(content)
        
        # Validate content quality
        title = slide_data.get("title", "").strip()
        body = slide_data.get("body", "").strip()
        bullets = slide_data.get("bullets", [])
        
        if not title or len(body) < 50:
            raise HTTPException(status_code=500, detail="OpenAI generated insufficient slide content")
        
        return {
            "title": title,
            "body": body,
            "bullets": bullets,
            "background_color": "#FFFFFF",
            "source_refs": [],
            "render_markdown": True
        }
        
    except Exception as e:
        logger.error(f"OpenAI slide generation error for {slide_type}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OpenAI slide generation failed for {slide_type}: {str(e)}")

def generate_slides_with_ai(title: str, content: str, ai_generated_content: Dict[str, Any], categories: List[str], content_type: str = "blog") -> List[Dict]:
    """Generate presentation slides using AI."""
    import random
    import hashlib
    
    # Determine slide count based on content richness (1-10 slides)
    content_length = len(content)
    key_points_count = len(ai_generated_content.get("key_points", []))
    summary_length = len(ai_generated_content.get("summary", ""))
    
    richness_score = min(
        (content_length / 1000) +
        (key_points_count * 0.5) +
        (summary_length / 500),
        10.0
    )
    
    base_slides = max(1, min(10, int(richness_score) + random.randint(-1, 2)))
    slide_count = max(1, min(10, base_slides))
    
    logger.info(f"Generating {slide_count} slides for '{title}'")
    
    # Generate background color for this feed
    background_color = generate_feed_background_color(title, content_type, categories)
    
    slides = []
    
    try:
        # 1. Title slide
        title_context = f"Create an engaging title slide for: {title}\nSummary: {ai_generated_content.get('summary', '')}\nCategories: {', '.join(categories)}\nType: {content_type}"
        title_slide = generate_slide_with_ai("title", title_context, categories, content_type)
        title_slide["order"] = 1
        title_slide["background_color"] = background_color
        slides.append(title_slide)
        
        if slide_count == 1:
            return slides
        
        # 2. Summary slide
        summary_context = f"Create a comprehensive summary slide for: {title}\nFull Summary: {ai_generated_content.get('summary', '')}\nType: {content_type}"
        summary_slide = generate_slide_with_ai("summary", summary_context, categories, content_type, slides)
        summary_slide["order"] = 2
        summary_slide["background_color"] = background_color
        slides.append(summary_slide)
        
        if slide_count == 2:
            return slides
        
        # 3. Key point slides
        key_points = ai_generated_content.get("key_points", [])
        remaining_slides = slide_count - 2
        key_point_slides_count = min(remaining_slides, len(key_points), 5)
        
        for i in range(key_point_slides_count):
            if i < len(key_points):
                point = key_points[i]
                key_point_context = f"Create a detailed key point slide for: {title}\nKey Point: {point}\nOther Key Points: {', '.join(key_points[:i] + key_points[i+1:][:2])}\nType: {content_type}"
                key_slide = generate_slide_with_ai("key_point", key_point_context, categories, content_type, slides)
                key_slide["order"] = len(slides) + 1
                key_slide["background_color"] = background_color
                slides.append(key_slide)
        
        # 4. Conclusion slide if space
        if len(slides) < slide_count and ai_generated_content.get("conclusion"):
            conclusion_context = f"Create a conclusion slide for: {title}\nConclusion: {ai_generated_content.get('conclusion', '')}\nKey Points Covered: {', '.join(key_points[:3])}\nType: {content_type}"
            conclusion_slide = generate_slide_with_ai("conclusion", conclusion_context, categories, content_type, slides)
            conclusion_slide["order"] = len(slides) + 1
            conclusion_slide["background_color"] = background_color
            slides.append(conclusion_slide)
        
        # 5. Fill remaining slots
        while len(slides) < slide_count:
            insight_context = f"Create an additional insights slide for: {title}\nAvailable Content: Summary - {ai_generated_content.get('summary', '')[:200]}...\nRemaining Key Points: {', '.join(key_points[len(slides)-2:]) if len(slides)-2 < len(key_points) else 'Various important aspects'}\nType: {content_type}"
            insight_slide = generate_slide_with_ai("additional_insights", insight_context, categories, content_type, slides)
            insight_slide["order"] = len(slides) + 1
            insight_slide["background_color"] = background_color
            slides.append(insight_slide)
        
        logger.info(f"Successfully generated {len(slides)} slides for '{title}'")
        return slides[:slide_count]
        
    except Exception as e:
        logger.error(f"Error in generate_slides_with_ai for '{title}': {str(e)}")
        raise

def generate_feed_background_color(title: str, content_type: str, categories: List[str]) -> str:
    """Generate background color from predefined distinct colors."""
    import hashlib
    
    distinct_colors = [
        "#1e3a8a", "#166534", "#581c87", "#991b1b", "#9a3412",
        "#115e59", "#831843", "#854d0e", "#3730a3", "#064e3b",
        "#1d4ed8", "#15803d", "#7c3aed", "#dc2626", "#ea580c",
        "#0f766e", "#be185d", "#ca8a04", "#4f46e5", "#047857",
    ]
    
    seed_string = f"{title}_{content_type}_{'_'.join(sorted(categories))}"
    hash_object = hashlib.md5(seed_string.encode())
    hash_int = int(hash_object.hexdigest()[:8], 16)
    
    color_index = hash_int % len(distinct_colors)
    color_hex = distinct_colors[color_index]
    
    logger.info(f"Generated predefined distinct color: {color_hex} for '{title}'")
    return color_hex

# ------------------ Core Feed Creation Functions ------------------

def create_feed_from_blog(db: Session, blog: Blog):
    """Generate feed and slides from a blog using AI."""
    try:
        # Check if feed already exists for this blog
        existing_feed = db.query(Feed).filter(Feed.blog_id == blog.id).first()
        if existing_feed:
            db.query(Slide).filter(Slide.feed_id == existing_feed.id).delete()
            db.flush()
        
        # Create new feed with enhanced metadata
        admin_categories = [c.name for c in db.query(Category).filter(Category.is_active == True).all()]
        if not admin_categories:
            raise HTTPException(status_code=500, detail="No active categories found in database")
            
        categories, skills, tools, roles, content_type = categorize_content_with_openai(blog.content, admin_categories)
        ai_generated_content = generate_feed_content_with_ai(blog.title, blog.content, categories, "blog")
        slides_data = generate_slides_with_ai(blog.title, blog.content, ai_generated_content, categories, "blog")
        
        feed_title = ai_generated_content.get("title", blog.title)
        feed = Feed(
            blog_id=blog.id, 
            title=feed_title,
            categories=categories, 
            skills=skills,
            tools=tools,
            roles=roles,
            content_type=content_type,
            status="ready",
            ai_generated_content=ai_generated_content,
            image_generation_enabled=False,
            source_type="blog",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(feed)
        db.flush()
        
        for slide_data in slides_data:
            slide = Slide(
                feed_id=feed.id,
                order=slide_data["order"],
                title=slide_data["title"],
                body=slide_data["body"],
                bullets=slide_data.get("bullets"),
                background_color=slide_data.get("background_color", "#FFFFFF"),
                source_refs=slide_data.get("source_refs", []),
                render_markdown=int(slide_data.get("render_markdown", True)),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.add(slide)
        
        db.commit()
        db.refresh(feed)
        logger.info(f"Successfully created AI-generated feed {feed.id} for blog {blog.id}")
        return feed
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating AI-generated feed for blog {blog.id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create feed for blog: {str(e)}")

def create_feed_from_transcript(db: Session, transcript: Transcript, overwrite: bool = False):
    """Generate feed and slides from a YouTube transcript using AI."""
    try:
        existing_feed = db.query(Feed).filter(Feed.transcript_id == transcript.transcript_id).first()
        
        if existing_feed and not overwrite:
            logger.info(f"Feed already exists for transcript {transcript.transcript_id}, skipping")
            return existing_feed
            
        admin_categories = [c.name for c in db.query(Category).filter(Category.is_active == True).all()]
        if not admin_categories:
            raise HTTPException(status_code=500, detail="No active categories found in database")
            
        categories, skills, tools, roles, content_type = categorize_content_with_openai(transcript.transcript_text, admin_categories)
        ai_generated_content = generate_feed_content_with_ai(transcript.title, transcript.transcript_text, categories, "transcript")
        slides_data = generate_slides_with_ai(transcript.title, transcript.transcript_text, ai_generated_content, categories, "transcript")
        
        feed_title = ai_generated_content.get("title", transcript.title)
        
        if existing_feed and overwrite:
            # UPDATE existing feed
            existing_feed.title = feed_title
            existing_feed.categories = categories
            existing_feed.skills = skills
            existing_feed.tools = tools
            existing_feed.roles = roles
            existing_feed.content_type = content_type
            existing_feed.status = "ready"
            existing_feed.ai_generated_content = ai_generated_content
            existing_feed.updated_at = datetime.utcnow()
            
            # Delete old slides
            db.query(Slide).filter(Slide.feed_id == existing_feed.id).delete()
            db.flush()
            
            # Create new slides
            for slide_data in slides_data:
                slide = Slide(
                    feed_id=existing_feed.id,
                    order=slide_data["order"],
                    title=slide_data["title"],
                    body=slide_data["body"],
                    bullets=slide_data.get("bullets"),
                    background_color=slide_data.get("background_color", "#FFFFFF"),
                    source_refs=slide_data.get("source_refs", []),
                    render_markdown=int(slide_data.get("render_markdown", True)),
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                db.add(slide)
            
            db.commit()
            db.refresh(existing_feed)
            logger.info(f"Successfully UPDATED AI-generated feed {existing_feed.id} for transcript {transcript.transcript_id}")
            return existing_feed
        else:
            # CREATE new feed
            feed = Feed(
                transcript_id=transcript.transcript_id,
                title=feed_title,
                categories=categories,
                skills=skills,
                tools=tools,
                roles=roles,
                content_type=content_type,
                status="ready",
                ai_generated_content=ai_generated_content,
                image_generation_enabled=False,
                source_type="youtube",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.add(feed)
            db.flush()
            
            # Create slides
            for slide_data in slides_data:
                slide = Slide(
                    feed_id=feed.id,
                    order=slide_data["order"],
                    title=slide_data["title"],
                    body=slide_data["body"],
                    bullets=slide_data.get("bullets"),
                    background_color=slide_data.get("background_color", "#FFFFFF"),
                    source_refs=slide_data.get("source_refs", []),
                    render_markdown=int(slide_data.get("render_markdown", True)),
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                db.add(slide)
            
            db.commit()
            db.refresh(feed)
            logger.info(f"Successfully CREATED AI-generated feed {feed.id} for transcript {transcript.transcript_id}")
            return feed
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating/updating AI-generated feed for transcript {transcript.transcript_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create feed for transcript: {str(e)}")

# ------------------ New Endpoints for Feed Router ------------------

@router.get("/sources/with-ids", response_model=List[dict])
def get_sources_with_ids(
    page: int = 1,
    limit: int = 50,
    active_only: bool = True,
    db: Session = Depends(get_db)
):
    """Get all sources with their IDs and feed counts."""
    try:
        query = db.query(Source)
        
        if active_only:
            query = query.filter(Source.is_active == True)
        
        total = query.count()
        sources = query.offset((page - 1) * limit).limit(limit).all()
        
        sources_with_counts = []
        for source in sources:
            # Count ALL feeds for this source (not just published)
            if source.source_type == "blog":
                feed_count = db.query(Feed).join(Blog).filter(
                    Blog.website == source.website
                ).count()
            else:  # youtube
                feed_count = db.query(Feed).filter(
                    Feed.source_type == "youtube"
                ).count()
            
            sources_with_counts.append({
                "id": source.id,
                "name": source.name,
                "website": source.website,
                "source_type": source.source_type,
                "feed_count": feed_count,
                "follower_count": source.follower_count,
                "is_active": source.is_active
            })
        
        return sources_with_counts
        
    except Exception as e:
        logger.error(f"Error fetching sources with IDs: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch sources")

@router.get("/source/{source_id}/feeds", response_model=dict)
def get_feeds_by_source_id(
    source_id: int,
    page: int = 1,
    limit: int = 20,
    content_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get all feeds (both published and unpublished) by source ID."""
    try:
        source = db.query(Source).filter(Source.id == source_id).first()
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        
        # Build query based on source type
        if source.source_type == "blog":
            query = db.query(Feed).options(
                joinedload(Feed.slides),
                joinedload(Feed.blog)
            ).join(Blog).filter(
                Blog.website == source.website
            )
        else:
            query = db.query(Feed).options(
                joinedload(Feed.slides)
            ).filter(
                Feed.source_type == "youtube"
            )
        
        # Apply content type filter with safe handling
        if content_type:
            try:
                # Convert string to FilterType enum
                content_type_enum = FilterType(content_type)
                query = query.filter(Feed.content_type == content_type_enum)
            except ValueError:
                # If invalid content_type provided, ignore the filter
                logger.warning(f"Invalid content_type provided: {content_type}")
        
        query = query.order_by(Feed.created_at.desc())
        
        total = query.count()
        feeds = query.offset((page - 1) * limit).limit(limit).all()
        
        # Format response with safe attribute access
        feeds_data = []
        for feed in feeds:
            # Check if published
            is_published = db.query(PublishedFeed).filter(
                PublishedFeed.feed_id == feed.id,
                PublishedFeed.is_active == True
            ).first() is not None
            
            # Safely get content_type
            feed_content_type = feed.content_type.value if feed.content_type else FilterType.BLOG.value
            
            feeds_data.append({
                "id": feed.id,
                "title": feed.title,
                "categories": feed.categories or [],
                "content_type": feed_content_type,
                "skills": getattr(feed, 'skills', []) or [],
                "tools": getattr(feed, 'tools', []) or [],
                "roles": getattr(feed, 'roles', []) or [],
                "status": feed.status,
                "is_published": is_published,
                "slides_count": len(feed.slides) if feed.slides else 0,
                "created_at": feed.created_at.isoformat() if feed.created_at else None,
                "source_type": feed.source_type
            })
        
        return {
            "source": {
                "id": source.id,
                "name": source.name,
                "website": source.website,
                "source_type": source.source_type
            },
            "feeds": feeds_data,
            "total": total,
            "page": page,
            "limit": limit,
            "has_more": (page * limit) < total
        }
        
    except Exception as e:
        logger.error(f"Error fetching feeds for source {source_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch feeds for source")

# ------------------ Existing Feed Router Endpoints ------------------

@router.get("/all", response_model=dict)
def get_all_feeds(
    response: Response, 
    page: int = 1, 
    limit: int = 20, 
    category: Optional[str] = None,
    status: Optional[str] = None,
    source_type: Optional[str] = None,
    is_published: Optional[bool] = None,
    content_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get all feeds (both published and unpublished) with filtering."""
    query = db.query(Feed).options(joinedload(Feed.blog))
    
    if category:
        query = query.filter(Feed.categories.contains([category]))
    if status:
        query = query.filter(Feed.status == status)
    if source_type:
        query = query.filter(Feed.source_type == source_type)
    if content_type:
        try:
            content_type_enum = FilterType(content_type)
            query = query.filter(Feed.content_type == content_type_enum)
        except ValueError:
            logger.warning(f"Invalid content_type provided: {content_type}")
    if is_published is not None:
        if is_published:
            published_feed_ids = db.query(PublishedFeed.feed_id).filter(PublishedFeed.is_active == True)
            query = query.filter(Feed.id.in_(published_feed_ids))
        else:
            published_feed_ids = db.query(PublishedFeed.feed_id).filter(PublishedFeed.is_active == True)
            query = query.filter(~Feed.id.in_(published_feed_ids))
    
    query = query.order_by(Feed.created_at.desc())
    total = query.count()
    feeds = query.offset((page - 1) * limit).limit(limit).all()

    items = []
    for feed in feeds:
        is_published_status = db.query(PublishedFeed).filter(
            PublishedFeed.feed_id == feed.id,
            PublishedFeed.is_active == True
        ).first() is not None
        
        feed_content_type = feed.content_type.value if feed.content_type else FilterType.BLOG.value
        
        items.append({
            "id": feed.id,
            "blog_id": feed.blog_id,
            "transcript_id": feed.transcript_id,
            "title": feed.title,
            "categories": feed.categories,
            "content_type": feed_content_type,
            "skills": getattr(feed, 'skills', []) or [],
            "tools": getattr(feed, 'tools', []) or [],
            "roles": getattr(feed, 'roles', []) or [],
            "status": feed.status,
            "source_type": feed.source_type or "blog",
            "is_published": is_published_status,
            "slides_count": len(feed.slides) if feed.slides else 0,
            "created_at": feed.created_at.isoformat() if feed.created_at else None,
            "updated_at": feed.updated_at.isoformat() if feed.updated_at else None,
            "ai_generated": feed.ai_generated_content is not None,
        })

    has_more = (page * limit) < total
    response.headers["X-Total-Count"] = str(total)
    response.headers["X-Page"] = str(page)
    response.headers["X-Limit"] = str(limit)
    
    return {
        "items": items,
        "page": page, 
        "limit": limit, 
        "total": total, 
        "has_more": has_more
    }
@router.get("/{feed_id}", response_model=dict)
def get_feed_by_id(feed_id: int, db: Session = Depends(get_db)):
    """Get full AI-generated feed with slides and is_published status."""
    feed = db.query(Feed).options(
        joinedload(Feed.blog), 
        joinedload(Feed.slides),
        joinedload(Feed.published_feed)
    ).filter(Feed.id == feed_id).first()
    
    if not feed:
        raise HTTPException(status_code=404, detail="Feed not found")

    ai_content = getattr(feed, 'ai_generated_content', {})
    
    # Determine if the feed is published
    is_published = feed.published_feed is not None
    
    # Get proper metadata
    meta = get_feed_metadata(feed, db)
    
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
        "slides": sorted([
            {
                "id": s.id,
                "order": s.order,
                "title": s.title,
                "body": s.body,
                "bullets": s.bullets,
                "background_color": s.background_color,
                "background_image_prompt": None,
                "source_refs": s.source_refs,
                "render_markdown": bool(s.render_markdown),
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "updated_at": s.updated_at.isoformat() if s.updated_at else None
            } for s in feed.slides
        ], key=lambda x: x["order"]),
        "created_at": feed.created_at.isoformat() if feed.created_at else None,
        "updated_at": feed.updated_at.isoformat() if feed.updated_at else None,
        "ai_generated": hasattr(feed, 'ai_generated_content') and feed.ai_generated_content is not None,
        "images_generated": False
    }

@router.post("/feeds", response_model=dict)
def create_feeds_from_website(
    request: FeedRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Create feeds for all blogs from a website - AI only, no fallbacks."""
    blogs = db.query(Blog).filter(Blog.website == request.website).all()
    if not blogs:
        raise HTTPException(status_code=404, detail="No blogs found for this website")

    background_tasks.add_task(
        process_blog_feeds_creation,
        db,
        blogs,
        request.website,
        request.overwrite,
        True,  # Always use AI now - no fallbacks
    )

    return {
        "website": request.website,
        "total_blogs": len(blogs),
        "use_ai": True,  # Always true now
        "generate_images": False,
        "source_type": "blog",
        "message": "Blog feed creation process started in background (AI-only, no fallbacks)",
        "status": "processing",
        # "warning": "If OpenAI API fails, feed creation will fail with clear error messages"
    }

@router.post("/feeds/youtube", response_model=dict)
def create_feeds_from_youtube(
    request: YouTubeFeedRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Create feeds from YouTube transcripts - AI only, no fallbacks."""
    transcripts = []
    job_identifier = "all_transcripts"
    
    if request.job_id:
        transcript_job = db.query(TranscriptJob).filter(TranscriptJob.job_id == request.job_id).first()
        if transcript_job:
            transcripts = db.query(Transcript).filter(Transcript.job_id == transcript_job.id).all()
            job_identifier = f"job_{request.job_id}"
        else:
            video_transcript = db.query(Transcript).filter(Transcript.video_id == request.job_id).first()
            if video_transcript:
                transcripts = [video_transcript]
                job_identifier = f"video_{request.job_id}"
            else:
                raise HTTPException(
                    status_code=404, 
                    detail=f"No transcripts found for job ID: {request.job_id}"
                )
    elif request.video_id:
        transcripts = db.query(Transcript).filter(Transcript.video_id == request.video_id).all()
        job_identifier = f"video_{request.video_id}"
        if not transcripts:
            raise HTTPException(status_code=404, detail=f"No transcript found for video ID: {request.video_id}")
    else:
        transcripts = db.query(Transcript).all()
        if not transcripts:
            raise HTTPException(status_code=404, detail="No transcripts found in database")

    if not transcripts:
        raise HTTPException(status_code=404, detail="No transcripts found for the given criteria")

    background_tasks.add_task(
        process_transcript_feeds_creation,
        db,
        transcripts,
        job_identifier,
        request.overwrite,
        True,  # Always use AI now - no fallbacks
    )

    return {
        "job_id": request.job_id,
        "video_id": request.video_id,
        "total_transcripts": len(transcripts),
        "use_ai": True,  # Always true now
        "generate_images": False,
        "source_type": "youtube",
        "message": f"YouTube transcript feed creation process started for {len(transcripts)} transcripts (AI-only, no fallbacks)",
        "status": "processing",
        # "warning": "If OpenAI API fails, feed creation will fail with clear error messages"
    }

@router.delete("/feeds/slides", response_model=dict)
def delete_slide_from_feed(
    request: DeleteSlideRequest, 
    db: Session = Depends(get_db)
):
    """Delete a specific slide from a feed using request body."""
    feed_id = request.feed_id
    slide_id = request.slide_id

    # First verify the feed exists
    feed = db.query(Feed).filter(Feed.id == feed_id).first()
    if not feed:
        raise HTTPException(status_code=404, detail="Feed not found")
    
    # Find the specific slide
    slide = db.query(Slide).filter(
        Slide.id == slide_id, 
        Slide.feed_id == feed_id
    ).first()
    
    if not slide:
        raise HTTPException(status_code=404, detail="Slide not found in this feed")
    
    try:
        # Delete the slide
        db.delete(slide)
        db.commit()
        
        logger.info(f"Successfully deleted slide {slide_id} from feed {feed_id}")
        
        return {
            "message": "Slide deleted successfully",
            "feed_id": feed_id,
            "slide_id": slide_id,
            "deleted_slide_title": slide.title
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting slide {slide_id} from feed {feed_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete slide")

@router.get("/source/{website}/categorized", response_model=dict)
def get_categorized_feeds_by_source(
    website: str,
    response: Response, 
    page: int = 1, 
    limit: int = 20,
    exclude_uncategorized: bool = True,
    db: Session = Depends(get_db)
):
    """Get categorized feeds from a specific source URL/website."""
    # First, verify the website exists and get its blogs
    blogs = db.query(Blog).filter(Blog.website == website).all()
    if not blogs:
        raise HTTPException(status_code=404, detail=f"No blogs found for website: {website}")
    
    # Get blog IDs for this website
    blog_ids = [blog.id for blog in blogs]
    
    # Query feeds for these blog IDs with category filters
    query = db.query(Feed).options(joinedload(Feed.blog)).filter(Feed.blog_id.in_(blog_ids))
    
    # Filter feeds that have categories
    query = query.filter(Feed.categories.isnot(None))
    
    if exclude_uncategorized:
        # Exclude feeds that contain "Uncategorized" in their categories
        query = query.filter(~Feed.categories.contains(["Uncategorized"]))
    
    # Order by creation date (newest first)
    query = query.order_by(Feed.created_at.desc())
    
    # Get total count and paginated results
    total = query.count()
    feeds = query.offset((page - 1) * limit).limit(limit).all()

    # Format the response
    items = []
    for f in feeds:
        # Additional validation to ensure meaningful categories
        if f.categories and (not exclude_uncategorized or "Uncategorized" not in f.categories):
            # Get proper metadata
            meta = get_feed_metadata(f, db)
            
            items.append({
                "id": f.id,
                "blog_id": f.blog_id,
                "title": f.title,
                "categories": f.categories,
                "status": f.status,
                "slides_count": len(f.slides),
                "meta": meta,
                "created_at": f.created_at.isoformat() if f.created_at else None,
                "updated_at": f.updated_at.isoformat() if f.updated_at else None,
                "ai_generated": hasattr(f, 'ai_generated_content') and f.ai_generated_content is not None,
                "images_generated": False
            })

    has_more = (page * limit) < total
    response.headers["X-Total-Count"] = str(total)
    response.headers["X-Page"] = str(page)
    response.headers["X-Limit"] = str(limit)

    return {
        "website": website,
        "items": items, 
        "page": page, 
        "limit": limit, 
        "total": total, 
        "has_more": has_more,
        "filters": {
            "exclude_uncategorized": exclude_uncategorized
        }
    }
# Add these to your feed_router.py

# ------------------ Search & Topic Management ------------------

# @router.get("/search", response_model=dict)
# def search_feeds_and_topics(
#     query: str,
#     page: int = 1,
#     limit: int = 20,
#     search_type: str = "all",  # all, topics, feeds, sources
#     user_id: Optional[int] = None,
#     auto_create: bool = True,  # Auto-create topics/sources if none exist
#     db: Session = Depends(get_db)
# ):
#     """Search across feeds, topics, and sources."""
#     if not query or len(query.strip()) < 2:
#         raise HTTPException(status_code=400, detail="Query must be at least 2 characters long")
    
#     # Auto-create topics and sources if none exist and flag is enabled
#     if auto_create:
#         auto_create_topics_from_feeds(db)
#         auto_create_sources_from_feeds(db)
    
#     search_query = f"%{query.strip().lower()}%"
#     results = {
#         "query": query,
#         "page": page,
#         "limit": limit,
#         "search_type": search_type
#     }
    
#     # Search topics - improved search
#     if search_type in ["all", "topics"]:
#         topics_query = db.query(Topic).filter(
#             Topic.is_active == True,
#             or_(
#                 Topic.name.ilike(search_query),
#                 Topic.description.ilike(search_query)
#             )
#         )
        
#         total_topics = topics_query.count()
        
#         # If no topics found, try to find relevant categories from feeds
#         if total_topics == 0 and search_type in ["all", "topics"]:
#             # Search for categories in feeds that match the query
#             matching_feeds = db.query(Feed).filter(
#                 Feed.status == "ready",
#                 Feed.categories.isnot(None),
#                 or_(
#                     Feed.categories.cast(String).ilike(search_query),
#                     Feed.title.ilike(search_query)
#                 )
#             ).all()
            
#             # Extract unique categories from matching feeds
#             relevant_categories = set()
#             for feed in matching_feeds:
#                 if feed.categories:
#                     for category in feed.categories:
#                         if query.lower() in category.lower():
#                             relevant_categories.add(category)
            
#             # Create temporary topic results
#             topic_results = []
#             for category_name in list(relevant_categories)[:limit]:
#                 feed_count = db.query(Feed).filter(
#                     Feed.categories.contains([category_name]),
#                     Feed.status == "ready"
#                 ).count()
                
#                 # Check if user follows this topic (by name since it might not be in Topic table yet)
#                 is_following = False
#                 if user_id:
#                     topic_in_db = db.query(Topic).filter(Topic.name == category_name).first()
#                     if topic_in_db:
#                         is_following = db.query(UserTopicFollow).filter(
#                             UserTopicFollow.user_id == user_id,
#                             UserTopicFollow.topic_id == topic_in_db.id
#                         ).first() is not None
                
#                 topic_results.append({
#                     "id": None,  # Not in Topic table yet
#                     "name": category_name,
#                     "description": f"Category from feeds: {category_name}",
#                     "feed_count": feed_count,
#                     "follower_count": 0,
#                     "is_following": is_following,
#                     "created_at": None,
#                     "is_auto_discovered": True  # Flag to indicate this came from feed categories
#                 })
            
#             results["topics"] = {
#                 "items": topic_results,
#                 "total": len(topic_results),
#                 "has_more": False,
#                 "auto_discovered": True  # Indicate these are from feed categories
#             }
#         else:
#             # Use existing topics from database
#             topics = topics_query.offset((page - 1) * limit).limit(limit).all()
            
#             topic_results = []
#             for topic in topics:
#                 # Get feed count for this topic
#                 feed_count = db.query(Feed).filter(
#                     Feed.categories.contains([topic.name]),
#                     Feed.status == "ready"
#                 ).count()
                
#                 # Check if user follows this topic
#                 is_following = False
#                 if user_id:
#                     is_following = db.query(UserTopicFollow).filter(
#                         UserTopicFollow.user_id == user_id,
#                         UserTopicFollow.topic_id == topic.id
#                     ).first() is not None
                
#                 topic_results.append({
#                     "id": topic.id,
#                     "name": topic.name,
#                     "description": topic.description,
#                     "feed_count": feed_count,
#                     "follower_count": topic.follower_count,
#                     "is_following": is_following,
#                     "created_at": topic.created_at.isoformat() if topic.created_at else None,
#                     "is_auto_discovered": False
#                 })
            
#             results["topics"] = {
#                 "items": topic_results,
#                 "total": total_topics,
#                 "has_more": (page * limit) < total_topics,
#                 "auto_discovered": False
#             }
    
#     # Search feeds - improved to find more results
#     if search_type in ["all", "feeds"]:
#         feeds_query = db.query(Feed).options(joinedload(Feed.blog)).filter(
#             Feed.status == "ready",
#             or_(
#                 Feed.title.ilike(search_query),
#                 Feed.categories.cast(String).ilike(search_query),
#                 Feed.ai_generated_content.cast(String).ilike(search_query),
#                 Feed.source_type.ilike(search_query)
#             )
#         ).order_by(Feed.created_at.desc())
        
#         total_feeds = feeds_query.count()
#         feeds = feeds_query.offset((page - 1) * limit).limit(limit).all()
        
#         feed_results = []
#         for feed in feeds:
#             # Determine source metadata
#             if feed.source_type == "youtube":
#                 meta = {
#                     "title": feed.title,
#                     "original_title": feed.title,
#                     "author": "YouTube Creator",
#                     "source_url": f"https://www.youtube.com/watch?v={feed.transcript_id}" if feed.transcript_id else "#",
#                     "source_type": "youtube"
#                 }
#             else:
#                 meta = {
#                     "title": feed.title,
#                     "original_title": feed.blog.title if feed.blog else "Unknown",
#                     "author": getattr(feed.blog, 'author', 'Admin'),
#                     "source_url": getattr(feed.blog, 'url', '#'),
#                     "source_type": "blog"
#                 }
            
#             feed_results.append({
#                 "id": feed.id,
#                 "title": feed.title,
#                 "categories": feed.categories,
#                 "source_type": feed.source_type,
#                 "slides_count": len(feed.slides),
#                 "meta": meta,
#                 "created_at": feed.created_at.isoformat() if feed.created_at else None,
#                 "ai_generated": bool(feed.ai_generated_content)
#             })
        
#         results["feeds"] = {
#             "items": feed_results,
#             "total": total_feeds,
#             "has_more": (page * limit) < total_feeds
#         }
    
#     # Search sources
#     if search_type in ["all", "sources"]:
#         sources_query = db.query(Source).filter(
#             Source.is_active == True,
#             or_(
#                 Source.name.ilike(search_query),
#                 Source.website.ilike(search_query),
#                 Source.source_type.ilike(search_query)
#             )
#         )
        
#         total_sources = sources_query.count()
#         sources = sources_query.offset((page - 1) * limit).limit(limit).all()
        
#         source_results = []
#         for source in sources:
#             # Get feed count for this source
#             if source.source_type == "blog":
#                 feed_count = db.query(Feed).join(Blog).filter(
#                     Blog.website == source.website,
#                     Feed.status == "ready"
#                 ).count()
#             else:
#                 feed_count = db.query(Feed).filter(
#                     Feed.source_type == "youtube",
#                     Feed.status == "ready"
#                 ).count()
            
#             # Check if user follows this source
#             is_following = False
#             if user_id:
#                 is_following = db.query(UserSourceFollow).filter(
#                     UserSourceFollow.user_id == user_id,
#                     UserSourceFollow.source_id == source.id
#                 ).first() is not None
            
#             source_results.append({
#                 "id": source.id,
#                 "name": source.name,
#                 "website": source.website,
#                 "source_type": source.source_type,
#                 "feed_count": feed_count,
#                 "follower_count": source.follower_count,
#                 "is_following": is_following,
#                 "created_at": source.created_at.isoformat() if source.created_at else None
#             })
        
#         results["sources"] = {
#             "items": source_results,
#             "total": total_sources,
#             "has_more": (page * limit) < total_sources
#         }
    
#     return results

# @router.get("/topics/popular", response_model=dict)
# def get_popular_topics(
#     page: int = 1,
#     limit: int = 20,
#     user_id: Optional[int] = None,
#     db: Session = Depends(get_db)
# ):
#     """Get popular topics based on feed count and follower count."""
#     # Get all active topics
#     topics_query = db.query(Topic).filter(Topic.is_active == True)
    
#     # Get topic statistics
#     topics_with_stats = []
#     for topic in topics_query.all():
#         feed_count = db.query(Feed).filter(
#             Feed.categories.contains([topic.name]),
#             Feed.status == "ready"
#         ).count()
        
#         # Only include topics with feeds
#         if feed_count > 0:
#             # Check if user follows this topic
#             is_following = False
#             if user_id:
#                 is_following = db.query(UserTopicFollow).filter(
#                     UserTopicFollow.user_id == user_id,
#                     UserTopicFollow.topic_id == topic.id
#                 ).first() is not None
            
#             topics_with_stats.append({
#                 "topic": topic,
#                 "feed_count": feed_count,
#                 "is_following": is_following
#             })
    
#     # Sort by feed count (popularity) and then by follower count
#     topics_with_stats.sort(key=lambda x: (x["feed_count"], x["topic"].follower_count), reverse=True)
    
#     # Paginate
#     start_idx = (page - 1) * limit
#     end_idx = start_idx + limit
#     paginated_topics = topics_with_stats[start_idx:end_idx]
    
#     # Format response
#     items = []
#     for item in paginated_topics:
#         topic = item["topic"]
#         items.append({
#             "id": topic.id,
#             "name": topic.name,
#             "description": topic.description,
#             "feed_count": item["feed_count"],
#             "follower_count": topic.follower_count,
#             "is_following": item["is_following"],
#             "created_at": topic.created_at.isoformat() if topic.created_at else None
#         })
    
#     total = len(topics_with_stats)
#     has_more = (page * limit) < total
    
#     return {
#         "items": items,
#         "page": page,
#         "limit": limit,
#         "total": total,
#         "has_more": has_more
#     }

# @router.post("/topics/{topic_name}/follow", response_model=dict)
# def follow_topic(
#     topic_name: str,
#     user_id: int,  # In real implementation, get from auth token
#     db: Session = Depends(get_db)
# ):
#     """Follow a topic."""
#     # Find or create topic
#     topic = db.query(Topic).filter(Topic.name == topic_name).first()
#     if not topic:
#         # Create topic if it doesn't exist
#         topic = Topic(name=topic_name, description=f"Topic for {topic_name}")
#         db.add(topic)
#         db.flush()
    
#     # Check if already following
#     existing_follow = db.query(UserTopicFollow).filter(
#         UserTopicFollow.user_id == user_id,
#         UserTopicFollow.topic_id == topic.id
#     ).first()
    
#     if existing_follow:
#         raise HTTPException(status_code=400, detail="Already following this topic")
    
#     # Create follow relationship
#     follow = UserTopicFollow(user_id=user_id, topic_id=topic.id)
#     db.add(follow)
    
#     # Update follower count
#     topic.follower_count += 1
#     topic.updated_at = datetime.utcnow()
    
#     db.commit()
    
#     return {
#         "message": f"Started following topic: {topic_name}",
#         "topic_id": topic.id,
#         "topic_name": topic.name,
#         "follower_count": topic.follower_count,
#         "is_following": True
#     }

# @router.post("/topics/{topic_name}/unfollow", response_model=dict)
# def unfollow_topic(
#     topic_name: str,
#     user_id: int,  # In real implementation, get from auth token
#     db: Session = Depends(get_db)
# ):
#     """Unfollow a topic."""
#     topic = db.query(Topic).filter(Topic.name == topic_name).first()
#     if not topic:
#         raise HTTPException(status_code=404, detail="Topic not found")
    
#     # Find follow relationship
#     follow = db.query(UserTopicFollow).filter(
#         UserTopicFollow.user_id == user_id,
#         UserTopicFollow.topic_id == topic.id
#     ).first()
    
#     if not follow:
#         raise HTTPException(status_code=400, detail="Not following this topic")
    
#     # Remove follow relationship
#     db.delete(follow)
    
#     # Update follower count
#     topic.follower_count = max(0, topic.follower_count - 1)
#     topic.updated_at = datetime.utcnow()
    
#     db.commit()
    
#     return {
#         "message": f"Stopped following topic: {topic_name}",
#         "topic_id": topic.id,
#         "topic_name": topic.name,
#         "follower_count": topic.follower_count,
#         "is_following": False
#     }

# @router.post("/sources/{source_id}/follow", response_model=dict)
# def follow_source(
#     source_id: int,
#     user_id: int,  # In real implementation, get from auth token
#     db: Session = Depends(get_db)
# ):
#     """Follow a source."""
#     source = db.query(Source).filter(Source.id == source_id).first()
#     if not source:
#         raise HTTPException(status_code=404, detail="Source not found")
    
#     # Check if already following
#     existing_follow = db.query(UserSourceFollow).filter(
#         UserSourceFollow.user_id == user_id,
#         UserSourceFollow.source_id == source_id
#     ).first()
    
#     if existing_follow:
#         raise HTTPException(status_code=400, detail="Already following this source")
    
#     # Create follow relationship
#     follow = UserSourceFollow(user_id=user_id, source_id=source_id)
#     db.add(follow)
    
#     # Update follower count
#     source.follower_count += 1
#     source.updated_at = datetime.utcnow()
    
#     db.commit()
    
#     return {
#         "message": f"Started following source: {source.name}",
#         "source_id": source.id,
#         "source_name": source.name,
#         "follower_count": source.follower_count,
#         "is_following": True
#     }

# @router.post("/sources/{source_id}/unfollow", response_model=dict)
# def unfollow_source(
#     source_id: int,
#     user_id: int,  # In real implementation, get from auth token
#     db: Session = Depends(get_db)
# ):
#     """Unfollow a source."""
#     source = db.query(Source).filter(Source.id == source_id).first()
#     if not source:
#         raise HTTPException(status_code=404, detail="Source not found")
    
#     # Find follow relationship
#     follow = db.query(UserSourceFollow).filter(
#         UserSourceFollow.user_id == user_id,
#         UserSourceFollow.source_id == source_id
#     ).first()
    
#     if not follow:
#         raise HTTPException(status_code=400, detail="Not following this source")
    
#     # Remove follow relationship
#     db.delete(follow)
    
#     # Update follower count
#     source.follower_count = max(0, source.follower_count - 1)
#     source.updated_at = datetime.utcnow()
    
#     db.commit()
    
#     return {
#         "message": f"Stopped following source: {source.name}",
#         "source_id": source.id,
#         "source_name": source.name,
#         "follower_count": source.follower_count,
#         "is_following": False
#     }

# @router.get("/user/feed", response_model=dict)
# def get_personalized_feed(
#     user_id: int,
#     page: int = 1,
#     limit: int = 20,
#     db: Session = Depends(get_db)
# ):
#     """Get personalized feed based on followed topics and sources."""
#     # Get user's followed topics
#     followed_topics = db.query(UserTopicFollow).filter(
#         UserTopicFollow.user_id == user_id
#     ).all()
    
#     # Get user's followed sources
#     followed_sources = db.query(UserSourceFollow).filter(
#         UserSourceFollow.user_id == user_id
#     ).all()
    
#     # Build query for feeds from followed topics and sources
#     feed_query = db.query(Feed).options(joinedload(Feed.blog)).filter(
#         Feed.status == "ready"
#     )
    
#     # Add conditions for followed topics
#     topic_conditions = []
#     for topic_follow in followed_topics:
#         topic = topic_follow.topic
#         topic_conditions.append(Feed.categories.contains([topic.name]))
    
#     # Add conditions for followed sources
#     source_conditions = []
#     for source_follow in followed_sources:
#         source = source_follow.source
#         if source.source_type == "blog":
#             source_conditions.append(Blog.website == source.website)
#         # Add YouTube source conditions if needed
    
#     # Combine conditions
#     if topic_conditions or source_conditions:
#         combined_conditions = []
#         if topic_conditions:
#             combined_conditions.append(or_(*topic_conditions))
#         if source_conditions:
#             # Join with Blog for website conditions
#             feed_query = feed_query.join(Blog)
#             combined_conditions.append(or_(*source_conditions))
        
#         feed_query = feed_query.filter(or_(*combined_conditions))
    
#     # Order by creation date (newest first)
#     feed_query = feed_query.order_by(Feed.created_at.desc())
    
#     # Paginate
#     total = feed_query.count()
#     feeds = feed_query.offset((page - 1) * limit).limit(limit).all()
    
#     # Format response
#     items = []
#     for feed in feeds:
#         if feed.source_type == "youtube":
#             meta = {
#                 "title": feed.title,
#                 "original_title": feed.title,
#                 "author": "YouTube Creator",
#                 "source_url": f"https://www.youtube.com/watch?v={feed.transcript_id}" if feed.transcript_id else "#",
#                 "source_type": "youtube"
#             }
#         else:
#             meta = {
#                 "title": feed.title,
#                 "original_title": feed.blog.title if feed.blog else "Unknown",
#                 "author": getattr(feed.blog, 'author', 'Admin'),
#                 "source_url": getattr(feed.blog, 'url', '#'),
#                 "source_type": "blog"
#             }
        
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
#         "items": items,
#         "page": page,
#         "limit": limit,
#         "total": total,
#         "has_more": has_more,
#         "followed_topics_count": len(followed_topics),
#         "followed_sources_count": len(followed_sources)
#     }

# # ------------------ Topic and Source Management ------------------

# def auto_create_topics_from_feeds(db: Session):
#     """Automatically create topics from feed categories."""
#     # Get all unique categories from feeds
#     all_feeds = db.query(Feed).filter(Feed.categories.isnot(None)).all()
    
#     unique_categories = set()
#     for feed in all_feeds:
#         if feed.categories:
#             unique_categories.update(feed.categories)
    
#     created_count = 0
#     for category_name in unique_categories:
#         # Skip empty or very short category names
#         if not category_name or len(category_name.strip()) < 2:
#             continue
            
#         category_name = category_name.strip()
        
#         # Check if topic already exists
#         existing_topic = db.query(Topic).filter(Topic.name == category_name).first()
#         if not existing_topic:
#             topic = Topic(
#                 name=category_name,
#                 description=f"Automatically created topic for {category_name}"
#             )
#             db.add(topic)
#             created_count += 1
    
#     if created_count > 0:
#         db.commit()
#         logger.info(f"Auto-created {created_count} topics from feed categories")
    
#     return created_count

# def auto_create_sources_from_feeds(db: Session):
#     """Automatically create sources from blogs and YouTube feeds."""
#     created_count = 0
    
#     # Create sources from blogs
#     blogs = db.query(Blog).filter(Blog.website.isnot(None)).all()
#     for blog in blogs:
#         if blog.website:
#             existing_source = db.query(Source).filter(
#                 Source.website == blog.website,
#                 Source.source_type == "blog"
#             ).first()
            
#             if not existing_source:
#                 source_name = blog.website.replace("https://", "").replace("http://", "").split("/")[0]
#                 source = Source(
#                     name=source_name,
#                     website=blog.website,
#                     source_type="blog"
#                 )
#                 db.add(source)
#                 created_count += 1
    
#     # Create sources from YouTube feeds
#     youtube_feeds = db.query(Feed).filter(
#         Feed.source_type == "youtube",
#         Feed.transcript_id.isnot(None)
#     ).all()
    
#     # You might want to extract channel names from YouTube data
#     # For now, we'll create a generic YouTube source
#     existing_youtube_source = db.query(Source).filter(
#         Source.name == "YouTube",
#         Source.source_type == "youtube"
#     ).first()
    
#     if not existing_youtube_source and youtube_feeds:
#         youtube_source = Source(
#             name="YouTube",
#             website="https://www.youtube.com",
#             source_type="youtube"
#         )
#         db.add(youtube_source)
#         created_count += 1
    
#     if created_count > 0:
#         db.commit()
#         logger.info(f"Auto-created {created_count} sources from feeds")
    
#     return created_count

# def update_topic_feed_counts(db: Session):
#     """Update feed counts for all topics."""
#     topics = db.query(Topic).filter(Topic.is_active == True).all()
    
#     for topic in topics:
#         feed_count = db.query(Feed).filter(
#             Feed.categories.contains([topic.name]),
#             Feed.status == "ready"
#         ).count()
        
#         # We're storing this dynamically in the response, but you could add a field to Topic model
#         # if you want to cache this value
#         logger.debug(f"Topic '{topic.name}' has {feed_count} feeds")
    
#     return len(topics)


# @router.post("/admin/initialize-topics", response_model=dict)
# def initialize_topics_from_feeds(db: Session = Depends(get_db)):
#     """Admin endpoint to initialize topics from existing feeds."""
#     try:
#         created_count = auto_create_topics_from_feeds(db)
#         updated_count = update_topic_feed_counts(db)
        
#         return {
#             "message": "Topics initialized successfully",
#             "topics_created": created_count,
#             "topics_updated": updated_count
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to initialize topics: {str(e)}")

# @router.post("/admin/initialize-sources", response_model=dict)
# def initialize_sources_from_feeds(db: Session = Depends(get_db)):
#     """Admin endpoint to initialize sources from existing feeds."""
#     try:
#         created_count = auto_create_sources_from_feeds(db)
        
#         return {
#             "message": "Sources initialized successfully",
#             "sources_created": created_count
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to initialize sources: {str(e)}")

# @router.get("/debug/feeds-with-categories", response_model=dict)
# def debug_feeds_with_categories(db: Session = Depends(get_db)):
#     """Debug endpoint to see what categories exist in feeds."""
#     feeds_with_categories = db.query(Feed).filter(
#         Feed.categories.isnot(None),
#         Feed.status == "ready"
#     ).all()
    
#     all_categories = set()
#     feed_data = []
    
#     for feed in feeds_with_categories:
#         if feed.categories:
#             all_categories.update(feed.categories)
#             feed_data.append({
#                 "id": feed.id,
#                 "title": feed.title,
#                 "categories": feed.categories,
#                 "source_type": feed.source_type
#             })
    
#     return {
#         "total_feeds_with_categories": len(feeds_with_categories),
#         "unique_categories": list(all_categories),
#         "feeds": feed_data
#     }