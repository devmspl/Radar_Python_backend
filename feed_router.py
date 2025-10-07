from fastapi import APIRouter, Depends, HTTPException, Response, BackgroundTasks
from sqlalchemy.orm import Session, joinedload
from datetime import datetime
import os
import re
import logging
import json
from typing import List, Optional, Dict, Any
from database import get_db
from models import Blog, Category, Feed, Slide, Transcript, TranscriptJob
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from schemas import FeedRequest, DeleteSlideRequest, YouTubeFeedRequest

router = APIRouter(prefix="/get", tags=["Feeds"])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------ AI Categorization Function ------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def categorize_content_with_openai(content: str, admin_categories: list) -> list:
    """Categorize content (blog or transcript) using OpenAI into admin-defined categories."""
    try:
        truncated_content = content[:4000] + "..." if len(content) > 4000 else content
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a content categorization assistant. Analyze the content and categorize it into the provided categories. Return only the category names that best match the content as a comma-separated list. Choose at most 3 most relevant categories."
                },
                {
                    "role": "user",
                    "content": f"Available categories: {', '.join(admin_categories)}.\n\nContent:\n{truncated_content}\n\nReturn only the category names as a comma-separated list."
                }
            ],
            temperature=0.1,
            max_tokens=100
        )
        
        text = response.choices[0].message.content.strip()
        text_items = [x.strip().lower() for x in re.split(r'[,;|]', text) if x.strip()]
        
        matched_categories = []
        for item in text_items:
            for category in admin_categories:
                if (item == category.lower() or item in category.lower() or category.lower() in item):
                    matched_categories.append(category)
                    break
        
        seen = set()
        unique_categories = []
        for cat in matched_categories:
            if cat not in seen:
                seen.add(cat)
                unique_categories.append(cat)
        
        return unique_categories[:3] if unique_categories else ["Uncategorized"]
    
    except Exception as e:
        logger.error(f"OpenAI categorization error: {e}")
        return ["Uncategorized"]

# ------------------ AI Content Generation Functions ------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_feed_content_with_ai(title: str, content: str, categories: List[str], content_type: str = "blog") -> Dict[str, Any]:
    """Generate engaging feed content using AI for both blogs and transcripts."""
    try:
        truncated_content = content[:6000] + "..." if len(content) > 6000 else content
        
        system_prompt = {
            "blog": "You are a content summarization and presentation expert. Create an engaging, structured feed from blog content. Return JSON with: title, summary, key_points (array), conclusion.",
            "transcript": "You are a video content summarization expert. Create an engaging, structured feed from video transcript content. Focus on the main ideas, key takeaways, and actionable insights. Return JSON with: title, summary, key_points (array), conclusion."
        }
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": system_prompt.get(content_type, system_prompt["blog"])
                },
                {
                    "role": "user",
                    "content": f"Title: {title}\nContent Type: {content_type}\nCategories: {', '.join(categories)}\nContent:\n{truncated_content}\n\nGenerate engaging feed content in JSON format."
                }
            ],
            temperature=0.7,
            max_tokens=2000
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        logger.error(f"OpenAI feed generation error: {e}")
        return generate_fallback_feed_content(title, content, content_type)

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_slide_with_ai(slide_type: str, context: str, categories: List[str], content_type: str = "blog", previous_slides: List[Dict] = None) -> Dict[str, Any]:
    """Generate a specific slide type using AI for both blogs and transcripts."""
    try:
        content_type_context = f"This is based on {content_type} content."
        
        messages = [
            {
                "role": "system", 
                "content": "You are a presentation design expert. Create engaging, concise slides. Return JSON with: title, body, bullets (array)."
            },
            {
                "role": "user",
                "content": f"Slide Type: {slide_type}\n{content_type_context}\nContext: {context}\nCategories: {', '.join(categories)}\nGenerate slide content in JSON format."
            }
        ]
        
        if previous_slides:
            messages[1]["content"] += f"\nPrevious Slides: {json.dumps(previous_slides[-2:])}"
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=1200
        )
        content = response.choices[0].message.content
        slide_data = json.loads(content)
        
        return {
            "title": slide_data.get("title", f"Slide {slide_type}"),
            "body": slide_data.get("body", ""),
            "bullets": slide_data.get("bullets"),
            "source_refs": [],
            "render_markdown": True
        }
    except Exception as e:
        logger.error(f"OpenAI slide generation error: {e}")
        return generate_fallback_slide(slide_type, context, categories, content_type)

def generate_slides_with_ai(title: str, content: str, ai_generated_content: Dict[str, Any], categories: List[str], content_type: str = "blog") -> List[Dict]:
    """Generate presentation slides using AI without image generation."""
    slides = []
    
    # Title slide
    title_context = f"Content: {title}\nSummary: {ai_generated_content.get('summary', '')}\nType: {content_type}"
    title_slide = generate_slide_with_ai("title", title_context, categories, content_type)
    title_slide["order"] = 1
    slides.append(title_slide)
    
    # Key point slides
    key_points = ai_generated_content.get("key_points", [])
    for i, point in enumerate(key_points[:4]):
        key_point_context = f"Key Point: {point}\nContent: {title}\nType: {content_type}"
        key_slide = generate_slide_with_ai("key_point", key_point_context, categories, content_type, slides)
        key_slide["order"] = len(slides) + 1
        slides.append(key_slide)
    
    # Summary slide
    summary_context = f"Summary: {ai_generated_content.get('summary', '')}\nKey Points: {', '.join(key_points)}\nType: {content_type}"
    summary_slide = generate_slide_with_ai("summary", summary_context, categories, content_type, slides)
    summary_slide["order"] = len(slides) + 1
    slides.append(summary_slide)
    
    # Conclusion slide
    conclusion_context = f"Conclusion: {ai_generated_content.get('conclusion', '')}\nContent: {title}\nType: {content_type}"
    conclusion_slide = generate_slide_with_ai("conclusion", conclusion_context, categories, content_type, slides)
    conclusion_slide["order"] = len(slides) + 1
    slides.append(conclusion_slide)
    
    return slides

# ------------------ Core Feed Creation Functions ------------------

def create_feed_from_blog(db: Session, blog: Blog):
    """Generate feed and slides from a blog using AI and store in DB."""
    try:
        admin_categories = [c.name for c in db.query(Category).filter(Category.is_active == True).all()]
        categories = categorize_content_with_openai(blog.content, admin_categories)
        ai_generated_content = generate_feed_content_with_ai(blog.title, blog.content, categories, "blog")
        slides_data = generate_slides_with_ai(blog.title, blog.content, ai_generated_content, categories, "blog")
        
        feed_title = ai_generated_content.get("title", blog.title)
        feed = Feed(
            blog_id=blog.id, 
            title=feed_title,
            categories=categories, 
            status="ready",
            ai_generated_content=ai_generated_content,
            image_generation_enabled=False,
            source_type="blog",  # New field to track source type
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
                background_image_url=None,
                background_image_prompt=None,
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
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating AI-generated feed for blog {blog.id}: {e}")
        raise

def create_feed_from_transcript(db: Session, transcript: Transcript):
    """Generate feed and slides from a YouTube transcript using AI and store in DB."""
    try:
        admin_categories = [c.name for c in db.query(Category).filter(Category.is_active == True).all()]
        categories = categorize_content_with_openai(transcript.transcript_text, admin_categories)
        ai_generated_content = generate_feed_content_with_ai(transcript.title, transcript.transcript_text, categories, "transcript")
        slides_data = generate_slides_with_ai(transcript.title, transcript.transcript_text, ai_generated_content, categories, "transcript")
        
        feed_title = ai_generated_content.get("title", transcript.title)
        feed = Feed(
            transcript_id=transcript.transcript_id,  # Link to transcript
            title=feed_title,
            categories=categories, 
            status="ready",
            ai_generated_content=ai_generated_content,
            image_generation_enabled=False,
            source_type="youtube",  # New field to track source type
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
                background_image_url=None,
                background_image_prompt=None,
                source_refs=slide_data.get("source_refs", []),
                render_markdown=int(slide_data.get("render_markdown", True)),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.add(slide)
        
        db.commit()
        db.refresh(feed)
        logger.info(f"Successfully created AI-generated feed {feed.id} for transcript {transcript.transcript_id}")
        return feed
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating AI-generated feed for transcript {transcript.transcript_id}: {e}")
        raise

# ------------------ Helper Functions ------------------

def generate_fallback_feed_content(title: str, content: str, content_type: str = "blog") -> Dict[str, Any]:
    """Fallback content generation for both blogs and transcripts."""
    if content_type == "transcript":
        # For transcripts, split by sentences or natural breaks
        sentences = re.split(r'(?<=[.!?])\s+', content)
        summary = sentences[0] if sentences else f"A summary of {title}"
        
        key_points = []
        for i, sentence in enumerate(sentences[1:6] if len(sentences) > 1 else sentences[:5]):
            if len(sentence) > 20:  # Only use substantial sentences
                key_points.append(sentence)
        
        while len(key_points) < 3:
            key_points.append(f"Key insight from {title}")
    else:
        # For blogs, use paragraph-based approach
        paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 50]
        summary = paragraphs[0] if paragraphs else f"An overview of {title}"
        
        key_points = []
        for i, p in enumerate(paragraphs[1:4] if len(paragraphs) > 1 else paragraphs[:3]):
            first_sentence = re.split(r'(?<=[.!?]) +', p)[0] if re.search(r'[.!?]', p) else p[:100] + "..."
            key_points.append(first_sentence)
    
    while len(key_points) < 3:
        key_points.append(f"Important aspect of {title}")
    
    return {
        "title": f"Summary: {title}",
        "summary": summary,
        "key_points": key_points[:5],
        "conclusion": f"Key insights from {title}"
    }

def generate_fallback_slide(slide_type: str, context: str, categories: List[str], content_type: str = "blog") -> Dict[str, Any]:
    """Fallback slide generation."""
    content_type_label = "Video" if content_type == "transcript" else "Blog"
    return {
        "title": f"{slide_type.replace('_', ' ').title()} - {content_type_label}",
        "body": context[:500] + "..." if len(context) > 500 else context,
        "bullets": None,
        "source_refs": [],
        "render_markdown": True
    }

def create_basic_feed_from_blog(db: Session, blog: Blog):
    """Fallback method to create basic feed without AI."""
    admin_categories = [c.name for c in db.query(Category).filter(Category.is_active == True).all()]
    categories = categorize_content_with_openai(blog.content, admin_categories)
    
    slides = []
    paragraphs = blog.content.split("\n\n")[:5]
    for idx, p in enumerate(paragraphs):
        slides.append({
            "order": idx + 1,
            "title": blog.title if idx == 0 else f"Slide {idx + 1}",
            "body": p,
            "bullets": None,
            "background_image_url": None,
            "source_refs": [],
            "render_markdown": True
        })
    
    feed = Feed(
        blog_id=blog.id, 
        categories=categories, 
        status="ready", 
        title=blog.title,
        image_generation_enabled=False,
        source_type="blog"
    )
    db.add(feed)
    db.flush()
    
    for s in slides:
        slide = Slide(
            feed_id=feed.id,
            order=s["order"],
            title=s["title"],
            body=s["body"],
            bullets=s.get("bullets"),
            background_image_url=None,
            source_refs=s.get("source_refs", []),
            render_markdown=int(s.get("render_markdown", True))
        )
        db.add(slide)
    
    db.commit()
    db.refresh(feed)
    return feed

def create_basic_feed_from_transcript(db: Session, transcript: Transcript):
    """Fallback method to create basic feed from transcript without AI."""
    admin_categories = [c.name for c in db.query(Category).filter(Category.is_active == True).all()]
    categories = categorize_content_with_openai(transcript.transcript_text, admin_categories)
    
    slides = []
    # Split transcript into meaningful chunks (sentences or paragraphs)
    chunks = re.split(r'(?<=[.!?])\s+', transcript.transcript_text)
    chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 30][:5]  # Take first 5 substantial chunks
    
    for idx, chunk in enumerate(chunks):
        slides.append({
            "order": idx + 1,
            "title": transcript.title if idx == 0 else f"Key Point {idx}",
            "body": chunk,
            "bullets": None,
            "background_image_url": None,
            "source_refs": [],
            "render_markdown": True
        })
    
    feed = Feed(
        transcript_id=transcript.transcript_id,
        categories=categories, 
        status="ready", 
        title=transcript.title,
        image_generation_enabled=False,
        source_type="youtube"
    )
    db.add(feed)
    db.flush()
    
    for s in slides:
        slide = Slide(
            feed_id=feed.id,
            order=s["order"],
            title=s["title"],
            body=s["body"],
            bullets=s.get("bullets"),
            background_image_url=None,
            source_refs=s.get("source_refs", []),
            render_markdown=int(s.get("render_markdown", True))
        )
        db.add(slide)
    
    db.commit()
    db.refresh(feed)
    return feed

# ------------------ Background Processing Functions ------------------

def process_blog_feeds_creation(db: Session, blogs: List[Blog], website: str, overwrite: bool = False, use_ai: bool = True):
    """Background task to process blog feed creation."""
    from database import SessionLocal
    db = SessionLocal()
    try:
        created_count = 0
        skipped_count = 0
        error_count = 0
        
        for blog in blogs:
            try:
                existing_feed = db.query(Feed).filter(Feed.blog_id == blog.id).first()
                if existing_feed and not overwrite:
                    skipped_count += 1
                    continue
                
                if existing_feed and overwrite:
                    db.delete(existing_feed)
                    db.flush()
                
                if use_ai:
                    feed = create_feed_from_blog(db, blog)
                else:
                    feed = create_basic_feed_from_blog(db, blog)
                
                if feed:
                    created_count += 1
            except Exception as e:
                error_count += 1
                logger.error(f"Error processing blog {blog.id}: {e}")
                continue
        
        logger.info(f"Completed blog feed creation for {website}: {created_count} created, {skipped_count} skipped, {error_count} errors")
    finally:
        db.close()

def process_transcript_feeds_creation(db: Session, transcripts: List[Transcript], job_id: str, overwrite: bool = False, use_ai: bool = True):
    """Background task to process transcript feed creation."""
    from database import SessionLocal
    db = SessionLocal()
    try:
        created_count = 0
        skipped_count = 0
        error_count = 0
        
        for transcript in transcripts:
            try:
                existing_feed = db.query(Feed).filter(Feed.transcript_id == transcript.transcript_id).first()
                if existing_feed and not overwrite:
                    skipped_count += 1
                    continue
                
                if existing_feed and overwrite:
                    db.delete(existing_feed)
                    db.flush()
                
                if use_ai:
                    feed = create_feed_from_transcript(db, transcript)
                else:
                    feed = create_basic_feed_from_transcript(db, transcript)
                
                if feed:
                    created_count += 1
            except Exception as e:
                error_count += 1
                logger.error(f"Error processing transcript {transcript.transcript_id}: {e}")
                continue
        
        logger.info(f"Completed transcript feed creation for job {job_id}: {created_count} created, {skipped_count} skipped, {error_count} errors")
    finally:
        db.close()

# ------------------ API Endpoints ------------------

@router.get("/feeds/all", response_model=dict)
def get_all_feeds(
    response: Response, 
    page: int = 1, 
    limit: int = 20, 
    category: Optional[str] = None,
    status: Optional[str] = None,
    source_type: Optional[str] = None,  # New filter: blog, youtube, or both
    db: Session = Depends(get_db)
):
    """Get all feeds summary with filtering options."""
    query = db.query(Feed).options(joinedload(Feed.blog))
    
    if category:
        query = query.filter(Feed.categories.contains([category]))
    if status:
        query = query.filter(Feed.status == status)
    if source_type:
        query = query.filter(Feed.source_type == source_type)
    
    query = query.order_by(Feed.created_at.desc())
    total = query.count()
    feeds = query.offset((page - 1) * limit).limit(limit).all()

    items = []
    for f in feeds:
        # Determine source metadata based on source_type
        if f.source_type == "youtube":
            meta = {
                "title": f.title,
                "original_title": f.title,  # Transcript title is the original
                "author": "YouTube Creator",
                "source_url": f"https://www.youtube.com/watch?v={f.transcript_id}" if f.transcript_id else "#",
                "source_type": "youtube"
            }
        else:  # blog
            meta = {
                "title": f.title,
                "original_title": f.blog.title if f.blog else "Unknown",
                "author": getattr(f.blog, 'author', 'Admin'),
                "source_url": getattr(f.blog, 'url', '#'),
                "source_type": "blog"
            }

        items.append({
            "id": f.id,
            "blog_id": f.blog_id,
            "transcript_id": f.transcript_id,
            "title": f.title,
            "categories": f.categories,
            "status": f.status,
            "source_type": f.source_type or "blog",  # Default to blog for backward compatibility
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
        "items": items,
        "page": page, 
        "limit": limit, 
        "total": total, 
        "has_more": has_more,
        "filters": {"category": category, "status": status, "source_type": source_type}
    }

@router.get("/feeds/{feed_id}", response_model=dict)
def get_feed_by_id(feed_id: int, db: Session = Depends(get_db)):
    """Get full AI-generated feed with slides."""
    feed = db.query(Feed).options(joinedload(Feed.blog), joinedload(Feed.slides)).filter(Feed.id == feed_id).first()
    if not feed:
        raise HTTPException(status_code=404, detail="Feed not found")

    ai_content = getattr(feed, 'ai_generated_content', {})
    
    # Determine source metadata based on source_type
    if feed.source_type == "youtube":
        meta = {
            "title": feed.title,
            "original_title": feed.title,
            "author": "YouTube Creator",
            "source_url": f"https://www.youtube.com/watch?v={feed.transcript_id}" if feed.transcript_id else "#",
            "source_type": "youtube"
        }
    else:
        meta = {
            "title": feed.title,
            "original_title": feed.blog.title if feed.blog else "Unknown",
            "author": getattr(feed.blog, 'author', 'Admin'),
            "source_url": getattr(feed.blog, 'url', '#'),
            "source_type": "blog"
        }
    
    return {
        "id": feed.id,
        "blog_id": feed.blog_id,
        "transcript_id": feed.transcript_id,
        "title": feed.title,
        "categories": feed.categories,
        "status": feed.status,
        "source_type": feed.source_type or "blog",
        "ai_generated_content": ai_content,
        "meta": meta,
        "slides": sorted([
            {
                "id": s.id,
                "order": s.order,
                "title": s.title,
                "body": s.body,
                "bullets": s.bullets,
                "background_image_url": None,
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
    """Create feeds for all blogs from a website (AI data only, no images)."""
    blogs = db.query(Blog).filter(Blog.website == request.website).all()
    if not blogs:
        raise HTTPException(status_code=404, detail="No blogs found for this website")

    background_tasks.add_task(
        process_blog_feeds_creation,
        db,
        blogs,
        request.website,
        request.overwrite,
        request.use_ai,
    )

    return {
        "website": request.website,
        "total_blogs": len(blogs),
        "use_ai": request.use_ai,
        "generate_images": False,
        "source_type": "blog",
        "message": "Blog feed creation process started in background",
        "status": "processing"
    }

@router.post("/feeds/youtube", response_model=dict)
def create_feeds_from_youtube(
    request: YouTubeFeedRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Create feeds from YouTube transcripts (AI data only, no images)."""
    # Get transcripts based on the request type
    if request.job_id:
        # Get transcripts from a specific job
        transcripts = db.query(Transcript).filter(Transcript.job_id == request.job_id).all()
        if not transcripts:
            raise HTTPException(status_code=404, detail="No transcripts found for this job")
    elif request.video_id:
        # Get specific video transcript
        transcripts = db.query(Transcript).filter(Transcript.video_id == request.video_id).all()
        if not transcripts:
            raise HTTPException(status_code=404, detail="No transcript found for this video")
    else:
        # Get all available transcripts
        transcripts = db.query(Transcript).all()
        if not transcripts:
            raise HTTPException(status_code=404, detail="No transcripts found in database")

    background_tasks.add_task(
        process_transcript_feeds_creation,
        db,
        transcripts,
        request.job_id or "all_transcripts",
        request.overwrite,
        request.use_ai,
    )

    return {
        "job_id": request.job_id,
        "video_id": request.video_id,
        "total_transcripts": len(transcripts),
        "use_ai": request.use_ai,
        "generate_images": False,
        "source_type": "youtube",
        "message": "YouTube transcript feed creation process started in background",
        "status": "processing"
    }

# Add the YouTubeFeedRequest schema to your schemas.py
"""
class YouTubeFeedRequest(BaseModel):
    job_id: Optional[str] = None  # Create feeds from a specific transcript job
    video_id: Optional[str] = None  # Create feed from a specific video
    overwrite: bool = False  # Overwrite existing feeds
    use_ai: bool = True  # Use AI for enhanced content generation
"""

# Also update your models.py to add the new fields to Feed model:
"""
class Feed(Base):
    # ... existing fields ...
    transcript_id = Column(String, ForeignKey('transcripts.transcript_id'), nullable=True)  # Link to YouTube transcripts
    source_type = Column(String, default='blog')  # 'blog' or 'youtube'
"""

# Keep all your existing endpoints (delete_slide_from_feed, get_categorized_feeds_by_source, etc.)
# ... [rest of your existing endpoints remain unchanged] ...

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

@router.get("/feeds/source/{website}/categorized", response_model=dict)
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
            items.append({
                "id": f.id,
                "blog_id": f.blog_id,
                "title": f.title,
                "categories": f.categories,
                "status": f.status,
                "slides_count": len(f.slides),
                "meta": {
                    "title": f.title,
                    "original_title": f.blog.title if f.blog else "Unknown",
                    "author": getattr(f.blog, 'author', 'Admin'),
                    "source_url": getattr(f.blog, 'url', '#'),
                    "website": website
                },
                "created_at": f.created_at.isoformat() if f.created_at else None,
                "updated_at": f.updated_at.isoformat() if f.updated_at else None,
                "ai_generated": hasattr(f, 'ai_generated_content') and f.ai_generated_content is not None,
                "images_generated": False  # Always false now
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