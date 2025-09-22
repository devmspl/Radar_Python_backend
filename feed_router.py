from fastapi import APIRouter, Depends, HTTPException, Response, BackgroundTasks
from sqlalchemy.orm import Session, joinedload
from datetime import datetime
import os
import re
import logging
from typing import List, Optional, Dict, Any
from database import get_db
from models import Blog, Category, Feed, Slide
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import json
router = APIRouter(prefix="/get", tags=["Feeds"])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------ AI Helper Functions ------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def categorize_blog_with_openai(blog_content: str, admin_categories: list) -> list:
    """Categorize blog content using OpenAI into admin-defined categories with retry logic."""
    try:
        # Extract first 4000 characters to avoid token limits while keeping context
        truncated_content = blog_content[:4000] + "..." if len(blog_content) > 4000 else blog_content
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are a blog categorization assistant. "
                        "Analyze the blog content and categorize it into the provided categories. "
                        "Return only the category names that best match the content as a comma-separated list. "
                        "Choose at most 3 most relevant categories. "
                        "If no category fits well, return 'Uncategorized'."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Available categories: {', '.join(admin_categories)}.\n\n"
                        f"Blog content:\n{truncated_content}\n\n"
                        "Return only the category names as a comma-separated list."
                    )
                }
            ],
            temperature=0.1,  # Lower temperature for more consistent results
            max_tokens=100
        )
        
        text = response.choices[0].message.content.strip()
        # Clean and parse the response
        text_items = [x.strip().lower() for x in re.split(r'[,;|]', text) if x.strip()]
        
        # Match with admin categories (case-insensitive)
        matched_categories = []
        for item in text_items:
            # Find exact or close matches
            for category in admin_categories:
                if (item == category.lower() or 
                    item in category.lower() or 
                    category.lower() in item):
                    matched_categories.append(category)
                    break
        
        # Remove duplicates while preserving order
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

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_feed_content_with_ai(blog_title: str, blog_content: str, categories: List[str]) -> Dict[str, Any]:
    """Generate engaging feed content using AI based on the blog content."""
    try:
        # Extract first 6000 characters to avoid token limits while keeping context
        truncated_content = blog_content[:6000] + "..." if len(blog_content) > 6000 else blog_content
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are a content summarization and presentation expert. "
                        "Create an engaging, structured feed from blog content with these sections:\n"
                        "1. An engaging title that's different from the original blog title\n"
                        "2. A concise summary (1-2 paragraphs)\n"
                        "3. 3-5 key points as bullet items\n"
                        "4. A compelling conclusion\n\n"
                        "Return your response as a JSON object with these fields:\n"
                        "- title: string (engaging feed title)\n"
                        "- summary: string (concise overview)\n"
                        "- key_points: array of strings (3-5 bullet points)\n"
                        "- conclusion: string (compelling ending)\n"
                        "Ensure the content is engaging, well-structured, and optimized for presentation."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Blog Title: {blog_title}\n"
                        f"Categories: {', '.join(categories)}\n"
                        f"Blog Content:\n{truncated_content}\n\n"
                        "Generate engaging feed content in JSON format as specified."
                    )
                }
            ],
            temperature=0.7,
            response_format={"type": "json_object"},
            max_tokens=1500
        )
        
        # Parse the JSON response
        content = response.choices[0].message.content
        return json.loads(content)
    
    except Exception as e:
        logger.error(f"OpenAI feed generation error: {e}")
        # Fallback to simple extraction
        return generate_fallback_feed_content(blog_title, blog_content)

def generate_fallback_feed_content(blog_title: str, blog_content: str) -> Dict[str, Any]:
    """Generate fallback feed content when AI generation fails."""
    # Simple extraction as fallback
    paragraphs = [p.strip() for p in blog_content.split('\n\n') if len(p.strip()) > 50]
    
    summary = paragraphs[0] if paragraphs else f"An overview of {blog_title}"
    
    # Extract key points (first sentences from paragraphs)
    key_points = []
    for i, p in enumerate(paragraphs[1:4] if len(paragraphs) > 1 else paragraphs[:3]):
        # Take first sentence or first 100 characters
        first_sentence = re.split(r'(?<=[.!?]) +', p)[0] if re.search(r'[.!?]', p) else p[:100] + "..."
        key_points.append(first_sentence)
    
    # Ensure we have at least 3 key points
    while len(key_points) < 3:
        key_points.append(f"Important aspect of {blog_title}")
    
    conclusion = paragraphs[-1] if paragraphs else f"Key insights from {blog_title}"
    
    return {
        "title": f"Summary: {blog_title}",
        "summary": summary,
        "key_points": key_points[:5],  # Max 5 key points
        "conclusion": conclusion
    }

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_slide_with_ai(slide_type: str, context: str, previous_slides: List[Dict] = None) -> Dict[str, Any]:
    """Generate a specific slide type using AI."""
    try:
        slide_prompts = {
            "title": "Create an engaging title slide with a compelling title and brief introduction.",
            "key_point": "Create a slide highlighting a key point from the content. Focus on one main idea.",
            "summary": "Create a summary slide that encapsulates the main takeaways.",
            "conclusion": "Create a compelling conclusion slide that leaves a lasting impression."
        }
        
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are a presentation design expert. Create engaging, concise slides for presentation. "
                    "Return your response as a JSON object with these fields:\n"
                    "- title: string (slide title)\n"
                    "- body: string (main content, 1-2 paragraphs)\n"
                    "- bullets: array of strings (3-5 bullet points, or null if not needed)\n"
                    "Ensure content is presentation-friendly and visually appealing."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Slide Type: {slide_type}\n"
                    f"Context: {context}\n"
                    f"Instruction: {slide_prompts.get(slide_type, 'Create an engaging slide.')}\n"
                    "Generate slide content in JSON format as specified."
                )
            }
        ]
        
        # Add previous slides for context if available
        if previous_slides:
            messages[1]["content"] += f"\nPrevious Slides: {json.dumps(previous_slides[-2:])}"  # Last 2 slides for context
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            response_format={"type": "json_object"},
            max_tokens=1000
        )
        
        # Parse the JSON response
        content = response.choices[0].message.content
        slide_data = json.loads(content)
        
        # Ensure proper structure
        return {
            "title": slide_data.get("title", f"Slide {slide_type}"),
            "body": slide_data.get("body", ""),
            "bullets": slide_data.get("bullets"),
            "background_image_url": None,
            "source_refs": [],
            "render_markdown": True
        }
    
    except Exception as e:
        logger.error(f"OpenAI slide generation error: {e}")
        # Fallback slide
        return {
            "title": f"{slide_type.replace('_', ' ').title()}",
            "body": context[:500] + "..." if len(context) > 500 else context,
            "bullets": None,
            "background_image_url": None,
            "source_refs": [],
            "render_markdown": True
        }

def generate_slides_with_ai(blog_title: str, blog_content: str, ai_generated_content: Dict[str, Any]) -> List[Dict]:
    """Generate presentation slides using AI based on the blog content."""
    slides = []
    
    # Generate title slide
    title_context = f"Blog: {blog_title}\nSummary: {ai_generated_content.get('summary', '')}"
    title_slide = generate_slide_with_ai("title", title_context)
    title_slide["order"] = 1
    slides.append(title_slide)
    
    # Generate key point slides
    key_points = ai_generated_content.get("key_points", [])
    for i, point in enumerate(key_points[:4]):  # Max 4 key point slides
        key_point_context = f"Key Point: {point}\nBlog: {blog_title}"
        key_slide = generate_slide_with_ai("key_point", key_point_context, slides)
        key_slide["order"] = len(slides) + 1
        slides.append(key_slide)
    
    # Generate summary slide
    summary_context = f"Summary: {ai_generated_content.get('summary', '')}\nKey Points: {', '.join(key_points)}"
    summary_slide = generate_slide_with_ai("summary", summary_context, slides)
    summary_slide["order"] = len(slides) + 1
    slides.append(summary_slide)
    
    # Generate conclusion slide
    conclusion_context = f"Conclusion: {ai_generated_content.get('conclusion', '')}\nBlog: {blog_title}"
    conclusion_slide = generate_slide_with_ai("conclusion", conclusion_context, slides)
    conclusion_slide["order"] = len(slides) + 1
    slides.append(conclusion_slide)
    
    return slides

def create_feed_from_blog(db: Session, blog: Blog):
    """Generate feed and slides from a blog using AI and store in DB."""
    try:
        # Get active categories
        admin_categories = [c.name for c in db.query(Category).filter(Category.is_active == True).all()]
        
        # Categorize content using AI
        categories = categorize_blog_with_openai(blog.content, admin_categories)
        
        # Generate feed content using AI
        ai_generated_content = generate_feed_content_with_ai(blog.title, blog.content, categories)
        
        # Generate slides using AI
        slides_data = generate_slides_with_ai(blog.title, blog.content, ai_generated_content)
        
        # Create feed with AI-generated title
        feed_title = ai_generated_content.get("title", blog.title)
        feed = Feed(
            blog_id=blog.id, 
            categories=categories, 
            status="ready",
            title=feed_title,  # Store AI-generated title
            ai_generated_content=ai_generated_content,  # Store all AI-generated content
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(feed)
        db.flush()  # get feed.id
        
        # Create slides
        for slide_data in slides_data:
            slide = Slide(
                feed_id=feed.id,
                order=slide_data["order"],
                title=slide_data["title"],
                body=slide_data["body"],
                bullets=slide_data.get("bullets"),
                background_image_url=slide_data.get("background_image_url"),
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

# ------------------ API Endpoints ------------------

@router.get("/feeds/all", response_model=dict)
def get_all_feeds(
    response: Response, 
    page: int = 1, 
    limit: int = 20, 
    category: Optional[str] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get all feeds summary with filtering options."""
    # Build query with filters
    query = db.query(Feed).options(joinedload(Feed.blog))
    
    if category:
        query = query.filter(Feed.categories.contains([category]))
    
    if status:
        query = query.filter(Feed.status == status)
    
    # Order and paginate
    query = query.order_by(Feed.created_at.desc())
    total = query.count()
    feeds = query.offset((page - 1) * limit).limit(limit).all()

    items = []
    for f in feeds:
        items.append({
            "id": f.id,
            "blog_id": f.blog_id,
            "title": f.title,  # AI-generated title
            "categories": f.categories,
            "status": f.status,
            "slides_count": len(f.slides),
            "meta": {
                "title": f.title,  # Use AI-generated title instead of blog title
                "original_title": f.blog.title if f.blog else "Unknown",
                "author": getattr(f.blog, 'author', 'Admin'),
                "source_url": getattr(f.blog, 'url', '#'),
            },
            "created_at": f.created_at.isoformat() if f.created_at else None,
            "updated_at": f.updated_at.isoformat() if f.updated_at else None,
            "ai_generated": hasattr(f, 'ai_generated_content') and f.ai_generated_content is not None
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
        "filters": {
            "category": category,
            "status": status
        }
    }


@router.get("/feeds/{feed_id}", response_model=dict)
def get_feed_by_id(feed_id: int, db: Session = Depends(get_db)):
    """Get full AI-generated feed with slides."""
    feed = db.query(Feed).options(joinedload(Feed.blog), joinedload(Feed.slides)).filter(Feed.id == feed_id).first()
    if not feed:
        raise HTTPException(status_code=404, detail="Feed not found")

    # Get AI-generated content if available
    ai_content = getattr(feed, 'ai_generated_content', {})
    
    return {
        "id": feed.id,
        "blog_id": feed.blog_id,
        "title": feed.title,  # AI-generated title
        "categories": feed.categories,
        "status": feed.status,
        "ai_generated_content": ai_content,  # Include all AI-generated content
        "meta": {
            "title": feed.title,  # AI-generated title
            "original_title": feed.blog.title if feed.blog else "Unknown",
            "author": getattr(feed.blog, 'author', 'Admin'),
            "source_url": getattr(feed.blog, 'url', '#'),
            # "published_at": feed.blog.created_at.isoformat() if feed.blog and feed.blog.created_at else None,
        },
        "slides": sorted([
            {
                "id": s.id,
                "order": s.order,
                "title": s.title,
                "body": s.body,
                "bullets": s.bullets,
                "background_image_url": s.background_image_url,
                "source_refs": s.source_refs,
                "render_markdown": bool(s.render_markdown),
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "updated_at": s.updated_at.isoformat() if s.updated_at else None
            } for s in feed.slides
        ], key=lambda x: x["order"]),
        "created_at": feed.created_at.isoformat() if feed.created_at else None,
        "updated_at": feed.updated_at.isoformat() if feed.updated_at else None,
        "ai_generated": hasattr(feed, 'ai_generated_content') and feed.ai_generated_content is not None
    }


@router.post("/feeds/{website}", response_model=dict)
def create_feeds_from_website(
    website: str, 
    background_tasks: BackgroundTasks,
    overwrite: bool = False,
    use_ai: bool = True,  # New parameter to control AI usage
    db: Session = Depends(get_db)
):
    """Create feeds for all blogs from a website. Optionally use AI for generation."""
    blogs = db.query(Blog).filter(Blog.website == website).all()
    if not blogs:
        raise HTTPException(status_code=404, detail="No blogs found for this website")

    # Process in background for better performance
    background_tasks.add_task(process_feeds_creation, db, blogs, website, overwrite, use_ai)
    
    return {
        "website": website,
        "total_blogs": len(blogs),
        "use_ai": use_ai,
        "message": "Feed creation process started in background",
        "status": "processing"
    }


def process_feeds_creation(db: Session, blogs: List[Blog], website: str, overwrite: bool = False, use_ai: bool = True):
    """Background task to process feed creation."""
    from database import SessionLocal
    # Create a new database session for the background task
    db = SessionLocal()
    try:
        created_count = 0
        skipped_count = 0
        error_count = 0
        
        for blog in blogs:
            try:
                # Check if feed already exists
                existing_feed = db.query(Feed).filter(Feed.blog_id == blog.id).first()
                
                if existing_feed and not overwrite:
                    skipped_count += 1
                    logger.info(f"Skipping blog {blog.id} - feed already exists")
                    continue
                
                # If exists and overwrite is True, delete existing feed
                if existing_feed and overwrite:
                    db.delete(existing_feed)
                    db.flush()
                
                # Create new feed (using AI or basic method)
                if use_ai:
                    feed = create_feed_from_blog(db, blog)
                else:
                    # Fallback to basic method if AI is disabled
                    feed = create_basic_feed_from_blog(db, blog)
                    
                if feed:
                    created_count += 1
                    logger.info(f"Created feed {feed.id} for blog {blog.id}")
                    
            except Exception as e:
                error_count += 1
                logger.error(f"Error processing blog {blog.id}: {e}")
                # Continue with next blog even if one fails
                continue
        
        logger.info(
            f"Completed feed creation for {website}: "
            f"{created_count} created, {skipped_count} skipped, {error_count} errors"
        )
        
    finally:
        db.close()


# @router.post("/feeds/blog/{blog_id}", response_model=dict)
# def create_feed_for_blog(
#     blog_id: int, 
#     overwrite: bool = False, 
#     use_ai: bool = True,
#     db: Session = Depends(get_db)
# ):
#     """Create feed for a specific blog. Optionally use AI for generation."""
#     blog = db.query(Blog).filter(Blog.id == blog_id).first()
#     if not blog:
#         raise HTTPException(status_code=404, detail="Blog not found")
    
#     # Check if feed already exists
#     existing_feed = db.query(Feed).filter(Feed.blog_id == blog_id).first()
#     if existing_feed and not overwrite:
#         raise HTTPException(
#             status_code=409, 
#             detail=f"Feed already exists for this blog. Use overwrite=true to replace it."
#         )
    
#     # If exists and overwrite is True, delete existing feed
#     if existing_feed and overwrite:
#         db.delete(existing_feed)
#         db.flush()
    
#     # Create new feed (using AI or basic method)
#     if use_ai:
#         feed = create_feed_from_blog(db, blog)
#         message = "AI-generated feed created successfully"
#     else:
#         feed = create_basic_feed_from_blog(db, blog)
#         message = "Basic feed created successfully"
    
#     return {
#         "feed_id": feed.id,
#         "blog_id": blog_id,
#         "status": "created",
#         "ai_generated": use_ai,
#         "message": message
#     }


def create_basic_feed_from_blog(db: Session, blog: Blog):
    """Fallback method to create basic feed without AI (original implementation)."""
    # This would be your original implementation for creating feeds
    # without AI enhancement - kept as a fallback option
    admin_categories = [c.name for c in db.query(Category).filter(Category.is_active == True).all()]
    categories = categorize_blog_with_openai(blog.content, admin_categories)
    
    # Simple slide generation (original implementation)
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
    
    feed = Feed(blog_id=blog.id, categories=categories, status="ready", title=blog.title)
    db.add(feed)
    db.flush()
    
    for s in slides:
        slide = Slide(
            feed_id=feed.id,
            order=s["order"],
            title=s["title"],
            body=s["body"],
            bullets=s.get("bullets"),
            background_image_url=s.get("background_image_url"),
            source_refs=s.get("source_refs", []),
            render_markdown=int(s.get("render_markdown", True))
        )
        db.add(slide)
    
    db.commit()
    db.refresh(feed)
    return feed