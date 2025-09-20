from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session
from datetime import datetime
import random
from database import get_db
from models import Blog, Category
import openai
from openai import OpenAI

router = APIRouter(prefix="/feed", tags=["Generate feeds"])

# Use environment variable for API key
# openai_api_key = "mykey"
# client = OpenAI(api_key="mykey")

def categorize_blog_with_openai(blog_content: str, admin_categories: list) -> list:
    """Categorize blog content using OpenAI into admin-defined categories."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-search-preview",
            messages=[
                {"role": "system", "content": "You are a blog categorization assistant."},
                {
                    "role": "user",
                    "content": (
                        f"Categorize this blog content into one or more of these categories: {admin_categories}.\n"
                        "Return only the category names as a comma-separated list.\n\n"
                        f"Blog content:\n{blog_content}"
                    )
                }
            ],
            temperature=0
        )

        text = response.choices[0].message.content.strip()
        text_items = [x.strip().lower() for x in text.replace("\n", ",").split(",") if x.strip()]
        matched = [c for c in admin_categories if c.lower() in text_items]

        return matched or ["Uncategorized"]
    except Exception as e:
        print(f"OpenAI categorization error: {e}")
        return ["Uncategorized"]

def generate_slides(blog):
    """Generate slides from blog content."""
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
    return slides

@router.get("/{website}", response_model=dict)
async def get_feed(website: str, response: Response, page: int = 1, limit: int = 20, db: Session = Depends(get_db)):
    """Generate feed for a website with real categories."""
    
    blogs_query = db.query(Blog).filter(Blog.website == website)
    total = blogs_query.count()
    blogs = blogs_query.offset((page - 1) * limit).limit(limit).all()
    
    if not blogs:
        raise HTTPException(status_code=404, detail="No blogs found for this website")

    # Fetch admin-defined categories
    admin_categories = [c.name for c in db.query(Category).filter(Category.is_active == True).all()]

    items = []
    for blog in blogs:
        blog_categories = categorize_blog_with_openai(blog.content, admin_categories)
        slides = generate_slides(blog)
        items.append({
            "id": f"cs_{blog.id}",
            "type": "article",
            "source_url": getattr(blog, "url", None),
            "categories": blog_categories,
            "meta": {
                "title": blog.title,
                "author": "Admin",
                "thumbnail_url": None,
                "duration_sec": None,
                "published_at": blog.created_at.isoformat() if hasattr(blog, "created_at") else datetime.utcnow().isoformat()
            },
            "slides": slides,
            "status": "ready",
            "created_at": blog.created_at.isoformat() if hasattr(blog, "created_at") else datetime.utcnow().isoformat(),
            "updated_at": blog.updated_at.isoformat() if hasattr(blog, "updated_at") else datetime.utcnow().isoformat()
        })

    has_more = (page * limit) < total

    response.headers["X-Total-Count"] = str(total)
    response.headers["X-Page"] = str(page)
    response.headers["X-Limit"] = str(limit)
    response.headers["X-Has-More"] = str(has_more).lower()

    return {
        "items": items,
        "page": page,
        "limit": limit,
        "total": total,
        "has_more": has_more
    }

@router.get("/all", response_model=dict)
async def getAllfeed(response: Response, page: int = 1, limit: int = 20, db: Session = Depends(get_db)):
    """Generate feed for a website with real categories."""
    
    blogs_query = db.query(Blog)
    total = blogs_query.count()
    blogs = blogs_query.offset((page - 1) * limit).limit(limit).all()
    
    if not blogs:
        raise HTTPException(status_code=404, detail="No blogs found for this website")

    # Fetch admin-defined categories
    admin_categories = [c.name for c in db.query(Category).filter(Category.is_active == True).all()]

    items = []
    for blog in blogs:
        blog_categories = categorize_blog_with_openai(blog.content, admin_categories)
        slides = generate_slides(blog)
        items.append({
            "id": f"cs_{blog.id}",
            "type": "article",
            "source_url": getattr(blog, "url", None),
            "categories": blog_categories,
            "meta": {
                "title": blog.title,
                "author": "Admin",
                "thumbnail_url": None,
                "duration_sec": None,
                "published_at": blog.created_at.isoformat() if hasattr(blog, "created_at") else datetime.utcnow().isoformat()
            },
            "slides": slides,
            "status": "ready",
            "created_at": blog.created_at.isoformat() if hasattr(blog, "created_at") else datetime.utcnow().isoformat(),
            "updated_at": blog.updated_at.isoformat() if hasattr(blog, "updated_at") else datetime.utcnow().isoformat()
        })

    has_more = (page * limit) < total

    response.headers["X-Total-Count"] = str(total)
    response.headers["X-Page"] = str(page)
    response.headers["X-Limit"] = str(limit)
    response.headers["X-Has-More"] = str(has_more).lower()

    return {
        "items": items,
        "page": page,
        "limit": limit,
        "total": total,
        "has_more": has_more
    }
