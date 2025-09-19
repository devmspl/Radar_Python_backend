# feed_router.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
import random
import openai

from database import get_db
from models import Blog, Category

router = APIRouter(prefix="/feed", tags=["feed"])

openai.api_key = "YOUR_OPENAI_KEY"  # Use environment variable preferably

def categorize_blog_with_openai(blog_content: str, admin_categories: list) -> list:
    """
    Categorize blog using OpenAI into admin-defined categories.
    Returns a list of category names.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a blog categorization assistant."},
                {
                    "role": "user",
                    "content": (
                        f"Categorize the following blog content into one or more of these categories: {admin_categories}.\n"
                        "Return only the category names as a comma-separated list.\n\n"
                        f"Blog content:\n{blog_content}"
                    )
                }
            ],
            temperature=0
        )
        text = response.choices[0].message.content.strip()

        # Normalize and match with admin categories
        text_items = [x.strip().lower() for x in text.replace("\n", ",").split(",") if x.strip()]
        matched = [c for c in admin_categories if c.lower() in text_items]

        return matched or ["Uncategorized"]
    except Exception as e:
        print(f"OpenAI categorization error: {e}")
        return ["Uncategorized"]

def generate_slides(blog: Blog) -> list:
    """Generate slides from blog content."""
    slides = []
    paragraphs = blog.content.split("\n\n")[:4]
    for idx, p in enumerate(paragraphs):
        slides.append({
            "order": idx + 1,
            "title": blog.title if idx == 0 else f"Slide {idx + 1}",
            "body": p,
            "bullets": None,
            "background_image_url": None,
            "background_color": "#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6)]),
            "text_color": "#FFFFFF",
            "render_markdown": True,
            "hasBookmarked": False,
            "slide_type": "intro" if idx == 0 else "content",
            "animation_type": "fade-in",
            "duration": 4000 + idx*500,
            "notes": f"Slide generated from blog {blog.id}"
        })
    return slides

@router.get("/{website}", response_model=dict)
async def get_feed(website: str, db: Session = Depends(get_db)):
    """Generate feed for a website with OpenAI-based admin category assignment."""
    
    blogs = db.query(Blog).filter(Blog.website == website).all()
    if not blogs:
        raise HTTPException(status_code=404, detail="No blogs found for this website")

    # Fetch all admin-defined categories
    admin_categories = [c.name for c in db.query(Category).filter(Category.is_active == True).all()]

    items = []
    for blog in blogs:
        blog_categories = categorize_blog_with_openai(blog.content, admin_categories)
        slides = generate_slides(blog)

        items.append({
            "id": f"cs_{blog.id}",
            "type": "Topics",
            "categories": blog_categories,
            "background_color": "#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6)]),
            "hasArticles": None,
            "created_at": blog.created_at.isoformat() if hasattr(blog, 'created_at') else datetime.utcnow().isoformat(),
            "updated_at": blog.updated_at.isoformat() if hasattr(blog, 'updated_at') else datetime.utcnow().isoformat(),
            "status": "published",
            "views_count": random.randint(100, 2000),
            "likes_count": random.randint(10, 500),
            "shares_count": random.randint(0, 100),
            "reading_time": f"{max(1, len(blog.content.split()) // 200)} min",
            "difficulty_level": random.choice(["beginner", "intermediate", "advanced"]),
            "tags": blog.category.split(",") if blog.category else [],
            "meta": {
                "author": "Admin",
                "thumbnail_url": "",
                "designation": "",
                "author_id": f"auth_{blog.id}",
                "author_bio": "",
                "author_social": {},
                "company": "",
                "location": ""
            },
            "slides": slides
        })

    return {"items": items}
