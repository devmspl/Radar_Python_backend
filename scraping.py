from fastapi import APIRouter, HTTPException, Depends
from enum import Enum
from typing import List
from sqlalchemy.orm import Session

from scrapper import scrape_website, save_blogs_to_db
from schemas import BlogListResponse
from database import get_db  # ✅ use correct dependency
from models import Blog  # ✅ import your Blog model

router = APIRouter(prefix="/website", tags=["web"])


class Website(str, Enum):
    outreach = "outreach"
    xactly = "xactly"
    gong = "gong"
    ziphq = "ziphq"


# @router.get("/scrape", response_model=BlogListResponse)
# async def scrape_site(website: Website):
#     """
#     Scrape all resources for the selected website (not saved in DB).
#     """
#     try:
#         blogs = scrape_website(website.value)
#         return {"blogs": blogs}
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))


@router.post("/scrape/save/{website}")
async def scrape_and_save(website: Website):
    """
    Scrape and save all blogs/resources from a website into DB.
    """
    try:
        save_blogs_to_db(website.value)  # ✅ save directly using your existing scrapper function
        return {"message": f"Scraping completed and stored for {website.value}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/blogs/{website}")
async def get_blogs(website: Website, db: Session = Depends(get_db)):
    """
    Get all blogs/resources stored for a given website.
    """
    blogs = db.query(Blog).filter(Blog.website == website.value).all()
    return {"blogs": [
        dict(
            id=b.id,
            website=b.website,
            category=b.category,
            title=b.title,
            description=b.description,
            content=b.content,
            url=b.url
        ) for b in blogs
    ]}


@router.get("/blogs/{website}/{category}")
async def get_blogs_by_category(website: Website, category: str, db: Session = Depends(get_db)):
    """
    Get all blogs/resources stored for a given website and category.
    """
    blogs = db.query(Blog).filter(
        Blog.website == website.value,
        Blog.category == category
    ).all()
    return {"blogs": [
        dict(
            id=b.id,
            website=b.website,
            category=b.category,
            title=b.title,
            description=b.description,
            content=b.content,
            url=b.url
        ) for b in blogs
    ]}
