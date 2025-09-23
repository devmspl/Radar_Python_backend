from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from database import get_db
from models import ScrapeJob, Blog
from scrapper import scrape_any_website
import uuid
from urllib.parse import urlparse
import time

router = APIRouter(prefix="/website", tags=["web"])
from schemas import ScrapeRequest

# Helper: start scraping in background
def run_scraping_job(db: Session, url: str, category: str, job_uid: str):
    job = db.query(ScrapeJob).filter(ScrapeJob.uid == job_uid).first()
    try:
        data = scrape_any_website(url, category)
        for item in data:
            # avoid duplicates
            existing = db.query(Blog).filter(Blog.url == item["url"]).first()
            if existing:
                continue
            blog = Blog(
                website=item["website"],
                category=item["category"],
                title=item["title"],
                description=item["description"],
                content=item["content"],
                url=item["url"],
                job_uid=job_uid  # ⚡ link blog to job
            )
            db.add(blog)
        db.commit()
        job.status = "done"
    except Exception as e:
        job.status = "failed"
    finally:
        db.commit()


# 1️⃣ Start scraping job
from fastapi import BackgroundTasks


@router.post("/scrape/start")
async def start_scraping(
    request: ScrapeRequest,  # receive the body
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    job_uid = str(uuid.uuid4())
    job = ScrapeJob(
        uid=job_uid, 
        website=urlparse(request.url).netloc, 
        url=request.url, 
        status="inprocess"
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Run scraping in the background
    background_tasks.add_task(run_scraping_job, db, request.url, request.category, job_uid)

    return {
        "uid": job_uid,
        "website": urlparse(request.url).netloc,
        "status": "inprocess",
        "link": request.url
    }


# 2️⃣ Get scraping job status
@router.get("/scrape/status/{uid}")
async def get_scrape_status(uid: str, db: Session = Depends(get_db)):
    job = db.query(ScrapeJob).filter(ScrapeJob.uid == uid).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "uid": job.uid,
        "website": job.website,
        "link": job.url,
        "status": job.status
    }


# 3️⃣ Get scraped data using UID
@router.get("/scrape/data/{uid}")
async def get_scrape_data(uid: str, db: Session = Depends(get_db)):
    job = db.query(ScrapeJob).filter(ScrapeJob.uid == uid).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    blogs = db.query(Blog).filter(Blog.job_uid == uid).all()
    return {
        "uid": uid,
        "website": job.website,
        "status": job.status,
        "blogs": [
            dict(
                id=b.id,
                website=b.website,
                category=b.category,
                title=b.title,
                description=b.description,
                content=b.content,
                url=b.url
            ) for b in blogs
        ]
    }


# 4️⃣ Get all jobs
@router.get("/jobs")
async def get_all_jobs(db: Session = Depends(get_db)):
    jobs = db.query(ScrapeJob).all()
    response = []
    for job in jobs:
        items_count = db.query(Blog).filter(Blog.job_uid == job.uid).count()
        response.append({
            "job_id": job.id,
            "uid": job.uid,
            "website": job.website,
            "url": job.url,
            "status": job.status,
            "items_processed": items_count,
            "created_at": job.created_at,
            "updated_at": job.updated_at
        })
    return {"jobs": response}


# 5️⃣ Delete job by UID
@router.delete("/jobs/{uid}")
async def delete_job(uid: str, db: Session = Depends(get_db)):
    job = db.query(ScrapeJob).filter(ScrapeJob.uid == uid).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Delete associated blogs
    db.query(Blog).filter(Blog.job_uid == uid).delete()

    # Delete the job
    db.delete(job)
    db.commit()

    return {"message": f"Job {uid} and its associated blogs have been deleted."}
