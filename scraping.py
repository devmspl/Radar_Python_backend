from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from database import get_db
from models import ScrapeJob, Blog
from scrapper import scrape_any_website
import uuid
from urllib.parse import urlparse
import time
from fastapi import UploadFile, File, Form
import csv
import io
from typing import List
from urllib.parse import urlparse
router = APIRouter(prefix="/website", tags=["web"])
from schemas import ScrapeRequest
import schemas
# Helper: start scraping in background
# In your scrapper.py

def run_scraping_job(db: Session, url: str, category: str, job_uid: str, generate_feed: bool = False):
    """Run scraping job with optional feed generation"""
    from feed_router import auto_generate_blog_feeds  # Import from your feed router
    
    job = db.query(ScrapeJob).filter(ScrapeJob.uid == job_uid).first()
    try:
        data = scrape_any_website(url, category)
        blog_ids = []
        
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
                job_uid=job_uid,
                generate_feed=generate_feed  # Set the flag
            )
            db.add(blog)
            db.flush()  # Get the ID
            blog_ids.append(blog.id)
        
        db.commit()
        job.status = "done"
        
        # Auto-generate feeds if requested
        if generate_feed and blog_ids:
            from feed_router import auto_generate_blog_feeds
            # Start feed generation in background
            import threading
            threading.Thread(
                target=auto_generate_blog_feeds,
                args=(db, blog_ids),
                daemon=True
            ).start()
            print(f"🚀 Started auto-feed generation for {len(blog_ids)} blogs")
        
    except Exception as e:
        job.status = "failed"
        print(f"❌ Scraping job failed: {e}")
    finally:
        db.commit()


# 1️⃣ Start scraping job
from fastapi import BackgroundTasks

@router.post("/batch-scrape", response_model=schemas.BatchScrapeResponse)
async def batch_scrape_content(
    file: UploadFile = File(..., description="CSV file with 'name' and 'url' columns"),
    generate_feed: bool = Form(False, description="Generate RSS feed for the scraped content"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    """
    Batch scrape websites from CSV file
    CSV should have columns 'name' and 'url' containing content names and website URLs
    """
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are allowed"
        )
    
    try:
        # Read and parse CSV file
        content = await file.read()
        csv_content = io.StringIO(content.decode('utf-8'))
        csv_reader = csv.DictReader(csv_content)
        
        # Check if required columns exist
        required_columns = ['name', 'url']
        missing_columns = [col for col in required_columns if col not in csv_reader.fieldnames]
        
        if missing_columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"CSV must contain columns: {', '.join(required_columns)}. Missing: {', '.join(missing_columns)}"
            )
        
        # Read all rows
        rows = []
        for row in csv_reader:
            if row.get('url') and row.get('name'):
                rows.append({
                    'name': row['name'].strip(),
                    'url': row['url'].strip()
                })
        
        if not rows:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid rows found in CSV file (must have both name and URL)"
            )
        
        # Process each row and create scraping jobs
        job_responses = []
        failed_rows = []
        
        for row in rows:
            try:
                # Create individual scraping job for each URL
                job_uid = str(uuid.uuid4())
                job = ScrapeJob(
                    uid=job_uid, 
                    website=urlparse(row['url']).netloc, 
                    url=row['url'], 
                    status="inprocess"
                )
                db.add(job)
                db.commit()
                db.refresh(job)

                # Add to background tasks - category is now handled by the scraper
                background_tasks.add_task(
                    run_scraping_job, 
                    db, 
                    row['url'], 
                    "",  # Empty category since it's removed
                    job_uid,
                    generate_feed  # Pass the flag
                )

                job_responses.append({
                    'job_uid': job_uid,
                    'name': row['name'],
                    'url': row['url'],
                    'generate_feed': generate_feed
                })
                
            except Exception as e:
                failed_rows.append({
                    'name': row['name'],
                    'url': row['url'],
                    'error': str(e)
                })
        
        return schemas.BatchScrapeResponse(
            message=f"Started {len(job_responses)} scraping jobs successfully, {len(failed_rows)} failed",
            total_rows=len(rows),
            successful_jobs=len(job_responses),
            failed_rows=failed_rows,
            job_responses=job_responses
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing CSV file: {str(e)}"
        )
@router.post("/scrape/start")
async def start_scraping(
    request: ScrapeRequest,
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

    # Run scraping in the background with feed generation option
    background_tasks.add_task(
        run_scraping_job, 
        db, 
        request.url, 
        request.category, 
        job_uid,
        request.generate_feed  # Pass the flag
    )

    return {
        "uid": job_uid,
        "website": urlparse(request.url).netloc,
        "status": "inprocess",
        "link": request.url,
        "generate_feed": request.generate_feed,
        "message": "Scraping started" + (" with auto-feed generation" if request.generate_feed else "")
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
