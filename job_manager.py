# job_manager.py
"""
Centralized Job Manager for controlling all background tasks and scheduled jobs.
Provides APIs to:
- Stop all running jobs
- View job status
- Pause/Resume specific job types
- Kill long-running tasks
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from enum import Enum
import threading
import logging
import os

from database import get_db
from dependencies import get_current_admin
import models


# ==================== Pydantic Response Models for Swagger ====================

class JobTypeEnum(str, Enum):
    feed_generation = "feed_generation"
    youtube_transcription = "youtube_transcription"
    blog_scraping = "blog_scraping"
    quiz_generation = "quiz_generation"
    all = "all"


class RunningJobInfo(BaseModel):
    """Information about a running job"""
    job_id: str = Field(..., description="Unique job identifier")
    type: str = Field(..., description="Type of job (feed_generation, youtube_transcription, etc.)")
    description: str = Field("", description="Job description")
    started_at: str = Field(..., description="ISO timestamp when job started")
    status: str = Field(..., description="Current status (running, stopped, completed)")
    running_for_seconds: float = Field(..., description="How long the job has been running")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "feed_gen_12345",
                "type": "feed_generation",
                "description": "Processing blog: How to Use AI",
                "started_at": "2026-01-02T12:30:00",
                "status": "running",
                "running_for_seconds": 45.5
            }
        }


class JobCountsByType(BaseModel):
    """Count of jobs by type"""
    feed_generation: int = 0
    youtube_transcription: int = 0
    blog_scraping: int = 0
    quiz_generation: int = 0
    other: int = 0


class JobManagerStatus(BaseModel):
    """Current status of the job manager"""
    manager_started_at: str
    stop_all_flag: bool
    paused_types: Dict[str, bool]
    running_jobs_count: int
    job_counts_by_type: JobCountsByType
    running_jobs: List[RunningJobInfo]


class SchedulerJobInfo(BaseModel):
    """APScheduler job information"""
    id: str
    name: Optional[str] = None
    next_run_time: Optional[str] = None


class APSchedulerStatus(BaseModel):
    """Status of APScheduler"""
    running: bool = False
    jobs: List[SchedulerJobInfo] = []
    error: Optional[str] = None


class StopAllResponse(BaseModel):
    """Response for stop-all endpoint"""
    success: bool = True
    timestamp: str
    initiated_by: str
    message: str
    jobs_affected: int
    scheduler_paused: bool

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "timestamp": "2026-01-02T12:30:00",
                "initiated_by": "admin@example.com",
                "message": "Stop signal sent to all jobs",
                "jobs_affected": 5,
                "scheduler_paused": True
            }
        }


class ResumeAllResponse(BaseModel):
    """Response for resume-all endpoint"""
    success: bool = True
    timestamp: str
    initiated_by: str
    message: str
    scheduler_resumed: bool


class JobStatusResponse(BaseModel):
    """Response for job status endpoint"""
    success: bool = True
    timestamp: str
    job_manager: Dict[str, Any]
    apscheduler: Dict[str, Any]


class PauseResumeResponse(BaseModel):
    """Response for pause/resume endpoints"""
    success: bool = True
    timestamp: str
    initiated_by: str
    message: str


class ClearStateResponse(BaseModel):
    """Response for clear-state endpoint"""
    success: bool = True
    timestamp: str
    initiated_by: str
    message: str


class RunningJobsResponse(BaseModel):
    """Response for running jobs list"""
    success: bool = True
    timestamp: str
    total_running: int
    jobs_by_type: Dict[str, int]
    jobs: List[Dict[str, Any]]


class TranscriptionJobInfo(BaseModel):
    """Transcription job from database"""
    id: int
    job_id: str
    url: str
    type: Optional[str]
    status: Optional[str]
    total_items: int
    processed_items: int
    created_at: Optional[str]
    completed_at: Optional[str]


class TranscriptionJobsResponse(BaseModel):
    """Response for transcription jobs list"""
    success: bool = True
    total: int
    jobs: List[TranscriptionJobInfo]


class ScrapeJobInfo(BaseModel):
    """Scrape job from database"""
    id: int
    uid: str
    website: str
    url: str
    status: str
    created_at: Optional[str]
    updated_at: Optional[str]


class ScrapeJobsResponse(BaseModel):
    """Response for scrape jobs list"""
    success: bool = True
    total: int
    jobs: List[ScrapeJobInfo]


class CancelProcessingResponse(BaseModel):
    """Response for cancelling processing jobs"""
    success: bool = True
    timestamp: str
    transcription_jobs_cancelled: int
    scrape_jobs_cancelled: int
    message: str


# Router with enhanced tags
router = APIRouter(
    prefix="/jobs", 
    tags=["🛠️ Job Management"],
    responses={
        401: {"description": "Unauthorized - Admin access required"},
        403: {"description": "Forbidden - Not an admin user"}
    }
)
logger = logging.getLogger(__name__)

# ==================== Global Job State Management ====================

class JobManager:
    """Central manager for all background jobs and tasks."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize job manager state."""
        self.running_jobs: Dict[str, Dict[str, Any]] = {}
        self.job_counts: Dict[str, int] = {
            "feed_generation": 0,
            "youtube_transcription": 0,
            "blog_scraping": 0,
            "quiz_generation": 0,
            "other": 0
        }
        self.is_paused: Dict[str, bool] = {
            "feed_generation": False,
            "youtube_transcription": False,
            "blog_scraping": False,
            "quiz_generation": False,
            "all": False
        }
        self.stop_all_flag = False
        self.scheduler = None
        self._lock = threading.Lock()
        self.started_at = datetime.utcnow()
    
    def register_job(self, job_id: str, job_type: str, description: str = "") -> bool:
        """Register a new job. Returns False if jobs are stopped/paused."""
        with self._lock:
            if self.stop_all_flag:
                logger.info(f"⛔ Job {job_id} blocked - all jobs are stopped")
                return False
            
            if self.is_paused.get("all", False):
                logger.info(f"⏸️ Job {job_id} blocked - all jobs are paused")
                return False
            
            if self.is_paused.get(job_type, False):
                logger.info(f"⏸️ Job {job_id} blocked - {job_type} jobs are paused")
                return False
            
            self.running_jobs[job_id] = {
                "job_id": job_id,
                "type": job_type,
                "description": description,
                "started_at": datetime.utcnow(),
                "status": "running"
            }
            self.job_counts[job_type] = self.job_counts.get(job_type, 0) + 1
            logger.info(f"✅ Registered job: {job_id} ({job_type})")
            return True
    
    def complete_job(self, job_id: str, status: str = "completed"):
        """Mark a job as completed."""
        with self._lock:
            if job_id in self.running_jobs:
                job = self.running_jobs[job_id]
                job_type = job.get("type", "other")
                self.job_counts[job_type] = max(0, self.job_counts.get(job_type, 1) - 1)
                job["status"] = status
                job["completed_at"] = datetime.utcnow()
                del self.running_jobs[job_id]
                logger.info(f"✅ Completed job: {job_id} ({status})")
    
    def should_stop(self) -> bool:
        """Check if jobs should stop immediately."""
        return self.stop_all_flag
    
    def is_job_type_paused(self, job_type: str) -> bool:
        """Check if a specific job type is paused."""
        return self.is_paused.get("all", False) or self.is_paused.get(job_type, False)
    
    def stop_all_jobs(self) -> Dict[str, Any]:
        """Signal all jobs to stop."""
        with self._lock:
            self.stop_all_flag = True
            stopped_count = len(self.running_jobs)
            
            # Stop APScheduler if available
            scheduler_stopped = False
            try:
                from quiz_router import scheduler as quiz_scheduler
                if quiz_scheduler and quiz_scheduler.running:
                    quiz_scheduler.pause()
                    scheduler_stopped = True
                    logger.info("⏸️ Quiz scheduler paused")
            except Exception as e:
                logger.error(f"Error pausing scheduler: {e}")
            
            # Mark all running jobs as stopped
            for job_id in list(self.running_jobs.keys()):
                self.running_jobs[job_id]["status"] = "stopped"
            
            logger.warning(f"🛑 STOP ALL JOBS signal sent! {stopped_count} jobs affected")
            
            return {
                "message": "Stop signal sent to all jobs",
                "jobs_affected": stopped_count,
                "scheduler_paused": scheduler_stopped
            }
    
    def resume_all_jobs(self) -> Dict[str, Any]:
        """Resume all job processing."""
        with self._lock:
            self.stop_all_flag = False
            self.is_paused = {k: False for k in self.is_paused}
            
            # Resume APScheduler if available
            scheduler_resumed = False
            try:
                from quiz_router import scheduler as quiz_scheduler
                if quiz_scheduler:
                    quiz_scheduler.resume()
                    scheduler_resumed = True
                    logger.info("▶️ Quiz scheduler resumed")
            except Exception as e:
                logger.error(f"Error resuming scheduler: {e}")
            
            logger.info("▶️ All jobs resumed")
            
            return {
                "message": "All jobs resumed",
                "scheduler_resumed": scheduler_resumed
            }
    
    def pause_job_type(self, job_type: str) -> Dict[str, Any]:
        """Pause a specific type of jobs."""
        with self._lock:
            if job_type in self.is_paused:
                self.is_paused[job_type] = True
                logger.info(f"⏸️ Paused job type: {job_type}")
                return {"message": f"Paused {job_type} jobs"}
            else:
                return {"error": f"Unknown job type: {job_type}"}
    
    def resume_job_type(self, job_type: str) -> Dict[str, Any]:
        """Resume a specific type of jobs."""
        with self._lock:
            if job_type in self.is_paused:
                self.is_paused[job_type] = False
                logger.info(f"▶️ Resumed job type: {job_type}")
                return {"message": f"Resumed {job_type} jobs"}
            else:
                return {"error": f"Unknown job type: {job_type}"}
    
    def get_status(self) -> Dict[str, Any]:
        """Get current job manager status."""
        with self._lock:
            return {
                "manager_started_at": self.started_at.isoformat(),
                "stop_all_flag": self.stop_all_flag,
                "paused_types": {k: v for k, v in self.is_paused.items() if v},
                "running_jobs_count": len(self.running_jobs),
                "job_counts_by_type": self.job_counts.copy(),
                "running_jobs": [
                    {
                        "job_id": job["job_id"],
                        "type": job["type"],
                        "description": job["description"],
                        "started_at": job["started_at"].isoformat(),
                        "status": job["status"],
                        "running_for_seconds": (datetime.utcnow() - job["started_at"]).total_seconds()
                    }
                    for job in self.running_jobs.values()
                ]
            }
    
    def clear_stopped_state(self) -> Dict[str, Any]:
        """Clear the stop flag and all paused states."""
        with self._lock:
            self.stop_all_flag = False
            self.is_paused = {k: False for k in self.is_paused}
            self.running_jobs.clear()
            self.job_counts = {k: 0 for k in self.job_counts}
            logger.info("🧹 Job manager state cleared")
            return {"message": "Job manager state cleared"}


# Global job manager instance
job_manager = JobManager()


# ==================== Helper Function for Other Modules ====================

def should_continue_job(job_id: str = None, job_type: str = None) -> bool:
    """
    Helper function to check if a job should continue executing.
    Call this in loops within long-running tasks.
    
    Usage in other modules:
        from job_manager import should_continue_job, job_manager
        
        # At start of job:
        if not job_manager.register_job(job_id, "feed_generation", "Processing blog XYZ"):
            return  # Job was blocked
        
        try:
            for item in items:
                if not should_continue_job():
                    logger.info("Job stopped by manager")
                    return
                # ... process item
        finally:
            job_manager.complete_job(job_id)
    """
    if job_manager.should_stop():
        return False
    if job_type and job_manager.is_job_type_paused(job_type):
        return False
    return True


# ==================== API Endpoints ====================

@router.post(
    "/stop-all", 
    summary="🛑 Stop All Running Jobs",
    response_model=StopAllResponse,
    description="Emergency endpoint to stop all running background jobs immediately. Requires admin authentication."
)
async def stop_all_jobs(
    current_user: models.User = Depends(get_current_admin)
):
    """
    **EMERGENCY: Stop all running background jobs immediately.**
    
    This will:
    - Set stop flag for all running background tasks
    - Pause the APScheduler (quiz updates)
    - Block new jobs from starting
    
    Running jobs will stop at their next checkpoint.
    """
    result = job_manager.stop_all_jobs()
    logger.warning(f"🛑 Admin {current_user.email} stopped all jobs")
    return {
        "success": True,
        "timestamp": datetime.utcnow().isoformat(),
        "initiated_by": current_user.email,
        **result
    }


@router.post(
    "/resume-all", 
    summary="▶️ Resume All Jobs",
    response_model=ResumeAllResponse,
    description="Resume all job processing after a stop. Allows new jobs to start."
)
async def resume_all_jobs(
    current_user: models.User = Depends(get_current_admin)
):
    """
    **Resume all job processing after a stop.**
    
    This will:
    - Clear the stop flag
    - Resume the APScheduler
    - Allow new jobs to start
    """
    result = job_manager.resume_all_jobs()
    logger.info(f"▶️ Admin {current_user.email} resumed all jobs")
    return {
        "success": True,
        "timestamp": datetime.utcnow().isoformat(),
        "initiated_by": current_user.email,
        **result
    }


@router.get(
    "/status", 
    summary="📊 Get Job Status",
    response_model=JobStatusResponse,
    description="Get comprehensive status of all background jobs including APScheduler status."
)
async def get_jobs_status(
    current_user: models.User = Depends(get_current_admin)
):
    """
    **Get the current status of all background jobs.**
    
    Returns:
    - Global stop flag status
    - Paused job types
    - Count of running jobs by type
    - List of currently running jobs with duration
    """
    status = job_manager.get_status()
    
    # Also try to get APScheduler status
    scheduler_info = {}
    try:
        from quiz_router import scheduler as quiz_scheduler
        if quiz_scheduler:
            scheduler_info = {
                "running": quiz_scheduler.running,
                "jobs": [
                    {
                        "id": job.id,
                        "name": job.name,
                        "next_run_time": str(job.next_run_time) if job.next_run_time else None
                    }
                    for job in quiz_scheduler.get_jobs()
                ]
            }
    except Exception as e:
        scheduler_info = {"error": str(e)}
    
    return {
        "success": True,
        "timestamp": datetime.utcnow().isoformat(),
        "job_manager": status,
        "apscheduler": scheduler_info
    }


@router.post(
    "/pause/{job_type}", 
    summary="⏸️ Pause Specific Job Type",
    response_model=PauseResumeResponse,
    description="Pause a specific type of background jobs. New jobs of this type will be blocked."
)
async def pause_job_type(
    job_type: JobTypeEnum,
    current_user: models.User = Depends(get_current_admin)
):
    """
    **Pause a specific type of jobs.**
    
    Valid job types:
    - `feed_generation` - AI feed creation
    - `youtube_transcription` - YouTube video transcription
    - `blog_scraping` - Blog content scraping
    - `quiz_generation` - Quiz creation
    - `all` - All job types
    """
    result = job_manager.pause_job_type(job_type.value)
    logger.info(f"⏸️ Admin {current_user.email} paused {job_type.value} jobs")
    return {
        "success": True,
        "timestamp": datetime.utcnow().isoformat(),
        "initiated_by": current_user.email,
        **result
    }


@router.post(
    "/resume/{job_type}", 
    summary="▶️ Resume Specific Job Type",
    response_model=PauseResumeResponse,
    description="Resume a specific type of background jobs that was previously paused."
)
async def resume_job_type(
    job_type: JobTypeEnum,
    current_user: models.User = Depends(get_current_admin)
):
    """
    **Resume a specific type of jobs.**
    
    Valid job types:
    - `feed_generation` - AI feed creation
    - `youtube_transcription` - YouTube video transcription
    - `blog_scraping` - Blog content scraping
    - `quiz_generation` - Quiz creation
    - `all` - All job types
    """
    result = job_manager.resume_job_type(job_type.value)
    logger.info(f"▶️ Admin {current_user.email} resumed {job_type.value} jobs")
    return {
        "success": True,
        "timestamp": datetime.utcnow().isoformat(),
        "initiated_by": current_user.email,
        **result
    }


@router.post(
    "/clear-state", 
    summary="🧹 Clear Job Manager State",
    response_model=ClearStateResponse,
    description="Reset the job manager to a clean state. Use after stopping jobs to start fresh."
)
async def clear_job_state(
    current_user: models.User = Depends(get_current_admin)
):
    """
    **Reset the job manager to a clean state.**
    
    This will:
    - Clear the stop flag
    - Clear all paused states
    - Clear the running jobs list
    - Reset job counts to zero
    
    Use this after stopping jobs to start fresh.
    """
    result = job_manager.clear_stopped_state()
    logger.info(f"🧹 Admin {current_user.email} cleared job manager state")
    return {
        "success": True,
        "timestamp": datetime.utcnow().isoformat(),
        "initiated_by": current_user.email,
        **result
    }


@router.post("/stop-scheduler", summary="🛑 Stop Quiz Scheduler")
async def stop_scheduler(
    current_user: models.User = Depends(get_current_admin)
):
    """
    **Stop the APScheduler (quiz auto-update scheduler).**
    """
    try:
        from quiz_router import scheduler as quiz_scheduler
        if quiz_scheduler:
            if quiz_scheduler.running:
                quiz_scheduler.shutdown(wait=False)
                logger.warning(f"🛑 Quiz scheduler stopped by {current_user.email}")
                return {
                    "success": True,
                    "message": "Quiz scheduler stopped",
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                return {
                    "success": True,
                    "message": "Quiz scheduler was not running",
                    "timestamp": datetime.utcnow().isoformat()
                }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/running", summary="📋 List Running Jobs")
async def list_running_jobs(
    current_user: models.User = Depends(get_current_admin)
):
    """
    **Get a detailed list of all currently running jobs.**
    """
    status = job_manager.get_status()
    
    return {
        "success": True,
        "timestamp": datetime.utcnow().isoformat(),
        "total_running": status["running_jobs_count"],
        "jobs_by_type": status["job_counts_by_type"],
        "jobs": status["running_jobs"]
    }


# ==================== Database Job Status ====================

@router.get("/db/transcription-jobs", summary="📺 Get Transcription Jobs Status")
async def get_transcription_jobs_status(
    status_filter: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_admin)
):
    """
    **Get status of YouTube transcription jobs from database.**
    
    Status can be: queued, processing, completed, failed
    """
    query = db.query(models.TranscriptJob)
    
    if status_filter:
        query = query.filter(models.TranscriptJob.status == status_filter)
    
    jobs = query.order_by(models.TranscriptJob.created_at.desc()).limit(limit).all()
    
    return {
        "success": True,
        "total": len(jobs),
        "jobs": [
            {
                "id": job.id,
                "job_id": job.job_id,
                "url": job.url,
                "type": job.type.value if job.type else None,
                "status": job.status.value if job.status else None,
                "total_items": job.total_items,
                "processed_items": job.processed_items,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None
            }
            for job in jobs
        ]
    }


@router.post("/db/cancel-processing-jobs", summary="🛑 Cancel Processing Jobs in DB")
async def cancel_processing_jobs(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_admin)
):
    """
    **Mark all 'processing' transcription jobs as 'failed' in the database.**
    
    This helps clean up stuck jobs.
    """
    from models import JobStatus
    
    # Update transcription jobs
    updated = db.query(models.TranscriptJob).filter(
        models.TranscriptJob.status == JobStatus.PROCESSING
    ).update({"status": JobStatus.FAILED})
    
    # Update scrape jobs
    scrape_updated = db.query(models.ScrapeJob).filter(
        models.ScrapeJob.status == "inprocess"
    ).update({"status": "failed"})
    
    db.commit()
    
    logger.warning(f"🛑 Admin {current_user.email} cancelled {updated} transcription jobs and {scrape_updated} scrape jobs")
    
    return {
        "success": True,
        "timestamp": datetime.utcnow().isoformat(),
        "transcription_jobs_cancelled": updated,
        "scrape_jobs_cancelled": scrape_updated,
        "message": f"Cancelled {updated + scrape_updated} processing jobs"
    }


@router.get("/db/scrape-jobs", summary="🕷️ Get Scrape Jobs Status")
async def get_scrape_jobs_status(
    status_filter: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_admin)
):
    """
    **Get status of web scraping jobs from database.**
    
    Status can be: inprocess, done, failed
    """
    query = db.query(models.ScrapeJob)
    
    if status_filter:
        query = query.filter(models.ScrapeJob.status == status_filter)
    
    jobs = query.order_by(models.ScrapeJob.created_at.desc()).limit(limit).all()
    
    return {
        "success": True,
        "total": len(jobs),
        "jobs": [
            {
                "id": job.id,
                "uid": job.uid,
                "website": job.website,
                "url": job.url,
                "status": job.status,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "updated_at": job.updated_at.isoformat() if job.updated_at else None
            }
            for job in jobs
        ]
    }
