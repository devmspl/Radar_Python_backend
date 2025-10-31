from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List,Union
import json
from database import get_db
from dependencies import get_current_user
import models
import schemas
from youtube_service import *

router = APIRouter(prefix="/youtube", tags=["youtube"])

@router.post("/transcribe", response_model=schemas.TranscriptResponse)
async def transcribe_content(
    request: schemas.TranscriptRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Start YouTube transcription with optional feed generation"""
    return create_transcript_job(db, request, current_user.id)

@router.get("/job/{job_id}", response_model=schemas.JobContentStatusResponse)
async def get_job_status(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    return get_job_status_service(db, job_id)

@router.get("/jobs", response_model=List[schemas.JobStatusResponse])
async def get_user_jobs(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Get all jobs for the current user"""
    jobs = db.query(models.TranscriptJob).filter(models.TranscriptJob.user_id == current_user.id).all()
    
    result = []
    for job in jobs:
        # Get playlists if available
        playlists = []
        if job.playlists:
            try:
                playlists = json.loads(job.playlists)
            except json.JSONDecodeError:
                pass
        
        # Get transcripts
        transcripts = []
        for transcript in job.transcripts:
            transcripts.append({
                'id': transcript.transcript_id,
                'video_id': transcript.video_id,
                'playlist_id': transcript.playlist_id,
                'title': transcript.title,
                'transcript': transcript.transcript_text,
                'duration': transcript.duration,
                'word_count': transcript.word_count,
                'created_at': transcript.created_at,
                'description': transcript.description
            })
        
        result.append(schemas.JobStatusResponse(
            job_id=job.job_id,
            url=job.url,
            type=job.type,
            status=job.status,
            created_at=job.created_at,
            completed_at=job.completed_at,
            total_items=job.total_items,
            processed_items=job.processed_items,
            description=job.description,
            content_name=job.content_name,
            playlists=playlists,
            transcripts=transcripts
        ))
    
    return result

@router.delete("/job/{job_id}")
async def delete_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Delete a job and its transcripts"""
    job = db.query(models.TranscriptJob).filter(
        models.TranscriptJob.job_id == job_id,
        models.TranscriptJob.user_id == current_user.id
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Delete associated transcripts
    db.query(models.Transcript).filter(models.Transcript.job_id == job.id).delete()
    
    # Delete the job
    db.delete(job)
    db.commit()
    
    return {"message": "Job deleted successfully"}
@router.get("/job/{job_id}/content", response_model=Union[
    schemas.ChannelPlaylistsResponse,
    schemas.PlaylistVideosResponse,
    schemas.VideoTranscriptResponse
])
async def get_job_content(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Get channel playlists, playlist videos, or video transcript depending on job type"""
    return fetch_job_content(db, job_id, current_user.id)

@router.get("/video/{video_id}", response_model=schemas.VideoTranscriptResponse)
def get_video_by_id(
    video_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Fetch a single video and its transcript by video_id"""
    # Find transcript for this video
    transcript_obj = db.query(models.Transcript).filter(models.Transcript.video_id == video_id).first()

    if not transcript_obj:
        raise HTTPException(status_code=404, detail="Transcript not found for this video")

    # Get video metadata (title, description, etc.)
    video_title = transcript_obj.title or get_youtube_title(f"https://www.youtube.com/watch?v={video_id}")

    video_data = schemas.VideoWithTranscript(
        id=video_id,
        title=video_title,
        description=transcript_obj.description,
        duration=(
            int(transcript_obj.duration)
            if transcript_obj.duration and str(transcript_obj.duration).isdigit()
            else None
        ),
        transcript=transcript_obj.transcript_text
    )

    return schemas.VideoTranscriptResponse(video_id=video_id, video=video_data)
@router.get("/channel/{channel_id}", response_model=schemas.ChannelWithPlaylists)
def get_channel_by_id(channel_id: str):
    """Fetch channel details with playlists and videos"""
    channel_url = f"https://www.youtube.com/{channel_id}"
    return fetch_channel_by_id(channel_url)


@router.get("/playlist/{playlist_id}", response_model=schemas.PlaylistWithVideos)
def get_playlist_by_id(playlist_id: str):
    """Fetch playlist details with its videos"""
    playlist_url = f"https://www.youtube.com/playlist?list={playlist_id}"
    return fetch_playlist_by_id(playlist_url)
