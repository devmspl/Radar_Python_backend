import os
import re
import uuid
import json
import time
import httpx
import logging
import isodate
from typing import List, Dict, Optional, Any
from datetime import datetime
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from concurrent.futures import ThreadPoolExecutor
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from database import get_db_session
from models import TranscriptJob, Transcript, ContentType, JobStatus
from schemas import TranscriptRequest, TranscriptResponse, JobStatusResponse
from schemas import *

# ===========================
# CONFIGURATION
# ===========================
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'
YOUTUBE_API_KEY = "AIzaSyBNcFGPlGcRS1ehchAxJdXDOo3s0k660tY" # Set your API key in environment

# ===========================
# LOGGING
# ===========================   
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================
# GLOBALS
# ===========================
background_executor = ThreadPoolExecutor(max_workers=3)

# ===========================
# API KEY VALIDATION
# ===========================
def validate_api_key():
    """Validate the YouTube API key"""
    if not YOUTUBE_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="YouTube API key not configured. Set YOUTUBE_API_KEY environment variable."
        )
    
    # Test the API key with a simple request
    try:
        youtube = build(API_SERVICE_NAME, API_VERSION, developerKey=YOUTUBE_API_KEY)
        request = youtube.videos().list(part="snippet", id="dQw4w9WgXcQ")  # Test with a known video
        request.execute()
        return True
    except HttpError as e:
        logger.error(f"API key validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"YouTube API key is invalid: {e}"
        )

# ===========================
# YOUTUBE API HELPERS
# ===========================
def get_youtube_service():
    """Get YouTube API service with API key"""
    validate_api_key()
    return build(API_SERVICE_NAME, API_VERSION, developerKey=YOUTUBE_API_KEY)

def extract_video_id(url: str) -> Optional[str]:
    """Extract video ID from YouTube URL"""
    patterns = [
        r"youtube\.com/watch\?v=([^&]+)",
        r"youtu\.be/([^?]+)",
        r"youtube\.com/embed/([^/?]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def extract_playlist_id(url: str) -> Optional[str]:
    """Extract playlist ID from YouTube URL"""
    patterns = [r"list=([^&]+)", r"youtube\.com/playlist\?list=([^&]+)"]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def extract_channel_id(url: str) -> Optional[str]:
    """Extract channel ID from YouTube URL"""
    patterns = [
        r"youtube\.com/channel/([^/]+)",
        r"youtube\.com/c/([^/]+)",
        r"youtube\.com/user/([^/]+)",
        r"youtube\.com/@([^/]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def determine_content_type(url: str) -> ContentType:
    """Determine if URL is a channel, playlist, or video"""
    if extract_channel_id(url):
        return ContentType.CHANNEL
    elif extract_playlist_id(url):
        return ContentType.PLAYLIST
    elif extract_video_id(url):
        return ContentType.VIDEO
    else:
        raise HTTPException(status_code=400, detail="Unsupported YouTube URL format")

# ===========================
# YOUTUBE DATA API FUNCTIONS
# ===========================
def get_video_details(youtube, video_id: str) -> Optional[Dict]:
    """Get video details using YouTube Data API"""
    try:
        request = youtube.videos().list(
            part="snippet,contentDetails",
            id=video_id
        )
        response = request.execute()
        
        if response['items']:
            item = response['items'][0]
            # Convert ISO 8601 duration to seconds
            duration = isodate.parse_duration(item['contentDetails']['duration']).total_seconds()
            
            return {
                'id': video_id,
                'title': item['snippet']['title'],
                'description': item['snippet'].get('description', ''),
                'duration': int(duration),
                'publishedAt': item['snippet']['publishedAt']
            }
        return None
    except HttpError as e:
        logger.error(f"Error getting video details: {e}")
        return None

def get_playlist_details(youtube, playlist_id: str) -> Optional[Dict]:
    """Get playlist details using YouTube Data API"""
    try:
        request = youtube.playlists().list(
            part="snippet",
            id=playlist_id
        )
        response = request.execute()
        
        if response['items']:
            item = response['items'][0]
            return {
                'id': playlist_id,
                'title': item['snippet']['title'],
                'description': item['snippet'].get('description', ''),
                'publishedAt': item['snippet']['publishedAt']
            }
        return None
    except HttpError as e:
        logger.error(f"Error getting playlist details: {e}")
        return None

def get_channel_details(youtube, channel_id: str) -> Optional[Dict]:
    """Get channel details using YouTube Data API"""
    try:
        request = youtube.channels().list(
            part="snippet",
            id=channel_id
        )
        response = request.execute()
        
        if response['items']:
            item = response['items'][0]
            return {
                'id': channel_id,
                'title': item['snippet']['title'],
                'description': item['snippet'].get('description', ''),
                'publishedAt': item['snippet']['publishedAt']
            }
        return None
    except HttpError as e:
        logger.error(f"Error getting channel details: {e}")
        return None

def get_playlist_videos(youtube, playlist_id: str) -> List[Dict]:
    """Get all videos from a playlist using YouTube Data API"""
    videos = []
    next_page_token = None
    
    try:
        while True:
            request = youtube.playlistItems().list(
                part="snippet",
                playlistId=playlist_id,
                maxResults=50,
                pageToken=next_page_token
            )
            response = request.execute()
            
            for item in response['items']:
                if 'resourceId' in item['snippet'] and 'videoId' in item['snippet']['resourceId']:
                    video_id = item['snippet']['resourceId']['videoId']
                    videos.append({
                        'id': video_id,
                        'title': item['snippet']['title'],
                        'description': item['snippet'].get('description', ''),
                        'publishedAt': item['snippet']['publishedAt']
                    })
            
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
                
    except HttpError as e:
        logger.error(f"Error getting playlist videos: {e}")
    
    return videos

def get_channel_playlists(youtube, channel_id: str) -> List[Dict]:
    """Get all playlists from a channel using YouTube Data API"""
    playlists = []
    next_page_token = None
    
    try:
        while True:
            request = youtube.playlists().list(
                part="snippet",
                channelId=channel_id,
                maxResults=50,
                pageToken=next_page_token
            )
            response = request.execute()
            
            for item in response['items']:
                playlists.append({
                    'id': item['id'],
                    'title': item['snippet']['title'],
                    'description': item['snippet'].get('description', ''),
                    'publishedAt': item['snippet']['publishedAt']
                })
            
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
                
    except HttpError as e:
        logger.error(f"Error getting channel playlists: {e}")
    
    return playlists

# ===========================
# CAPTIONS API FUNCTIONS
# ===========================
def get_video_captions(video_id: str) -> Optional[str]:
    """Get captions/transcript for a video using YouTube Captions API"""
    try:
        # For API key access, we need to use a different approach since captions
        # typically require OAuth. We'll use a workaround with the transcripts API
        # or fall back to a different method if available
        
        # This is a placeholder - you might need to implement a different approach
        # for captions with API key only access
        
        logger.warning(f"Caption access with API key only is limited. Video ID: {video_id}")
        return None
        
    except Exception as e:
        logger.error(f"Error getting captions for video {video_id}: {e}")
        return None

# Alternative approach: Use a third-party service or different API for transcripts
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

def get_transcript_fallback(video_id: str) -> Optional[str]:
    """Fallback method to get transcripts if YouTube API doesn't work"""
    try:
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t["text"] for t in transcript_data])

    except TranscriptsDisabled:
        logger.warning(f"Transcripts are disabled for video {video_id}")
    except NoTranscriptFound:
        logger.warning(f"No transcript found for video {video_id}")
    # except TooManyRequests:
    #     logger.error(f"Rate limited by YouTube while fetching transcript for {video_id}")
    except Exception as e:
        logger.error(f"Unexpected error in fallback transcript method for {video_id}: {e}")

    return None


# ===========================
# PROCESSING FUNCTIONS
# ===========================
def update_job_status(job_id: str, status: JobStatus, total_items: int = None, processed_items: int = None):
    """Update job status in database"""
    db = next(get_db_session())
    try:
        job = db.query(TranscriptJob).filter(TranscriptJob.job_id == job_id).first()
        if not job:
            return
        
        job.status = status
        if status == JobStatus.COMPLETED or status == JobStatus.FAILED:
            job.completed_at = datetime.now()
        if total_items is not None:
            job.total_items = total_items
        if processed_items is not None:
            job.processed_items = processed_items
        
        db.commit()
    finally:
        db.close()

def process_video_background(video_url: str, job_id: str, store_in_db: bool, playlist_id: Optional[str] = None):
    """Process a single video in background using YouTube APIs"""
    try:
        youtube = get_youtube_service()
        video_id = extract_video_id(video_url)
        
        if not video_id:
            update_job_status(job_id, JobStatus.FAILED)
            return False
        
        # Get video details
        video_details = get_video_details(youtube, video_id)
        if not video_details:
            update_job_status(job_id, JobStatus.FAILED)
            return False
        
        # Get captions/transcript
        transcript = get_video_captions(video_id)
        
        # If YouTube API doesn't work, try fallback
        if not transcript:
            transcript = get_transcript_fallback(video_id)
        
        if transcript:
            # Generate description
            description = f"Transcript available for: {video_details['title']}"
            
            if store_in_db:
                db = next(get_db_session())
                try:
                    job = db.query(TranscriptJob).filter(TranscriptJob.job_id == job_id).first()
                    if job:
                        transcript_obj = Transcript(
                            transcript_id=str(uuid.uuid4()),
                            video_id=video_id,
                            playlist_id=playlist_id,
                            title=video_details['title'],
                            transcript_text=transcript,
                            word_count=len(transcript.split()),
                            description=description,
                            duration=video_details['duration'],
                            job_id=job.id
                        )
                        db.add(transcript_obj)
                        db.commit()
                finally:
                    db.close()
            
            update_job_status(job_id, JobStatus.COMPLETED, total_items=1, processed_items=1)
            return True
        
        update_job_status(job_id, JobStatus.FAILED)
        return False
        
    except Exception as e:
        logger.error(f"Error processing video {video_url}: {e}")
        update_job_status(job_id, JobStatus.FAILED)
        return False

def process_playlist_background(playlist_url: str, job_id: str, store_in_db: bool):
    """Process playlist in background using YouTube APIs"""
    try:
        youtube = get_youtube_service()
        playlist_id = extract_playlist_id(playlist_url)
        
        if not playlist_id:
            update_job_status(job_id, JobStatus.FAILED, 0, 0)
            return
        
        # Get playlist details
        playlist_details = get_playlist_details(youtube, playlist_id)
        playlist_name = playlist_details['title'] if playlist_details else "Unknown Playlist"
        
        # Update job with playlist name
        db = next(get_db_session())
        try:
            job = db.query(TranscriptJob).filter(TranscriptJob.job_id == job_id).first()
            if job:
                job.content_name = playlist_name
                db.commit()
        finally:
            db.close()
        
        # Get playlist videos
        videos = get_playlist_videos(youtube, playlist_id)
        
        if not videos:
            update_job_status(job_id, JobStatus.FAILED, 0, 0)
            return
        
        update_job_status(job_id, JobStatus.PROCESSING, len(videos), 0)
        
        processed_count = 0
        for i, video in enumerate(videos):
            success = process_video_background(
                f"https://www.youtube.com/watch?v={video['id']}",
                job_id,
                store_in_db,
                playlist_id
            )
            if success:
                processed_count += 1
            update_job_status(job_id, JobStatus.PROCESSING, len(videos), i + 1)
        
        update_job_status(job_id, JobStatus.COMPLETED, len(videos), processed_count)
        
    except Exception as e:
        logger.error(f"Error processing playlist: {e}")
        update_job_status(job_id, JobStatus.FAILED)

def process_channel_background(channel_url: str, job_id: str, store_in_db: bool):
    """Process channel in background using YouTube APIs"""
    try:
        youtube = get_youtube_service()
        channel_id = extract_channel_id(channel_url)
        
        if not channel_id:
            update_job_status(job_id, JobStatus.FAILED, 0, 0)
            return
        
        # Get channel details
        channel_details = get_channel_details(youtube, channel_id)
        channel_name = channel_details['title'] if channel_details else "Unknown Channel"
        
        # Update job with channel name
        db = next(get_db_session())
        try:
            job = db.query(TranscriptJob).filter(TranscriptJob.job_id == job_id).first()
            if job:
                job.content_name = channel_name
                db.commit()
        finally:
            db.close()
        
        # Get channel playlists
        playlists = get_channel_playlists(youtube, channel_id)
        
        if not playlists:
            update_job_status(job_id, JobStatus.FAILED, 0, 0)
            return
        
        # Store playlist information
        playlist_info = []
        for playlist in playlists:
            playlist_info.append({
                'id': playlist['id'],
                'title': playlist['title'],
                'description': playlist.get('description', '')
            })
        
        # Update job with playlists
        db = next(get_db_session())
        try:
            job = db.query(TranscriptJob).filter(TranscriptJob.job_id == job_id).first()
            if job:
                job.playlists = json.dumps(playlist_info)
                db.commit()
        finally:
            db.close()
        
        # Calculate total videos across all playlists
        total_videos = 0
        for playlist in playlists:
            videos = get_playlist_videos(youtube, playlist['id'])
            total_videos += len(videos) if videos else 0
        
        update_job_status(job_id, JobStatus.PROCESSING, total_videos, 0)
        
        # Process videos from all playlists
        processed_count = 0
        for playlist in playlists:
            playlist_videos = get_playlist_videos(youtube, playlist['id'])
            
            if not playlist_videos:
                continue
                
            for video in playlist_videos:
                success = process_video_background(
                    f"https://www.youtube.com/watch?v={video['id']}",
                    job_id,
                    store_in_db,
                    playlist['id']
                )
                if success:
                    processed_count += 1
                update_job_status(job_id, JobStatus.PROCESSING, total_videos, processed_count)
        
        update_job_status(job_id, JobStatus.COMPLETED, total_videos, processed_count)
        
    except Exception as e:
        logger.error(f"Error processing channel: {e}")
        update_job_status(job_id, JobStatus.FAILED)

# ===========================
# MAIN SERVICE FUNCTIONS
# ===========================   
def create_transcript_job(db: Session, request: TranscriptRequest, user_id: int) -> TranscriptResponse:
    """Create a new transcript job using YouTube APIs"""
    try:
        url = request.youtube_url
        content_type = determine_content_type(url)
        
        youtube = get_youtube_service()
        content_name = None
        playlists = []
        
        if content_type == ContentType.VIDEO:
            video_id = extract_video_id(url)
            video_details = get_video_details(youtube, video_id)
            content_name = video_details['title'] if video_details else "Unknown Video"
            message = "Started processing video"
            
        elif content_type == ContentType.PLAYLIST:
            playlist_id = extract_playlist_id(url)
            playlist_details = get_playlist_details(youtube, playlist_id)
            content_name = playlist_details['title'] if playlist_details else "Unknown Playlist"
            message = "Started processing playlist"
            
        elif content_type == ContentType.CHANNEL:
            channel_id = extract_channel_id(url)
            channel_details = get_channel_details(youtube, channel_id)
            content_name = channel_details['title'] if channel_details else "Unknown Channel"
            playlists_data = get_channel_playlists(youtube, channel_id)
            playlists = [{'id': p['id'], 'title': p['title']} for p in playlists_data]
            message = "Started processing channel"
        
        # Create job in database
        job = TranscriptJob(
            url=url,
            type=content_type,
            status=JobStatus.PROCESSING,
            model_size="youtube-api",
            store_in_db=request.store_in_db,
            content_name=content_name,
            playlists=json.dumps(playlists) if playlists else None,
            user_id=user_id
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        
        # Start background processing
        if content_type == ContentType.VIDEO:
            background_executor.submit(
                process_video_background, 
                url, job.job_id, request.store_in_db
            )
            
        elif content_type == ContentType.PLAYLIST:
            background_executor.submit(
                process_playlist_background,
                url, job.job_id, request.store_in_db
            )
            
        elif content_type == ContentType.CHANNEL:
            background_executor.submit(
                process_channel_background,
                url, job.job_id, request.store_in_db
            )
        
        return TranscriptResponse(
            job_id=job.job_id,
            status=JobStatus.PROCESSING,
            message=message,
            content_type=content_type,
            content_name=content_name,
            playlists=playlists
        )
        
    except Exception as e:
        logger.error(f"Error creating transcript job: {e}")
        raise HTTPException(status_code=400, detail=str(e)) 

# ===========================
# ADDITIONAL SERVICE FUNCTIONS
# ===========================
def get_job_status_service(db: Session, job_id: str) -> JobContentStatusResponse:
    """Get job status with content information"""
    job = db.query(TranscriptJob).filter(TranscriptJob.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    transcripts = db.query(Transcript).filter(Transcript.job_id == job.id).all()
    processed_items = job.processed_items or len(transcripts)
    total_items = job.total_items or (len(transcripts) if job.type == ContentType.VIDEO else 0)

    return JobContentStatusResponse(
        job_id=job.job_id,
        status=job.status,
        processed_items=processed_items,
        total_items=total_items,
        type=job.type,
    )

def search_youtube_videos(query: str, max_results: int = 10) -> List[Dict]:
    """Search for YouTube videos using Data API"""
    try:
        youtube = get_youtube_service()
        request = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            maxResults=max_results
        )
        response = request.execute()
        
        videos = []
        for item in response['items']:
            videos.append({
                'id': item['id']['videoId'],
                'title': item['snippet']['title'],
                'description': item['snippet'].get('description', ''),
                'publishedAt': item['snippet']['publishedAt'],
                'channelTitle': item['snippet']['channelTitle']
            })
        
        return videos
    except HttpError as e:
        logger.error(f"Error searching videos: {e}")
        return []

def get_video_statistics(video_id: str) -> Optional[Dict]:
    """Get video statistics using Data API"""
    try:
        youtube = get_youtube_service()
        request = youtube.videos().list(
            part="statistics",
            id=video_id
        )
        response = request.execute()
        
        if response['items']:
            stats = response['items'][0]['statistics']
            return {
                'viewCount': int(stats.get('viewCount', 0)),
                'likeCount': int(stats.get('likeCount', 0)),
                'commentCount': int(stats.get('commentCount', 0))
            }
        return None
    except HttpError as e:
        logger.error(f"Error getting video statistics: {e}")
        return None

def get_channel_by_id(channel_url: str) -> ChannelWithPlaylists:
    """Fetch channel metadata + playlists + videos"""
    youtube = get_youtube_service()
    channel_id = extract_channel_id(channel_url)
    
    if not channel_id:
        raise HTTPException(status_code=400, detail="Invalid channel URL")
    
    # Get channel details
    channel_details = get_channel_details(youtube, channel_id)
    if not channel_details:
        raise HTTPException(status_code=404, detail="Channel not found")
    
    # Get channel playlists
    playlists = get_channel_playlists(youtube, channel_id)
    enriched_playlists = []

    for pl in playlists:
        videos = get_playlist_videos(youtube, pl['id'])
        enriched_playlists.append(
            ChannelPlaylist(
                id=pl['id'],
                title=pl['title'],
                description=pl.get('description', ''),
                videos=[
                    PlaylistVideo(video_id=v['id'], title=v['title']) for v in videos
                ],
            )
        )

    return ChannelWithPlaylists(
        channel_id=channel_id,
        title=channel_details['title'],
        description=channel_details.get('description', ''),
        playlists=enriched_playlists,
    )

def get_playlist_by_id(playlist_url: str) -> PlaylistWithVideos:
    """Fetch playlist metadata + videos"""
    youtube = get_youtube_service()
    playlist_id = extract_playlist_id(playlist_url)
    
    if not playlist_id:
        raise HTTPException(status_code=400, detail="Invalid playlist URL")
    
    # Get playlist details
    playlist_details = get_playlist_details(youtube, playlist_id)
    if not playlist_details:
        raise HTTPException(status_code=404, detail="Playlist not found")
    
    # Get playlist videos
    videos = get_playlist_videos(youtube, playlist_id)

    return PlaylistWithVideos(
        playlist_id=playlist_id,
        title=playlist_details['title'],
        description=playlist_details.get('description', ''),
        videos=[PlaylistVideo(video_id=v['id'], title=v['title']) for v in videos],
    )

# ===========================
# CONTENT FETCHING FUNCTIONS
# ===========================
def fetch_job_content(db: Session, job_id: str, user_id: int):
    """Fetch channel playlists, playlist videos, or video transcript based on job type."""
    job = db.query(TranscriptJob).filter_by(job_id=job_id, user_id=user_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    youtube = get_youtube_service()

    if job.type == ContentType.CHANNEL:
        channel_id = extract_channel_id(job.url)
        playlists = get_channel_playlists(youtube, channel_id)
        enriched_playlists = []

        for pl in playlists:
            videos = get_playlist_videos(youtube, pl['id'])
            video_items = [
                PlaylistVideo(
                    video_id=v['id'],
                    title=v['title']
                )
                for v in videos
            ]   
            enriched_playlists.append(
                ChannelPlaylist(
                    id=pl['id'],
                    title=pl['title'],
                    description=pl.get('description', ''),
                    videos=video_items
                )
            )

        return ChannelPlaylistsResponse(
            channel_id=channel_id,
            playlists=enriched_playlists
        )

    elif job.type == ContentType.PLAYLIST:
        playlist_id = extract_playlist_id(job.url)
        videos = get_playlist_videos(youtube, playlist_id)

        video_items = [
            PlaylistVideo(
                video_id=v['id'],
                title=v['title']
            )
            for v in videos
        ]

        return PlaylistVideosResponse(
            playlist_id=playlist_id,
            playlist_name=job.content_name,
            videos=video_items
        )

    elif job.type == ContentType.VIDEO:
        transcript_obj = db.query(Transcript).filter_by(job_id=job.id).first()
        video_id = extract_video_id(job.url)
        video_details = get_video_details(youtube, video_id)

        video_data = VideoWithTranscript(
            id=video_id,
            title=video_details['title'] if video_details else "Unknown Title",
            description=transcript_obj.description if transcript_obj else None,
            duration=transcript_obj.duration if transcript_obj else None,
            transcript=transcript_obj.transcript_text if transcript_obj else None,
        )

        return VideoTranscriptResponse(
            video_id=video_id,
            video=video_data
        )

    else:
        raise HTTPException(status_code=400, detail="Unsupported content type")