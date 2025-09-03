import os
import json
import whisper
import yt_dlp
from typing import List, Dict, Optional
import re
import uuid
from datetime import datetime
from fastapi import HTTPException
from sqlalchemy.orm import Session
from concurrent.futures import ThreadPoolExecutor
import torch
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer
from database import get_db_session
from models import TranscriptJob, Transcript, ContentType, JobStatus
from schemas import TranscriptRequest, TranscriptResponse, JobStatusResponse
import utils

# Initialize BART summarization model
try:
    bart_model_name = "facebook/bart-large-cnn"
    print("Loading BART model...")
    bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)
    bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name)
    summarizer = pipeline("summarization", model=bart_model, tokenizer=bart_tokenizer)
    print("BART model loaded successfully!")
except Exception as e:
    print(f"Error loading BART model: {e}")
    summarizer = None

# Global variables
MODEL_CACHE = {}
background_executor = ThreadPoolExecutor(max_workers=3)

def get_whisper_model(model_size: str):
    """Get Whisper model from cache or load it"""
    if model_size not in MODEL_CACHE:
        MODEL_CACHE[model_size] = whisper.load_model(model_size)
    return MODEL_CACHE[model_size]

def determine_content_type(url: str) -> ContentType:
    """Determine if URL is a channel, playlist, or video"""
    if "youtube.com/channel/" in url or "youtube.com/c/" in url or "youtube.com/user/" in url or "youtube.com/@" in url:
        return ContentType.CHANNEL
    elif "youtube.com/playlist" in url or "list=" in url:
        return ContentType.PLAYLIST
    elif "youtube.com/watch" in url or "youtu.be/" in url:
        return ContentType.VIDEO
    else:
        raise HTTPException(status_code=400, detail="Unsupported YouTube URL format")

def extract_video_id(url: str) -> Optional[str]:
    """Extract video ID from YouTube URL"""
    patterns = [
        r"youtube\.com/watch\?v=([^&]+)",
        r"youtu\.be/([^?]+)",
        r"youtube\.com/embed/([^/?]+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def extract_playlist_id(url: str) -> Optional[str]:
    """Extract playlist ID from YouTube URL"""
    patterns = [
        r"list=([^&]+)",
        r"youtube\.com/playlist\?list=([^&]+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_playlist_name(playlist_url: str) -> str:
    """Get playlist name using yt-dlp"""
    try:
        ydl_opts = {
            'extract_flat': True,
            'skip_download': True,
            'quiet': True,
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(playlist_url, download=False)
            return info.get('title', 'Unknown Playlist')
    except Exception as e:
        print(f"Error getting playlist name: {e}")
        return "Unknown Playlist"

def get_channel_name(channel_url: str) -> str:
    """Get channel name using yt-dlp"""
    try:
        ydl_opts = {
            'extract_flat': True,
            'skip_download': True,
            'quiet': True,
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(channel_url, download=False)
            return info.get('title', 'Unknown Channel') or info.get('uploader', 'Unknown Channel')
    except Exception as e:
        print(f"Error getting channel name: {e}")
        return "Unknown Channel"

def get_youtube_title(url: str) -> str:
    """Extract YouTube video title using yt-dlp"""
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            return info_dict.get('title', 'Unknown Title')
    except Exception as e:
        print(f"Error getting YouTube title: {e}")
        return "Unknown Title"

def get_playlist_videos_ytdlp(playlist_url: str) -> List[Dict]:
    """Get videos from a playlist using yt-dlp"""
    ydl_opts = {
        'extract_flat': True,
        'skip_download': True,
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': True,
        'force_json': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(playlist_url, download=False)
            videos = []
            if "entries" in info:
                for entry in info["entries"]:
                    if entry and entry.get('id'):
                        video_id = entry.get('id')
                        video_title = entry.get('title', 'Unknown Title')
                        video_url = f"https://www.youtube.com/watch?v={video_id}"
                        videos.append({'id': video_id, 'title': video_title, 'url': video_url})
            return videos
    except Exception as e:
        print(f"Error getting playlist videos: {e}")
        return []

def get_channel_playlists_ytdlp(channel_url: str) -> List[Dict]:
    """Get all playlists from a YouTube channel using yt-dlp"""
    ydl_opts = {
        'extract_flat': True,
        'skip_download': True,
        'quiet': True,
        'no_warnings': True,
        'extractor_args': {'youtube': {'skip': ['webpage', 'auth', 'webpage', 'webpage', 'webpage']}}
    }
    try:
        if "/playlists" not in channel_url:
            channel_url = channel_url.rstrip("/") + "/playlists"
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(channel_url, download=False)
            playlists = []
            if "entries" in info:
                for entry in info["entries"]:
                    if entry and 'url' in entry:
                        playlist_id = extract_playlist_id(entry['url'])
                        if playlist_id:
                            playlists.append({
                                'id': playlist_id,
                                'title': entry.get('title', f'Playlist {playlist_id}'),
                                'url': entry['url']
                            })
            return playlists
    except Exception as e:
        print(f"Error getting channel playlists: {e}")
        return []

def download_youtube_audio(video_id: str, output_dir: str = "temp_audio") -> Optional[str]:
    """Download audio from YouTube video"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{video_id}.wav")
    
    if os.path.exists(output_file):
        return output_file
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_dir, f"{video_id}.%(ext)s"),
        'quiet': False,
        'no_warnings': False,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        if os.path.exists(output_file):
            return output_file
        else:
            for ext in ['wav', 'mp3', 'm4a', 'webm']:
                possible_file = os.path.join(output_dir, f"{video_id}.{ext}")
                if os.path.exists(possible_file):
                    return possible_file
            return None
    except Exception as e:
        print(f"Error downloading audio for {video_id}: {e}")
        return None

def transcribe_audio(audio_file: str, model_size: str = "base") -> Optional[str]:
    """Transcribe audio using Whisper"""
    try:
        if not os.path.exists(audio_file):
            return None
        file_size = os.path.getsize(audio_file)
        if file_size == 0:
            return None
        model = get_whisper_model(model_size)
        result = model.transcribe(audio_file, fp16=False)
        transcript = result["text"].strip()
        return transcript if transcript else None
    except Exception as e:
        print(f"Error transcribing audio {audio_file}: {e}")
        return None

def summarize_with_bart(text: str, max_length: int = 150, min_length: int = 30) -> str:
    """Summarize text using Facebook's BART model"""
    if not summarizer:
        return "Description unavailable (BART model not loaded)"
    try:
        text = text[:4000]
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False, truncation=True)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error generating summary with BART: {e}")
        return "Description unavailable due to error"

def generate_description(transcript: str, content_type: ContentType, title: Optional[str] = None) -> str:
    """Generate a brief description of the content using BART"""
    try:
        if content_type == ContentType.VIDEO:
            context = f"This is a YouTube video titled '{title}'. "
            prompt = context + transcript[:4000]
        elif content_type == ContentType.PLAYLIST:
            context = "This is a YouTube playlist containing videos about: "
            prompt = context + transcript[:4000]
        else:
            context = "This is a YouTube channel featuring content about: "
            prompt = context + transcript[:4000]
        description = summarize_with_bart(prompt, max_length=120, min_length=40)
        return description
    except Exception as e:
        print(f"Error generating description: {e}")
        return "Description unavailable due to error"

def update_job_status(job_id: str, status: JobStatus, total_items: int = None, processed_items: int = None):
    """Update job status in database with proper session handling"""
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


def process_video_background(video_url: str, job_id: str, model_size: str, store_in_db: bool, playlist_id: Optional[str] = None):
    """Process a single video in background"""
    try:
        video_id = extract_video_id(video_url)
        if not video_id:
            update_job_status(job_id, JobStatus.FAILED)
            return False
        
        title = get_youtube_title(video_url)
        audio_file = download_youtube_audio(video_id)
        if not audio_file:
            update_job_status(job_id, JobStatus.FAILED)
            return False
        
        transcript = transcribe_audio(audio_file, model_size)
        
        # Clean up audio file
        try:
            if os.path.exists(audio_file):
                os.remove(audio_file)
        except:
            pass
        
        if transcript:
            description = generate_description(transcript, ContentType.VIDEO, title)
            
            if store_in_db:
                db = next(get_db_session())
                try:
                    # Get the job to associate with transcript
                    job = db.query(TranscriptJob).filter(TranscriptJob.job_id == job_id).first()
                    if job:
                        transcript_obj = Transcript(
                            transcript_id=str(uuid.uuid4()),
                            video_id=video_id,
                            playlist_id=playlist_id,
                            title=title,
                            transcript_text=transcript,
                            word_count=len(transcript.split()),
                            description=description,
                            job_id=job.id
                        )
                        db.add(transcript_obj)
                        db.commit()
                finally:
                    db.close()
            
            # ✅ Mark job as COMPLETED once transcript is saved
            update_job_status(job_id, JobStatus.COMPLETED, total_items=1, processed_items=1)
            return True
        
        # If transcript failed
        update_job_status(job_id, JobStatus.FAILED)
        return False
        
    except Exception as e:
        print(f"Error processing video {video_url}: {e}")
        update_job_status(job_id, JobStatus.FAILED)
        return False



def process_playlist_background(playlist_url: str, job_id: str, model_size: str, store_in_db: bool):
    """Process playlist in background"""
    db = next(get_db_session())
    try:
        playlist_name = get_playlist_name(playlist_url)
        job = db.query(TranscriptJob).filter(TranscriptJob.job_id == job_id).first()
        if job:
            job.content_name = playlist_name
            db.commit()
        
        videos = get_playlist_videos_ytdlp(playlist_url)
        playlist_id = extract_playlist_id(playlist_url)
        
        if not videos:
            update_job_status(job_id, JobStatus.FAILED, 0, 0)
            return
        
        update_job_status(job_id, JobStatus.PROCESSING, len(videos), 0)
        
        processed_count = 0
        for i, video in enumerate(videos):
            success = process_video_background(video['url'], job_id, model_size, store_in_db, playlist_id)
            if success:
                processed_count += 1
            update_job_status(job_id, JobStatus.PROCESSING, len(videos), i + 1)
        
        update_job_status(job_id, JobStatus.COMPLETED, len(videos), processed_count)
        
    except Exception as e:
        print(f"Error processing playlist: {e}")
        update_job_status(job_id, JobStatus.FAILED)
    finally:
        db.close()
def process_channel_background(channel_url: str, job_id: str, model_size: str, store_in_db: bool):
    """Process channel in background"""
    try:
        # Get channel name
        channel_name = get_channel_name(channel_url)
        
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
        playlists = get_channel_playlists_ytdlp(channel_url)
        
        if not playlists:
            update_job_status(job_id, JobStatus.FAILED, 0, 0)
            return
        
        # Store playlist information
        playlist_info = []
        for playlist in playlists:
            playlist_info.append({
                'id': playlist['id'],
                'title': playlist['title'],
                'url': playlist['url']
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
        
        # Calculate total videos
        total_videos = 0
        for playlist in playlists:
            videos = get_playlist_videos_ytdlp(playlist['url'])
            total_videos += len(videos) if videos else 0
        
        update_job_status(job_id, JobStatus.PROCESSING, total_videos, 0)
        
        # Process videos from all playlists
        processed_count = 0
        for playlist in playlists:
            playlist_videos = get_playlist_videos_ytdlp(playlist['url'])
            playlist_id = playlist['id']
            
            if not playlist_videos:
                continue
                
            for video in playlist_videos:
                success = process_video_background(video['url'], job_id, model_size, store_in_db, playlist_id)
                if success:
                    processed_count += 1
                update_job_status(job_id, JobStatus.PROCESSING, total_videos, processed_count)
        
        update_job_status(job_id, JobStatus.COMPLETED, total_videos, processed_count)
        
    except Exception as e:
        print(f"Error processing channel: {e}")
        update_job_status(job_id, JobStatus.FAILED)

def create_transcript_job(db: Session, request: TranscriptRequest, user_id: int) -> TranscriptResponse:
    """Create a new transcript job"""
    try:
        url = request.youtube_url
        content_type = determine_content_type(url)
        
        # Get content name based on type
        content_name = None
        playlists = []
        
        if content_type == ContentType.VIDEO:
            content_name = get_youtube_title(url)
            message = "Started processing video"
            
        elif content_type == ContentType.PLAYLIST:
            content_name = get_playlist_name(url)
            message = "Started processing playlist"
            
        elif content_type == ContentType.CHANNEL:
            content_name = get_channel_name(url)
            playlists_data = get_channel_playlists_ytdlp(url)
            playlists = [{'id': p['id'], 'title': p['title'], 'url': p['url']} for p in playlists_data]
            message = "Started processing channel"
        
        # Create job in database
        job = TranscriptJob(
            url=url,
            type=content_type,
            status=JobStatus.PROCESSING,  # Set to PROCESSING immediately
            model_size=request.model_size,
            store_in_db=request.store_in_db,
            content_name=content_name,
            playlists=json.dumps(playlists) if playlists else None,
            user_id=user_id
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        
        # Start background processing (remove db parameter)
        if content_type == ContentType.VIDEO:
            background_executor.submit(
                process_video_background, 
                url, job.job_id, request.model_size, request.store_in_db
            )
            
        elif content_type == ContentType.PLAYLIST:
            background_executor.submit(
                process_playlist_background,
                url, job.job_id, request.model_size, request.store_in_db
            )
            
        elif content_type == ContentType.CHANNEL:
            background_executor.submit(
                process_channel_background,
                url, job.job_id, request.model_size, request.store_in_db
            )
        
        return TranscriptResponse(
            job_id=job.job_id,
            status=JobStatus.PROCESSING,  # Return PROCESSING status
            message=message,
            content_type=content_type,
            content_name=content_name,
            playlists=playlists
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def get_job_status_service(db: Session, job_id: str) -> JobStatusResponse:
    """Get job status and results"""
    # Refresh the session to get latest data
    db.expire_all()
    
    job = db.query(TranscriptJob).filter(TranscriptJob.job_id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    db.refresh(job)
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
    
    return JobStatusResponse(
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
    )