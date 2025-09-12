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
from schemas import *
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

# Path to exported YouTube cookies
COOKIES_FILE = "/var/www/client-projects/rohit/Radar_Python_backend/cookie.txt"

import subprocess
import time

def refresh_cookies():
    """Refresh cookies from Chrome and save to COOKIES_FILE"""
    print("🔄 Refreshing cookies from Chrome...")
    try:
        subprocess.run([
            "yt-dlp",
            "--cookies-from-browser", "chrome",
            "--cookies", COOKIES_FILE,
            "--skip-download",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # dummy video
        ], check=True)
        print(f"✅ Cookies refreshed successfully: {COOKIES_FILE}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to refresh cookies: {e}")

def get_yt_opts(extra_opts: Optional[dict] = None) -> dict:
    """Return yt-dlp options, auto-refresh cookies if missing"""
    opts = {
        "quiet": True,
        "no_warnings": True,
    }

    # Refresh cookies if missing or too old
    if not os.path.exists(COOKIES_FILE):
        print(f"⚠️ No cookies file found at {COOKIES_FILE}, refreshing...")
        refresh_cookies()
    else:
        # Optional: refresh if older than 6 hours
        max_age = 6 * 3600  # seconds
        age = time.time() - os.path.getmtime(COOKIES_FILE)
        if age > max_age:
            print(f"⚠️ Cookies file is older than {max_age/3600} hours, refreshing...")
            refresh_cookies()
        else:
            print(f"✅ Using cookies from {COOKIES_FILE}")

    opts["cookiefile"] = COOKIES_FILE

    if extra_opts:
        opts.update(extra_opts)
    return opts



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
            "extract_flat": True,
            "skip_download": True,
        }
        with yt_dlp.YoutubeDL(get_yt_opts(ydl_opts)) as ydl:
            info = ydl.extract_info(playlist_url, download=False)
            return info.get("title", "Unknown Playlist")
    except Exception as e:
        print(f"Error getting playlist name: {e}")
        return "Unknown Playlist"

def get_channel_name(channel_url: str) -> str:
    """Get channel name using yt-dlp"""
    try:
        ydl_opts = {
            "extract_flat": True,
            "skip_download": True,
        }
        with yt_dlp.YoutubeDL(get_yt_opts(ydl_opts)) as ydl:
            info = ydl.extract_info(channel_url, download=False)
            return info.get("title", "Unknown Channel") or info.get("uploader", "Unknown Channel")
    except Exception as e:
        print(f"Error getting channel name: {e}")
        return "Unknown Channel"

def get_youtube_title(url: str) -> str:
    """Extract YouTube video title using yt-dlp"""
    try:
        with yt_dlp.YoutubeDL(get_yt_opts()) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            title = info_dict.get("title")
            if not title:
                raise RuntimeError("yt-dlp returned no title (cookies may be invalid or expired)")
            return title
    except Exception as e:
        print(f"❌ Error getting YouTube title for {url}: {e}")
        if os.path.exists(COOKIES_FILE):
            print("   🔎 Cookies file exists, but might be expired or missing auth tokens (SID/HSID/SSID).")
        else:
            print("   ⚠️ No cookies file found at runtime.")
        return "Unknown Title"

def get_playlist_videos_ytdlp(playlist_url: str) -> List[Dict]:
    """Get videos from a playlist using yt-dlp"""
    ydl_opts = {
        "extract_flat": True,
        "skip_download": True,
        "ignoreerrors": True,
        "force_json": True,
    }
    try:
        with yt_dlp.YoutubeDL(get_yt_opts(ydl_opts)) as ydl:
            info = ydl.extract_info(playlist_url, download=False)
            videos = []
            if "entries" in info:
                for entry in info["entries"]:
                    if entry and entry.get("id"):
                        video_id = entry.get("id")
                        video_title = entry.get("title", "Unknown Title")
                        video_url = f"https://www.youtube.com/watch?v={video_id}"
                        videos.append({"id": video_id, "title": video_title, "url": video_url})
            return videos
    except Exception as e:
        print(f"Error getting playlist videos: {e}")
        return []

def get_channel_playlists_ytdlp(channel_url: str) -> List[Dict]:
    """Get all playlists from a YouTube channel using yt-dlp"""
    ydl_opts = {
        "extract_flat": True,
        "skip_download": True,
        "extractor_args": {"youtube": {"skip": ["webpage", "auth", "webpage", "webpage", "webpage"]}}
    }
    try:
        if "/playlists" not in channel_url:
            channel_url = channel_url.rstrip("/") + "/playlists"
        with yt_dlp.YoutubeDL(get_yt_opts(ydl_opts)) as ydl:
            info = ydl.extract_info(channel_url, download=False)
            playlists = []
            if "entries" in info:
                for entry in info["entries"]:
                    if entry and "url" in entry:
                        playlist_id = extract_playlist_id(entry["url"])
                        if playlist_id:
                            playlists.append({
                                "id": playlist_id,
                                "title": entry.get("title", f"Playlist {playlist_id}"),
                                "url": entry["url"]
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
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
        "outtmpl": os.path.join(output_dir, f"{video_id}.%(ext)s"),
    }
    
    try:
        with yt_dlp.YoutubeDL(get_yt_opts(ydl_opts)) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        if os.path.exists(output_file):
            return output_file
        else:
            for ext in ["wav", "mp3", "m4a", "webm"]:
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

def get_job_status_service(db: Session, job_id: str) -> JobContentStatusResponse:
    job = db.query(TranscriptJob).filter(TranscriptJob.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    transcripts = db.query(Transcript).filter(Transcript.job_id == job.id).all()
    processed_items = job.processed_items or len(transcripts)
    total_items = job.total_items or (len(transcripts) if job.type == ContentType.VIDEO else 0)

    content = {}

    if job.type == ContentType.CHANNEL:
        playlists = json.loads(job.playlists or "[]")
        channel_content = []
        for pl in playlists:
            pl_videos = get_playlist_videos_ytdlp(pl["url"])
            channel_content.append({
                "id": pl["id"],
                "title": pl["title"],
                "videos": [v["title"] for v in pl_videos]
            })
        content = {"playlists": channel_content}

    elif job.type == ContentType.PLAYLIST:
        videos = get_playlist_videos_ytdlp(job.url)
        content = {
            "playlist_name": job.content_name,
            "videos": [v["title"] for v in videos]
        }

    elif job.type == ContentType.VIDEO:
        content = {"video_title": job.content_name}

    return JobContentStatusResponse(
        job_id=job.job_id,
        status=job.status,
        processed_items=processed_items,
        total_items=total_items,
        type=job.type,
        # content=content
    )


def fetch_job_content(db: Session, job_id: str, user_id: int):
    """Fetch channel playlists, playlist videos, or video transcript based on job type."""
    job = db.query(TranscriptJob).filter_by(job_id=job_id, user_id=user_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.type == ContentType.CHANNEL:
        playlists = get_channel_playlists_ytdlp(job.url)
        enriched_playlists = []

        for pl in playlists:
            videos = get_playlist_videos_ytdlp(pl["url"])
            video_items = [
                PlaylistVideo(
                    video_id=v["id"],   # ✅ map correctly
                    title=v["title"]
                )
                for v in videos
            ]
            enriched_playlists.append(
                ChannelPlaylist(
                    id=pl["id"],
                    title=pl["title"],
                    description=pl.get("description"),
                    videos=video_items
                )
            )

        return ChannelPlaylistsResponse(
            channel_id=job.url,
            playlists=enriched_playlists
        )

    elif job.type == ContentType.PLAYLIST:
        playlist_id = extract_playlist_id(job.url)
        videos = get_playlist_videos_ytdlp(job.url)

        video_items = [
            PlaylistVideo(
                video_id=v["id"],
                title=v["title"]
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
        video_title = get_youtube_title(job.url)

        video_data = VideoWithTranscript(   # ✅ use Pydantic model instead of dict
            id=video_id,
            title=video_title,
            # description=transcript_obj.description if transcript_obj else None,
            # duration=(
            #     int(transcript_obj.duration)
            #     if transcript_obj and transcript_obj.duration and str(transcript_obj.duration).isdigit()
            #     else None
            # ),
            # transcript=transcript_obj.transcript_text if transcript_obj else None,
        )

        return VideoTranscriptResponse(
            video_id=video_id,
            video=video_data
        )

    else:
        raise HTTPException(status_code=400, detail="Unsupported content type")


def fetch_playlist_video(db: Session, job_id: str, video_id: str, user_id: int):
    """Fetch a single video from a playlist with transcript, description, and duration."""
    job = db.query(TranscriptJob).filter_by(job_id=job_id, user_id=user_id).first()
    if not job or job.type != ContentType.PLAYLIST:
        raise HTTPException(status_code=404, detail="Playlist job not found")

    # Get transcript for this video
    transcript_obj = (
        db.query(Transcript)
        .filter_by(job_id=job.id, video_id=video_id)
        .first()
    )

    # Get video title via yt-dlp if transcript not in DB yet
    title = get_youtube_title(f"https://www.youtube.com/watch?v={video_id}")

    return {
        "video_id": video_id,
        "title": title,
        "description": transcript_obj.description if transcript_obj else None,
        "duration": (
            int(transcript_obj.duration)
            if transcript_obj and transcript_obj.duration and str(transcript_obj.duration).isdigit()
            else None
        ),
        "transcript": transcript_obj.transcript_text if transcript_obj else None,
    }
def fetch_channel_playlist_video(db: Session, job_id: str, playlist_id: str, video_id: str, user_id: int):
    """Fetch a video from a channel's playlist with transcript, description, and duration."""
    job = db.query(TranscriptJob).filter_by(job_id=job_id, user_id=user_id).first()
    if not job or job.type != ContentType.CHANNEL:
        raise HTTPException(status_code=404, detail="Channel job not found")

    # Find transcript if already processed
    transcript_obj = (
        db.query(Transcript)
        .filter_by(job_id=job.id, video_id=video_id, playlist_id=playlist_id)
        .first()
    )

    # Fallback: fetch metadata with yt-dlp
    title = get_youtube_title(f"https://www.youtube.com/watch?v={video_id}")

    return {
        "video_id": video_id,
        "title": title,
        "description": transcript_obj.description if transcript_obj else None,
        "duration": (
            int(transcript_obj.duration)
            if transcript_obj and transcript_obj.duration and str(transcript_obj.duration).isdigit()
            else None
        ),
        "transcript": transcript_obj.transcript_text if transcript_obj else None,
    }
def fetch_channel_by_id(channel_url: str) -> ChannelWithPlaylists:
    opts = get_yt_opts()
    info = yt_dlp.YoutubeDL(opts).extract_info(channel_url, download=False)

    channel_title = info.get("title") or "Untitled Channel"
    channel_description = info.get("description") or ""

    playlists = get_channel_playlists_ytdlp(channel_url)
    enriched_playlists = []

    for pl in playlists:
        videos = get_playlist_videos_ytdlp(pl["url"])
        enriched_playlists.append(
            ChannelPlaylist(
                id=pl["id"],
                title=pl["title"],
                description=pl.get("description") or "",
                videos=[
                    PlaylistVideo(video_id=v["id"], title=v["title"]) for v in videos
                ],
            )
        )

    return ChannelWithPlaylists(
        channel_id=channel_url,
        title=channel_title,
        description=channel_description,
        playlists=enriched_playlists,
    )


def fetch_playlist_by_id(playlist_url: str) -> PlaylistWithVideos:
    playlist_id = extract_playlist_id(playlist_url)

    opts = {
        "quiet": True,
        "cookies": COOKIES_FILE,
    }

    # fetch metadata with yt-dlp
    info = yt_dlp.YoutubeDL(opts).extract_info(playlist_url, download=False)
    playlist_title = info.get("title")
    playlist_description = info.get("description")

    videos = get_playlist_videos_ytdlp(playlist_url)

    return PlaylistWithVideos(
        playlist_id=playlist_id,
        title=playlist_title,
        description=playlist_description,
        videos=[PlaylistVideo(video_id=v["id"], title=v["title"]) for v in videos],
    )

