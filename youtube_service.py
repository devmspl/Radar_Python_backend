import os
import re
import uuid
import json
import time
import requests
from typing import List, Dict, Optional
from datetime import datetime
from fastapi import HTTPException
from sqlalchemy.orm import Session
from concurrent.futures import ThreadPoolExecutor

# Selenium imports for transcript fetching
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import pyperclip

from database import get_db_session
from models import TranscriptJob, Transcript, ContentType, JobStatus
from schemas import TranscriptRequest, TranscriptResponse, JobStatusResponse
from schemas import *
import utils

# ===========================
# GLOBALS
# ===========================
background_executor = ThreadPoolExecutor(max_workers=2)
COOKIES_FILE = "cookies.txt"
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# ===========================
# YOUTUBE DATA API v3 CLIENT
# ===========================
class YouTubeAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/youtube/v3"
    
    def _make_request(self, endpoint: str, params: dict) -> Optional[dict]:
        """Make request to YouTube Data API"""
        params['key'] = self.api_key
        try:
            response = requests.get(f"{self.base_url}/{endpoint}", params=params, timeout=30)
            if response.status_code == 404:
                print(f"YouTube API 404: {endpoint} - {params}")
                return None
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"YouTube API request failed: {e}")
            return None
    
    def get_video_title(self, video_id: str) -> Optional[str]:
        """Get video title using YouTube Data API"""
        data = self._make_request("videos", {
            'part': 'snippet',
            'id': video_id
        })
        
        if data and 'items' in data and len(data['items']) > 0:
            return data['items'][0]['snippet']['title']
        return None
    
    def get_playlist_info(self, playlist_id: str) -> Optional[dict]:
        """Get playlist information"""
        data = self._make_request("playlists", {
            'part': 'snippet',
            'id': playlist_id
        })
        
        if data and 'items' in data and len(data['items']) > 0:
            item = data['items'][0]['snippet']
            return {
                'title': item['title'],
                'description': item.get('description', '')
            }
        return None
    
    def get_playlist_videos(self, playlist_id: str, max_results: int = 50) -> List[Dict]:
        """Get all videos from a playlist"""
        videos = []
        page_token = None
        
        while len(videos) < max_results:
            params = {
                'part': 'snippet',
                'playlistId': playlist_id,
                'maxResults': min(50, max_results - len(videos))
            }
            if page_token:
                params['pageToken'] = page_token
            
            data = self._make_request("playlistItems", params)
            if not data or 'items' not in data:
                break
                
            for item in data['items']:
                if 'snippet' in item and 'resourceId' in item['snippet']:
                    video_id = item['snippet']['resourceId']['videoId']
                    videos.append({
                        'id': video_id,
                        'title': item['snippet']['title'],
                        'url': f"https://www.youtube.com/watch?v={video_id}"
                    })
            
            page_token = data.get('nextPageToken')
            if not page_token:
                break
        
        return videos
    
    def get_channel_info(self, channel_id: str) -> Optional[dict]:
        """Get channel information with proper handle (@username) support"""
        print(f"🎯 Fetching channel info for: {channel_id}")
    
        # If it's a handle (starts with @), we need to search for it first
        if channel_id.startswith('@'):
            print(f"🔍 Handling @username format: {channel_id}")
        
            # Remove the @ symbol for search
            handle = channel_id[1:]
            print(f"🔍 Searching for handle: {handle}")
        
            # Search for the channel by handle
            search_data = self._make_request("search", {
                'part': 'snippet',
                'type': 'channel',
                'q': handle,
                'maxResults': 5
            })
        
            print(f"📦 Search API response: {search_data}")
        
            if not search_data or 'items' not in search_data or len(search_data['items']) == 0:
                print(f"❌ No channels found for handle: {handle}")
                return None
        
            # Find the exact match for the handle
            exact_match = None
            for item in search_data['items']:
                snippet = item['snippet']
                channel_title = snippet.get('title', '')
                channel_handle = snippet.get('customUrl', '')
            
                print(f"🔍 Checking result: {channel_title} (handle: {channel_handle})")
            
                # Check if this is an exact match
                if (channel_handle and channel_handle.lower() == f"@{handle.lower()}") or \
                (channel_title and handle.lower() in channel_title.lower()):
                    exact_match = item
                    break
        
            # If no exact match found, use the first result
            if not exact_match:
                exact_match = search_data['items'][0]
                print(f"⚠️ No exact handle match found, using first result")
        
            channel_id = exact_match['id']['channelId']
            print(f"✅ Found channel ID for @{handle}: {channel_id}")
    
        # Now get channel info with the proper channel ID
        print(f"📡 Making channels API request for ID: {channel_id}")
        channels_data = self._make_request("channels", {
            'part': 'snippet',
            'id': channel_id
        })
    
        print(f"📦 Channels API response: {channels_data}")
    
        if channels_data and 'items' in channels_data and len(channels_data['items']) > 0:
            item = channels_data['items'][0]['snippet']
            result = {
                'title': item['title'],
                'description': item.get('description', '')
            }
            print(f"✅ Successfully fetched channel info: {result['title']}")
            return result
    
        print(f"❌ No channel found with ID: {channel_id}")
        return None
    
    def get_channel_playlists(self, channel_id: str, max_results: int = 50) -> List[Dict]:
        """Get all playlists from a channel with handle support"""
        playlists = []
        page_token = None
    
        print(f"🎯 Fetching playlists for channel: {channel_id}")
    
        # If it's a handle, first get the channel ID
        if channel_id.startswith('@'):
            print(f"🔍 Converting handle to channel ID: {channel_id}")
            channel_info = self.get_channel_info(channel_id)
            if not channel_info:
                print(f"❌ Could not get channel info for handle: {channel_id}")
                return []
        
            # We need to search again to get the actual channel ID
            handle = channel_id[1:]
            search_data = self._make_request("search", {
                'part': 'snippet',
                'type': 'channel',
                'q': handle,
                'maxResults': 1
            })
        
            if search_data and 'items' in search_data and len(search_data['items']) > 0:
                channel_id = search_data['items'][0]['id']['channelId']
                print(f"✅ Got channel ID for playlists: {channel_id}")
            else:
                print(f"❌ Could not find channel ID for handle: {channel_id}")
                return []
    
        print(f"📡 Fetching playlists for channel ID: {channel_id}")
    
        while len(playlists) < max_results:
            params = {
                'part': 'snippet',
                'channelId': channel_id,
                'maxResults': min(50, max_results - len(playlists))
            }
            if page_token:
                params['pageToken'] = page_token
        
            data = self._make_request("playlists", params)
            if not data or 'items' not in data:
                print(f"❌ No playlists data returned for channel: {channel_id}")
                break
            
            for item in data['items']:
                playlists.append({
                    'id': item['id'],
                    'title': item['snippet']['title'],
                    'description': item['snippet'].get('description', ''),
                    'url': f"https://www.youtube.com/playlist?list={item['id']}"
                })
        
            print(f"📚 Found {len(data['items'])} playlists in this page")
        
            page_token = data.get('nextPageToken')
            if not page_token:
                break
    
        print(f"✅ Total playlists found: {len(playlists)}")
        return playlists
    
    def search_channel_videos(self, channel_id: str, max_results: int = 20) -> List[Dict]:
        """Search for videos in a channel (fallback if no playlists)"""
        videos = []
        page_token = None
        
        while len(videos) < max_results:
            params = {
                'part': 'snippet',
                'channelId': channel_id,
                'type': 'video',
                'order': 'date',
                'maxResults': min(50, max_results - len(videos))
            }
            if page_token:
                params['pageToken'] = page_token
            
            data = self._make_request("search", params)
            if not data or 'items' not in data:
                break
                
            for item in data['items']:
                if 'id' in item and 'videoId' in item['id']:
                    videos.append({
                        'id': item['id']['videoId'],
                        'title': item['snippet']['title'],
                        'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}"
                    })
            
            page_token = data.get('nextPageToken')
            if not page_token:
                break
        
        return videos

# Initialize YouTube API client
youtube_api = YouTubeAPI(YOUTUBE_API_KEY) if YOUTUBE_API_KEY else None

# ===========================
# SELENIUM TRANSCRIPT FETCHER (Using your exact logic)
# ===========================
def copy_transcript(youtube_url, headless=True, max_retries=2):
    """Fetch transcript using Selenium and Tactiq with retry logic"""
    
    for retry_count in range(max_retries + 1):
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        
        # Performance optimizations
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-plugins")
        options.add_argument("--disable-images")
        options.add_argument("--blink-settings=imagesEnabled=false")
        
        # Timeout settings
        options.add_argument("--disable-features=VizDisplayCompositor")
        options.add_argument("--disable-background-timer-throttling")
        options.add_argument("--disable-renderer-backgrounding")
        options.add_argument("--disable-backgrounding-occluded-windows")

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)

        try:
            if retry_count > 0:
                print(f"🔄 Retry attempt {retry_count}/{max_retries} for: {youtube_url}")
            
            # Set timeouts
            driver.set_page_load_timeout(60)  # 60 seconds for page load
            driver.implicitly_wait(10)  # 10 seconds for element finding

            # 1. Open the Tactiq YouTube transcript tool
            print("Opening Tactiq transcript tool...")
            driver.get("https://tactiq.io/tools/youtube-transcript")

            wait = WebDriverWait(driver, 20)  # Increased wait time

            # 2. Find the input field
            print("Looking for input field...")
            input_selectors = [
                "#yt-2",
                "input[type='url']",
                "input[placeholder*='youtube']",
                "input[placeholder*='YouTube']",
                "input.youtube-url",
                "input#youtube-url"
            ]
            
            input_box = None
            for selector in input_selectors:
                try:
                    input_box = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                    print(f"Found input with selector: {selector}")
                    break
                except TimeoutException:
                    continue
            
            if not input_box:
                if retry_count < max_retries:
                    print("Input field not found, retrying...")
                    continue
                driver.save_screenshot("debug_input_not_found.png")
                raise Exception("Could not find input field")
            
            input_box.clear()
            input_box.send_keys(youtube_url)
            print("YouTube URL entered")

            # 3. Find and click the submit button
            print("Looking for submit button...")
            button_selectors = [
                "input.button-primary.w-button",
                "button[type='submit']",
                "input[type='submit']",
                "button.primary",
                ".w-button"
            ]
            
            submit_btn = None
            for selector in button_selectors:
                try:
                    submit_btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
                    print(f"Found button with selector: {selector}")
                    break
                except TimeoutException:
                    continue
            
            if not submit_btn:
                if retry_count < max_retries:
                    print("Submit button not found, retrying...")
                    continue
                driver.save_screenshot("debug_button_not_found.png")
                raise Exception("Could not find submit button")
            
            # Click using JavaScript for reliability
            driver.execute_script("arguments[0].click();", submit_btn)
            print("Submit button clicked")

            # 4. Wait for transcript to load with better waiting strategy
            print("Waiting for transcript to load...")
            
            # Wait for processing to complete
            transcript = wait_for_transcript_processing(driver, wait)
            
            if transcript:
                print(f"✅ Successfully extracted clean transcript ({len(transcript)} characters)")
                return transcript
            else:
                if retry_count < max_retries:
                    print("Transcript not found, retrying...")
                    continue
                print("❌ Could not extract transcript after all retries")
                return None

        except TimeoutException as e:
            print(f"⏰ Timeout error (attempt {retry_count + 1}/{max_retries + 1}): {e}")
            if retry_count < max_retries:
                print("Retrying after timeout...")
                continue
            else:
                print("Max retries exceeded for timeout")
                return None
                
        except Exception as e:
            print(f"❌ Error (attempt {retry_count + 1}/{max_retries + 1}): {e}")
            if retry_count < max_retries:
                print("Retrying after error...")
                continue
            else:
                print("Max retries exceeded")
                return None

        finally:
            driver.quit()
    
    return None


def wait_for_transcript_processing(driver, wait, max_wait=60):
    """Wait for transcript to process with better detection"""
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            # Check for loading indicators
            loading_selectors = [
                "//*[contains(text(), 'Loading')]",
                "//*[contains(text(), 'Processing')]",
                "//*[contains(@class, 'loading')]",
                "//*[contains(@class, 'spinner')]",
            ]
            
            # If loading indicators found, wait
            loading_found = False
            for selector in loading_selectors:
                try:
                    driver.find_element(By.XPATH, selector)
                    loading_found = True
                    break
                except:
                    continue
            
            if loading_found:
                print("⏳ Still processing...")
                time.sleep(3)
                continue
            
            # Try to extract transcript
            transcript = extract_clean_transcript_from_page(driver)
            if transcript:
                return transcript
            
            # Wait before next attempt
            time.sleep(2)
            
        except Exception as e:
            print(f"⚠️ Error during transcript wait: {e}")
            time.sleep(2)
    
    print("⏰ Timeout waiting for transcript processing")
    return None


def extract_clean_transcript_from_page(driver):
    """Extract and clean transcript from the page"""
    try:
        # Get all text from the page
        body_text = driver.find_element(By.TAG_NAME, "body").text
        
        # Clean the transcript
        clean_transcript = clean_transcript_text(body_text)
        
        if clean_transcript and len(clean_transcript.strip()) > 100:
            return clean_transcript.strip()
        
    except Exception as e:
        print(f"Error extracting transcript: {e}")
    
    return None


def clean_transcript_text(raw_text):
    """Clean the raw transcript text to remove timestamps and Tactiq headers"""
    if not raw_text:
        return None
    
    lines = raw_text.split('\n')
    clean_lines = []
    
    # Patterns to exclude
    exclude_patterns = [
        'Get started for FREE',
        'Get the transcript:',
        'YouTube Transcript Generator',
        'Focus on the Meeting',
        'Get live, in-meeting transcriptions',
        'Upload the transcript to Tactiq',
        '©.*Tactiq',
        'All rights reserved',
        'Made with',
        'Tactiq'
    ]
    
    found_transcript_start = False
    
    for line in lines:
        line = line.strip()
        
        if not line:
            continue
            
        # Skip excluded patterns
        if any(pattern in line for pattern in exclude_patterns):
            continue
            
        # Skip timestamp lines
        if re.match(r'^\d{2}:\d{2}:\d{2}\.\d{3}$', line) or re.match(r'^\d{2}:\d{2}:\d{2}$', line):
            continue
            
        # Skip very short lines
        if len(line) < 5:
            continue
            
        # Find start of actual content
        if not found_transcript_start:
            if (len(line) > 10 and 
                not any(keyword in line.lower() for keyword in ['transcript', 'get', 'free', 'started']) and
                not line.isupper()):
                found_transcript_start = True
                clean_lines.append(line)
        else:
            clean_lines.append(line)
    
    clean_text = '\n'.join(clean_lines)
    
    # Remove remaining timestamps
    clean_text = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3}', '', clean_text)
    clean_text = re.sub(r'\d{2}:\d{2}:\d{2}', '', clean_text)
    
    # Clean up whitespace
    clean_text = re.sub(r'\s+', ' ', clean_text)
    clean_text = clean_text.strip()
    
    return clean_text
# ===========================
# YOUTUBE HELPERS (Using YouTube Data API)
# ===========================
def determine_content_type(url: str) -> ContentType:
    """Determine if URL is a channel, playlist, or video"""
    if any(x in url for x in ["youtube.com/channel/", "youtube.com/c/", "youtube.com/user/", "youtube.com/@"]):
        return ContentType.CHANNEL
    elif "youtube.com/playlist" in url or "list=" in url:
        return ContentType.PLAYLIST
    elif "youtube.com/watch" in url or "youtu.be/" in url:
        return ContentType.VIDEO
    else:
        raise HTTPException(status_code=400, detail="Unsupported YouTube URL format")

def extract_video_id(url: str) -> Optional[str]:
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
    patterns = [r"list=([^&]+)", r"youtube\.com/playlist\?list=([^&]+)"]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def extract_channel_id(url: str) -> Optional[str]:
    """Extract channel ID from various YouTube URL formats with better parsing"""
    print(f"🔗 Parsing channel URL: {url}")
    
    patterns = [
        r"youtube\.com/channel/([a-zA-Z0-9_-]+)",
        r"youtube\.com/c/([a-zA-Z0-9_-]+)",
        r"youtube\.com/user/([a-zA-Z0-9_-]+)",
        r"youtube\.com/@([a-zA-Z0-9_.-]+)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            channel_id = match.group(1)
            print(f"✅ Extracted channel ID with pattern '{pattern}': {channel_id}")
            return channel_id
    
    print("❌ No channel ID pattern matched the URL")
    return None

def get_youtube_title(url: str) -> str:
    """Get video title using YouTube Data API"""
    if not youtube_api:
        raise HTTPException(status_code=500, detail="YouTube API not configured")
    
    video_id = extract_video_id(url)
    if video_id:
        title = youtube_api.get_video_title(video_id)
        if title:
            return title
    
    raise HTTPException(status_code=404, detail="Could not fetch video title")

def get_playlist_name(playlist_url: str) -> str:
    """Get playlist name using YouTube Data API"""
    if not youtube_api:
        raise HTTPException(status_code=500, detail="YouTube API not configured")
    
    playlist_id = extract_playlist_id(playlist_url)
    if playlist_id:
        info = youtube_api.get_playlist_info(playlist_id)
        if info:
            return info['title']
    
    raise HTTPException(status_code=404, detail="Could not fetch playlist info")

def get_channel_name(channel_url: str) -> str:
    """Get channel name using YouTube Data API with handle support"""
    if not youtube_api:
        print("❌ YouTube API not configured")
        raise HTTPException(status_code=500, detail="YouTube API not configured")
    
    print(f"🔍 Processing channel URL: {channel_url}")
    channel_id = extract_channel_id(channel_url)
    print(f"📋 Extracted channel identifier: {channel_id}")
    
    if not channel_id:
        print("❌ Could not extract channel identifier from URL")
        raise HTTPException(status_code=400, detail="Invalid channel URL format")
    
    try:
        # For handle URLs, we need to keep the @ symbol
        if channel_url.startswith('https://www.youtube.com/@'):
            channel_id = f"@{channel_id}"
            print(f"🔧 Using handle format: {channel_id}")
        
        info = youtube_api.get_channel_info(channel_id)
        print(f"📊 Channel API response: {info}")
        
        if info and info.get('title'):
            print(f"✅ Got channel name: {info['title']}")
            return info['title']
        else:
            print(f"❌ No channel info found for: {channel_id}")
            raise HTTPException(status_code=404, detail=f"Channel not found: {channel_url}")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error fetching channel info: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching channel information: {str(e)}")

def get_playlist_videos_api(playlist_url: str) -> List[Dict]:
    """Get playlist videos using YouTube Data API"""
    if not youtube_api:
        raise HTTPException(status_code=500, detail="YouTube API not configured")
    
    playlist_id = extract_playlist_id(playlist_url)
    if playlist_id:
        videos = youtube_api.get_playlist_videos(playlist_id)
        if videos:
            return videos
    
    raise HTTPException(status_code=404, detail="No videos found in playlist")

def get_channel_playlists_api(channel_url: str) -> List[Dict]:
    """Get channel playlists using YouTube Data API with handle support"""
    if not youtube_api:
        raise HTTPException(status_code=500, detail="YouTube API not configured")
    
    channel_id = extract_channel_id(channel_url)
    if not channel_id:
        raise HTTPException(status_code=400, detail="Invalid channel URL")
    
    print(f"🔍 Getting playlists for channel: {channel_url}")
    
    # For handle URLs, we need to keep the @ symbol
    if channel_url.startswith('https://www.youtube.com/@'):
        channel_id = f"@{channel_id}"
        print(f"🔧 Using handle format for playlists: {channel_id}")
    
    playlists = youtube_api.get_channel_playlists(channel_id)
    if playlists:
        print(f"✅ Found {len(playlists)} playlists")
        return playlists
    
    print(f"❌ No playlists found for channel: {channel_url}")
    raise HTTPException(status_code=404, detail="No playlists found in channel")
# ===========================
# PROCESSING FUNCTIONS
# ===========================
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
    """Process a single video in background with improved error handling"""
    try:
        video_id = extract_video_id(video_url)
        if not video_id:
            print(f"❌ Could not extract video ID from: {video_url}")
            return False
        
        title = get_youtube_title(video_url)
        print(f"🎬 Processing video: {title}")
        
        # Use Selenium to get transcript with retry logic
        transcript = copy_transcript(video_url, headless=True, max_retries=2)
        
        if not transcript:
            print(f"❌ Failed to fetch transcript for: {title}")
            return False
        
        # Additional validation
        if len(transcript.strip()) < 50:
            print(f"❌ Transcript too short for: {title}")
            return False
        
        print(f"✅ Successfully extracted transcript ({len(transcript)} characters)")
        
        if store_in_db:
            db = next(get_db_session())
            try:
                job = db.query(TranscriptJob).filter(TranscriptJob.job_id == job_id).first()
                if job:
                    transcript_obj = Transcript(
                        transcript_id=str(uuid.uuid4()),
                        video_id=video_id,
                        playlist_id=playlist_id,
                        title=title,
                        transcript_text=transcript,
                        word_count=len(transcript.split()),
                        description=None,
                        job_id=job.id
                    )
                    db.add(transcript_obj)
                    db.commit()
                    print(f"💾 Saved transcript to database for: {title}")
            except Exception as e:
                print(f"❌ Database error for {title}: {e}")
                db.rollback()
                return False
            finally:
                db.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing video {video_url}: {e}")
        return False
def process_playlist_background(playlist_url: str, job_id: str, model_size: str, store_in_db: bool):
    """Process playlist in background - process all videos in the playlist"""
    db = next(get_db_session())
    try:
        playlist_name = get_playlist_name(playlist_url)
        job = db.query(TranscriptJob).filter(TranscriptJob.job_id == job_id).first()
        if job:
            job.content_name = playlist_name
            db.commit()
        
        # Get all videos from the playlist
        videos = get_playlist_videos_api(playlist_url)
        playlist_id = extract_playlist_id(playlist_url)
        
        if not videos:
            print(f"❌ No videos found in playlist: {playlist_name}")
            update_job_status(job_id, JobStatus.FAILED, 0, 0)
            return
        
        print(f"🎯 Processing playlist '{playlist_name}' with {len(videos)} videos")
        
        # Update job with total items
        update_job_status(job_id, JobStatus.PROCESSING, len(videos), 0)
        
        processed_count = 0
        successful_count = 0
        
        # Process each video in the playlist
        for i, video in enumerate(videos):
            print(f"📹 Processing video {i+1}/{len(videos)}: {video['title']}")
            
            try:
                success = process_video_background(video['url'], job_id, model_size, store_in_db, playlist_id)
                if success:
                    successful_count += 1
                    print(f"✅ Successfully processed: {video['title']}")
                else:
                    print(f"❌ Failed to process: {video['title']}")
                
                processed_count += 1
                
                # Update progress
                update_job_status(job_id, JobStatus.PROCESSING, len(videos), processed_count)
                
                # Add a small delay between videos to avoid overwhelming the system
                time.sleep(2)
                
            except Exception as e:
                print(f"❌ Error processing video {video['title']}: {e}")
                processed_count += 1
                update_job_status(job_id, JobStatus.PROCESSING, len(videos), processed_count)
                continue
        
        # Final status update
        if successful_count > 0:
            print(f"🎉 Playlist processing completed: {successful_count}/{len(videos)} videos successful")
            update_job_status(job_id, JobStatus.COMPLETED, len(videos), successful_count)
        else:
            print(f"💥 Playlist processing failed: 0/{len(videos)} videos successful")
            update_job_status(job_id, JobStatus.FAILED, len(videos), 0)
        
    except Exception as e:
        print(f"❌ Error processing playlist: {e}")
        update_job_status(job_id, JobStatus.FAILED)
    finally:
        db.close()

def process_channel_background(channel_url: str, job_id: str, model_size: str, store_in_db: bool):
    """Process channel in background using only YouTube API and Selenium"""
    try:
        print(f"🏢 Starting channel processing: {channel_url}")
        
        # Get channel info
        channel_name = get_channel_name(channel_url)
        print(f"📝 Channel: {channel_name}")
        
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
        print("🔍 Fetching channel playlists...")
        playlists = get_channel_playlists_api(channel_url)
        
        if not playlists:
            print("❌ No playlists found in channel")
            update_job_status(job_id, JobStatus.FAILED, 0, 0)
            return
        
        print(f"📚 Found {len(playlists)} playlists")
        
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
        print("📊 Calculating total videos...")
        total_videos = 0
        playlist_videos_map = {}
        
        for playlist in playlists:
            playlist_id = playlist['id']
            playlist_title = playlist['title']
            
            print(f"   📋 Processing playlist: {playlist_title}")
            try:
                videos = get_playlist_videos_api(playlist['url'])
                if videos:
                    playlist_videos_map[playlist_id] = videos
                    total_videos += len(videos)
                    print(f"   ✅ Found {len(videos)} videos")
                else:
                    print(f"   ⚠️ No videos found in playlist")
            except Exception as e:
                print(f"   ❌ Failed to get videos for playlist: {e}")
        
        print(f"🎯 Total videos to process: {total_videos}")
        
        if total_videos == 0:
            print("❌ No videos found in any playlist")
            update_job_status(job_id, JobStatus.FAILED, 0, 0)
            return
        
        # Update job with total items
        update_job_status(job_id, JobStatus.PROCESSING, total_videos, 0)
        
        # Process videos from all playlists
        processed_count = 0
        successful_count = 0
        
        for playlist_index, playlist in enumerate(playlists, 1):
            playlist_id = playlist['id']
            playlist_title = playlist['title']
            videos = playlist_videos_map.get(playlist_id, [])
            
            if not videos:
                print(f"⏭️ Skipping empty playlist: {playlist_title}")
                continue
            
            print(f"\n🎬 Processing playlist {playlist_index}/{len(playlists)}: {playlist_title}")
            
            for video_index, video in enumerate(videos, 1):
                video_title = video['title']
                
                print(f"   🎥 Processing video {video_index}/{len(videos)}: {video_title}")
                
                try:
                    success = process_video_background(
                        video['url'], 
                        job_id, 
                        model_size, 
                        store_in_db, 
                        playlist_id
                    )
                    
                    if success:
                        successful_count += 1
                        print(f"   ✅ Successfully processed: {video_title}")
                    else:
                        print(f"   ❌ Failed to process: {video_title}")
                    
                    processed_count += 1
                    
                    # Update progress
                    update_job_status(job_id, JobStatus.PROCESSING, total_videos, processed_count)
                    
                    # Add delay between videos
                    time.sleep(5)
                    
                except Exception as e:
                    print(f"   💥 Error processing video {video_title}: {e}")
                    processed_count += 1
                    update_job_status(job_id, JobStatus.PROCESSING, total_videos, processed_count)
                    time.sleep(10)  # Longer delay after errors
                    continue
        
        # Final status
        print(f"\n📊 Processing complete: {successful_count}/{total_videos} successful")
        
        if successful_count > 0:
            update_job_status(job_id, JobStatus.COMPLETED, total_videos, successful_count)
            print(f"🎉 Channel processing completed successfully!")
        else:
            update_job_status(job_id, JobStatus.FAILED, total_videos, 0)
            print(f"💥 Channel processing failed")
        
    except Exception as e:
        print(f"❌ Critical error processing channel: {e}")
        update_job_status(job_id, JobStatus.FAILED)

# ===========================
# KEEP ALL OTHER EXISTING FUNCTIONS 
# ===========================
def create_transcript_job(db: Session, request: TranscriptRequest, user_id: int) -> TranscriptResponse:
    """Create a new transcript job with proper error handling"""
    try:
        print(f"🎬 Creating transcript job for URL: {request.youtube_url}")
        
        url = request.youtube_url
        content_type = determine_content_type(url)
        print(f"📝 Content type: {content_type}")
        
        # Get content name based on type
        content_name = None
        playlists = []
        message = ""
        
        try:
            if content_type == ContentType.VIDEO:
                content_name = get_youtube_title(url)
                message = "Started processing video"
                print(f"🎥 Video title: {content_name}")
                
            elif content_type == ContentType.PLAYLIST:
                content_name = get_playlist_name(url)
                message = "Started processing playlist"
                print(f"📚 Playlist name: {content_name}")
                
            elif content_type == ContentType.CHANNEL:
                content_name = get_channel_name(url)
                playlists_data = get_channel_playlists_api(url)
                playlists = [{'id': p['id'], 'title': p['title'], 'url': p['url']} for p in playlists_data]
                message = "Started processing channel"
                print(f"🏢 Channel name: {content_name}, Playlists: {len(playlists)}")
                
        except HTTPException as e:
            print(f"❌ Error getting content info: {e.detail}")
            raise e
        except Exception as e:
            print(f"❌ Unexpected error getting content info: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to get content information: {str(e)}")
        
        # Create job in database
        job = TranscriptJob(
            url=url,
            type=content_type,
            status=JobStatus.PROCESSING,
            model_size=request.model_size,
            store_in_db=request.store_in_db,
            content_name=content_name,
            playlists=json.dumps(playlists) if playlists else None,
            user_id=user_id
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        
        print(f"✅ Job created with ID: {job.job_id}")
        
        # Start background processing
        try:
            if content_type == ContentType.VIDEO:
                background_executor.submit(
                    process_video_background, 
                    url, job.job_id, request.model_size, request.store_in_db
                )
                print("🚀 Started video background processing")
                
            elif content_type == ContentType.PLAYLIST:
                background_executor.submit(
                    process_playlist_background,
                    url, job.job_id, request.model_size, request.store_in_db
                )
                print("🚀 Started playlist background processing")
                
            elif content_type == ContentType.CHANNEL:
                background_executor.submit(
                    process_channel_background,
                    url, job.job_id, request.model_size, request.store_in_db
                )
                print("🚀 Started channel background processing")
                
        except Exception as e:
            print(f"❌ Error starting background processing: {e}")
            # Update job status to failed
            job.status = JobStatus.FAILED
            db.commit()
            raise HTTPException(status_code=500, detail=f"Failed to start background processing: {str(e)}")
        
        return TranscriptResponse(
            job_id=job.job_id,
            status=JobStatus.PROCESSING,
            message=message,
            content_type=content_type,
            content_name=content_name,
            playlists=playlists
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"❌ Critical error in create_transcript_job: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Keep all other existing functions exactly as they were...
# [get_job_status_service, fetch_job_content, fetch_playlist_video, 
# fetch_channel_playlist_video, fetch_channel_by_id, fetch_playlist_by_id, 
# validate_cookie_file, etc.]

def get_job_status_service(db: Session, job_id: str) -> JobContentStatusResponse:
    job = db.query(TranscriptJob).filter(TranscriptJob.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    transcripts = db.query(Transcript).filter(Transcript.job_id == job.id).all()
    processed_items = job.processed_items or len(transcripts)
    total_items = job.total_items or (len(transcripts) if job.type == ContentType.VIDEO else 0)

    # Remove yt-dlp dependencies from content
    content = {}

    if job.type == ContentType.CHANNEL:
        playlists = json.loads(job.playlists or "[]")
        channel_content = []
        for pl in playlists:
            # Use YouTube API instead of yt-dlp
            try:
                pl_videos = get_playlist_videos_api(pl["url"])
                channel_content.append({
                    "id": pl["id"],
                    "title": pl["title"],
                    "videos": [v["title"] for v in pl_videos]
                })
            except:
                channel_content.append({
                    "id": pl["id"],
                    "title": pl["title"],
                    "videos": []
                })
        content = {"playlists": channel_content}

    elif job.type == ContentType.PLAYLIST:
        try:
            videos = get_playlist_videos_api(job.url)
            content = {
                "playlist_name": job.content_name,
                "videos": [v["title"] for v in videos]
            }
        except:
            content = {
                "playlist_name": job.content_name,
                "videos": []
            }

    elif job.type == ContentType.VIDEO:
        content = {"video_title": job.content_name}

    return JobContentStatusResponse(
        job_id=job.job_id,
        status=job.status,
        processed_items=processed_items,
        total_items=total_items,
        type=job.type,
    )



def fetch_job_content(db: Session, job_id: str, user_id: int):
    """Fetch channel playlists, playlist videos, or video transcript based on job type."""
    job = db.query(TranscriptJob).filter_by(job_id=job_id, user_id=user_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.type == ContentType.CHANNEL:
        playlists = get_channel_playlists_api(job.url)  # Use YouTube API
        enriched_playlists = []

        for pl in playlists:
            videos = get_playlist_videos_api(pl["url"])  # Use YouTube API
            video_items = [
                PlaylistVideo(
                    video_id=v["id"],
                    title=v["title"]
                )
                for v in videos
            ]
            enriched_playlists.append(
                ChannelPlaylist(
                    id=pl["id"],
                    title=pl["title"],
                    description=pl.get("description", ""),
                    videos=video_items
                )
            )

        return ChannelPlaylistsResponse(
            channel_id=extract_channel_id(job.url) or job.url,
            playlists=enriched_playlists
        )

    elif job.type == ContentType.PLAYLIST:
        playlist_id = extract_playlist_id(job.url)
        videos = get_playlist_videos_api(job.url)  # Use YouTube API

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
        video_title = get_youtube_title(job.url)  # Already uses YouTube API

        video_data = VideoWithTranscript(
            id=video_id,
            title=video_title,
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

    # Use YouTube API for title
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

    # Use YouTube API for title
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
    """Fetch channel metadata + playlists + videos using YouTube API only"""
    if not youtube_api:
        raise HTTPException(status_code=500, detail="YouTube API not configured")
    
    channel_id = extract_channel_id(channel_url)
    if not channel_id:
        raise HTTPException(status_code=400, detail="Invalid channel URL")
    
    # Get channel info
    channel_info = youtube_api.get_channel_info(channel_id)
    if not channel_info:
        raise HTTPException(status_code=404, detail="Channel not found")
    
    channel_title = channel_info.get("title") or "Untitled Channel"
    channel_description = channel_info.get("description") or ""
    
    # Get channel playlists
    playlists = youtube_api.get_channel_playlists(channel_id)
    enriched_playlists = []
    
    for pl in playlists:
        # Get videos for each playlist
        videos = youtube_api.get_playlist_videos(pl["id"])
        video_list = [
            PlaylistVideo(video_id=v["id"], title=v["title"]) for v in videos
        ]
        
        enriched_playlists.append(
            ChannelPlaylist(
                id=pl["id"],
                title=pl["title"],
                description=pl.get("description") or "",
                videos=video_list,
            )
        )
    
    return ChannelWithPlaylists(
        channel_id=channel_id,
        title=channel_title,
        description=channel_description,
        playlists=enriched_playlists,
    )


def fetch_playlist_by_id(playlist_url: str) -> PlaylistWithVideos:
    """Fetch playlist metadata + videos using YouTube API only"""
    if not youtube_api:
        raise HTTPException(status_code=500, detail="YouTube API not configured")
    
    playlist_id = extract_playlist_id(playlist_url)
    if not playlist_id:
        raise HTTPException(status_code=400, detail="Invalid playlist URL")
    
    # Get playlist info
    playlist_info = youtube_api.get_playlist_info(playlist_id)
    if not playlist_info:
        raise HTTPException(status_code=404, detail="Playlist not found")
    
    playlist_title = playlist_info.get("title") or f"Playlist {playlist_id}"
    playlist_description = playlist_info.get("description") or ""
    
    # Get playlist videos
    videos = youtube_api.get_playlist_videos(playlist_id)
    video_list = [PlaylistVideo(video_id=v["id"], title=v["title"]) for v in videos]
    
    return PlaylistWithVideos(
        playlist_id=playlist_id,
        title=playlist_title,
        description=playlist_description,
        videos=video_list,
    )