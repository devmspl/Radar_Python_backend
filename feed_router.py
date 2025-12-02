from fastapi import APIRouter, Depends, HTTPException, Response, BackgroundTasks
from sqlalchemy.orm import Session, joinedload
from datetime import datetime
import os
import re
import logging
import json
from typing import List, Optional, Dict, Any
from database import get_db
from models import Blog, Category, Feed, Slide, Transcript, TranscriptJob, Source, PublishedFeed, FilterType
from openai import OpenAI, APIError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential
from schemas import FeedRequest, DeleteSlideRequest, YouTubeFeedRequest
from sqlalchemy import or_, String
import requests
from enum import Enum

router = APIRouter(prefix="/get", tags=["Feeds"])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None
    logger.warning("OpenAI API key not found. AI features will not work.")

class ContentType(str, Enum):
    WEBINAR = "Webinar"
    BLOG = "Blog"
    PODCAST = "Podcast"
    VIDEO = "Video"

# ------------------ Helper Functions ------------------

def validate_openai_client():
    """Validate OpenAI client is available and has quota."""
    if not client:
        raise HTTPException(status_code=503, detail="OpenAI service not configured")
    
    # Check if API key exists (basic check)
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-") and len(OPENAI_API_KEY) < 10:
        raise HTTPException(status_code=503, detail="OpenAI API key not configured properly")


def check_openai_availability() -> Dict[str, Any]:
    """Check if OpenAI API is available and has quota."""
    if not OPENAI_API_KEY:
        return {"available": False, "reason": "OpenAI API key not configured"}
    
    try:
        # Try a simple, cheap API call to check quota
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'test'"}],
            max_tokens=5,
            timeout=10
        )
        return {"available": True, "reason": "OpenAI API is working"}
    except RateLimitError as e:
        logger.error(f"OpenAI quota exceeded: {e}")
        return {"available": False, "reason": f"OpenAI quota exceeded: {str(e)}"}
    except APIError as e:
        logger.error(f"OpenAI API error: {e}")
        return {"available": False, "reason": f"OpenAI API error: {str(e)}"}
    except Exception as e:
        logger.error(f"OpenAI check failed: {e}")
        return {"available": False, "reason": f"OpenAI check failed: {str(e)}"}


def get_youtube_channel_info(video_id: str) -> Dict[str, Any]:
    """Get YouTube channel information using YouTube Data API."""
    if not YOUTUBE_API_KEY:
        logger.warning("YouTube API key not found. Using fallback channel info.")
        return {
            "channel_name": "YouTube Creator",
            "channel_id": None,
            "available": False
        }
    
    try:
        # First, get video details to get channel ID
        video_url = f"https://www.googleapis.com/youtube/v3/videos"
        params = {
            "key": YOUTUBE_API_KEY,
            "id": video_id,
            "part": "snippet"
        }
        
        response = requests.get(video_url, params=params, timeout=10)
        if response.status_code != 200:
            logger.warning(f"Failed to get video info: {response.status_code}")
            return {
                "channel_name": "YouTube Creator",
                "channel_id": None,
                "available": False
            }
        
        video_data = response.json()
        if not video_data.get("items"):
            return {
                "channel_name": "YouTube Creator",
                "channel_id": None,
                "available": False
            }
        
        snippet = video_data["items"][0]["snippet"]
        channel_id = snippet.get("channelId")
        channel_title = snippet.get("channelTitle", "YouTube Creator")
        
        # Get channel details
        if channel_id:
            channel_url = f"https://www.googleapis.com/youtube/v3/channels"
            channel_params = {
                "key": YOUTUBE_API_KEY,
                "id": channel_id,
                "part": "snippet,statistics"
            }
            
            channel_response = requests.get(channel_url, params=channel_params, timeout=10)
            if channel_response.status_code == 200:
                channel_data = channel_response.json()
                if channel_data.get("items"):
                    channel_info = channel_data["items"][0]["snippet"]
                    stats = channel_data["items"][0].get("statistics", {})
                    
                    return {
                        "channel_name": channel_title,
                        "channel_id": channel_id,
                        "description": channel_info.get("description", ""),
                        "thumbnail": channel_info.get("thumbnails", {}).get("default", {}).get("url", ""),
                        "subscriber_count": stats.get("subscriberCount", "0"),
                        "video_count": stats.get("videoCount", "0"),
                        "available": True
                    }
        
        return {
            "channel_name": channel_title,
            "channel_id": channel_id,
            "available": True
        }
        
    except Exception as e:
        logger.error(f"Error fetching YouTube channel info: {e}")
        return {
            "channel_name": "YouTube Creator",
            "channel_id": None,
            "available": False
        }

def handle_openai_error(e: Exception) -> None:
    """Handle OpenAI errors and raise appropriate HTTPException."""
    if isinstance(e, RateLimitError):
        logger.error(f"OpenAI rate limit/quota exceeded: {e}")
        raise HTTPException(
            status_code=429,
            detail=f"OpenAI quota exceeded. Please check your OpenAI billing and quota. Error: {str(e)}"
        )
    elif isinstance(e, APIError):
        logger.error(f"OpenAI API error: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"OpenAI API error: {str(e)}"
        )
    else:
        logger.error(f"OpenAI error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"OpenAI processing failed: {str(e)}"
        )

# ------------------ AI Categorization Function ------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry_error_callback=lambda retry_state: handle_openai_error(retry_state.outcome.exception())
)
def categorize_content_with_openai(content: str, admin_categories: list) -> tuple:
    """Categorize content using OpenAI and extract skills, tools, roles."""
    try:
        validate_openai_client()
        
        truncated_content = content[:4000] + "..." if len(content) > 4000 else content
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": """You are a content analysis assistant. Analyze the content and:
                    1. Categorize it into the provided categories
                    2. Extract relevant skills mentioned
                    3. Extract tools/technologies mentioned  
                    4. Extract relevant job roles
                    5. Determine content type (Blog, Video, Podcast, Webinar)
                    
                    Return JSON with this structure:
                    {
                        "categories": ["category1", "category2"],
                        "skills": ["skill1", "skill2"],
                        "tools": ["tool1", "tool2"],
                        "roles": ["role1", "role2"],
                        "content_type": "Blog/Video/Podcast/Webinar"
                    }"""
                },
                {
                    "role": "user",
                    "content": f"Available categories: {', '.join(admin_categories)}.\n\nContent:\n{truncated_content}\n\nReturn JSON with categories, skills, tools, roles, and content_type."
                }
            ],
            temperature=0.1,
            max_tokens=500,
            response_format={ "type": "json_object" }
        )
        
        analysis = json.loads(response.choices[0].message.content.strip())
        
        # Validate categories
        categories = analysis.get("categories", [])
        matched_categories = []
        for item in categories:
            for category in admin_categories:
                if (item.lower() == category.lower() or 
                    item.lower() in category.lower() or 
                    category.lower() in item.lower()):
                    matched_categories.append(category)
                    break
        
        # Remove duplicates and limit
        seen = set()
        unique_categories = []
        for cat in matched_categories:
            if cat not in seen:
                seen.add(cat)
                unique_categories.append(cat)
        
        if not unique_categories:
            # Use first available category as fallback
            unique_categories = [admin_categories[0]] if admin_categories else ["Uncategorized"]
        
        # Extract other metadata
        skills = list(set(analysis.get("skills", [])))[:10]
        tools = list(set(analysis.get("tools", [])))[:10]
        roles = list(set(analysis.get("roles", [])))[:10]
        
        # Determine content type
        content_type_str = analysis.get("content_type", "Blog")
        content_type = ContentType.BLOG  # default
        
        if "video" in content_type_str.lower() or "youtube" in content_type_str.lower():
            content_type = ContentType.VIDEO
        elif "podcast" in content_type_str.lower():
            content_type = ContentType.PODCAST
        elif "webinar" in content_type_str.lower():
            content_type = ContentType.WEBINAR
        
        return unique_categories[:3], skills, tools, roles, content_type
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OpenAI categorization error: {e}")
        handle_openai_error(e)

# ------------------ AI Content Generation Functions ------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry_error_callback=lambda retry_state: handle_openai_error(retry_state.outcome.exception())
)
def generate_feed_content_with_ai(title: str, content: str, categories: List[str], content_type: str = "blog") -> Dict[str, Any]:
    """Generate engaging feed content using AI."""
    try:
        validate_openai_client()
        
        truncated_content = content[:6000] + "..." if len(content) > 6000 else content
        
        system_prompt = {
            "blog": """You are a content summarization and presentation expert. Create an engaging, structured feed from blog content. 
            
            Return JSON with the following structure:
            {
                "title": "Engaging title that captures the essence",
                "summary": "2-3 paragraph comprehensive summary of the main content",
                "key_points": ["Point 1", "Point 2", "Point 3", "Point 4", "Point 5"],
                "conclusion": "1-2 paragraph concluding thoughts or key takeaways"
            }""",
            
            "transcript": """You are a video content summarization expert. Create an engaging, structured feed from video transcript content.
            
            Return JSON with the following structure:
            {
                "title": "Engaging title that captures the video's essence", 
                "summary": "2-3 paragraph summary of the video's main content and message",
                "key_points": ["Key insight 1", "Key insight 2", "Key insight 3", "Key insight 4", "Key insight 5"],
                "conclusion": "1-2 paragraph conclusion with main takeaways and actionable advice"
            }"""
        }
        
        user_prompt = f"""
        Original Title: {title}
        Content Type: {content_type}
        Categories: {', '.join(categories)}
        
        Content:
        {truncated_content}
        
        Please generate engaging, structured feed content in the specified JSON format.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": system_prompt.get(content_type, system_prompt["blog"])
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.7,
            max_tokens=2000,
            response_format={ "type": "json_object" }
        )
        
        content = response.choices[0].message.content
        ai_content = json.loads(content)
        
        # Validate required fields
        required_fields = ["title", "summary", "key_points", "conclusion"]
        for field in required_fields:
            if field not in ai_content:
                raise HTTPException(status_code=500, detail=f"OpenAI response missing required field: {field}")
        
        return ai_content
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OpenAI feed generation error: {e}")
        handle_openai_error(e)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry_error_callback=lambda retry_state: handle_openai_error(retry_state.outcome.exception())
)
def generate_slide_with_ai(slide_type: str, context: str, categories: List[str], content_type: str = "blog", previous_slides: List[Dict] = None) -> Dict[str, Any]:
    """Generate a specific slide type using AI."""
    try:
        validate_openai_client()
        
        system_prompt = {
            "role": "system", 
            "content": f"""You are a presentation design expert. Create engaging, concise slides.
            Return ONLY valid JSON with this structure:
            {{
                "title": "Engaging slide title",
                "body": "Substantive body content that summarizes key points",
                "bullets": ["Bullet point 1", "Bullet point 2", "Bullet point 3"]
            }}"""
        }
        
        messages = [
            system_prompt,
            {
                "role": "user",
                "content": f"Context: {context}\nCategories: {', '.join(categories)}\nGenerate engaging slide content as valid JSON with title, body, and bullets array."
            }
        ]
        
        if previous_slides:
            previous_titles = [f"Slide {s.get('order')}: {s.get('title', '')}" for s in previous_slides[-2:]]
            messages[1]["content"] += f"\nPrevious slides (avoid repetition): {', '.join(previous_titles)}"
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=1200,
            response_format={ "type": "json_object" }
        )
        
        content = response.choices[0].message.content
        slide_data = json.loads(content)
        
        # Validate content quality
        title = slide_data.get("title", "").strip()
        body = slide_data.get("body", "").strip()
        bullets = slide_data.get("bullets", [])
        
        if not title or len(body) < 50:
            # Generate minimal slide as fallback
            return {
                "title": f"{content_type} Insights",
                "body": "Content processing requires OpenAI API access. Please check your API key and quota.",
                "bullets": ["AI processing not available"],
                "background_color": "#FFFFFF",
                "source_refs": [],
                "render_markdown": True
            }
        
        return {
            "title": title,
            "body": body,
            "bullets": bullets,
            "background_color": "#FFFFFF",
            "source_refs": [],
            "render_markdown": True
        }
        
    except Exception as e:
        logger.error(f"OpenAI slide generation error for {slide_type}: {str(e)}")
        handle_openai_error(e)

def generate_slides_with_ai(title: str, content: str, ai_generated_content: Dict[str, Any], categories: List[str], content_type: str = "blog") -> List[Dict]:
    """Generate presentation slides using AI."""
    import random
    import hashlib
    
    # Determine slide count based on content richness (1-10 slides)
    content_length = len(content)
    key_points_count = len(ai_generated_content.get("key_points", []))
    summary_length = len(ai_generated_content.get("summary", ""))
    
    richness_score = min(
        (content_length / 1000) +
        (key_points_count * 0.5) +
        (summary_length / 500),
        10.0
    )
    
    base_slides = max(1, min(10, int(richness_score) + random.randint(-1, 2)))
    slide_count = max(1, min(10, base_slides))
    
    logger.info(f"Generating {slide_count} slides for '{title}'")
    
    # Generate background color for this feed
    background_color = generate_feed_background_color(title, content_type, categories)
    
    slides = []
    
    try:
        # 1. Title slide
        title_context = f"Create an engaging title slide for: {title}\nSummary: {ai_generated_content.get('summary', '')}\nCategories: {', '.join(categories)}\nType: {content_type}"
        title_slide = generate_slide_with_ai("title", title_context, categories, content_type)
        title_slide["order"] = 1
        title_slide["background_color"] = background_color
        slides.append(title_slide)
        
        if slide_count == 1:
            return slides
        
        # 2. Summary slide
        summary_context = f"Create a comprehensive summary slide for: {title}\nFull Summary: {ai_generated_content.get('summary', '')}\nType: {content_type}"
        summary_slide = generate_slide_with_ai("summary", summary_context, categories, content_type, slides)
        summary_slide["order"] = 2
        summary_slide["background_color"] = background_color
        slides.append(summary_slide)
        
        if slide_count == 2:
            return slides
        
        # 3. Key point slides
        key_points = ai_generated_content.get("key_points", [])
        remaining_slides = slide_count - 2
        key_point_slides_count = min(remaining_slides, len(key_points), 5)
        
        for i in range(key_point_slides_count):
            if i < len(key_points):
                point = key_points[i]
                key_point_context = f"Create a detailed key point slide for: {title}\nKey Point: {point}\nOther Key Points: {', '.join(key_points[:i] + key_points[i+1:][:2])}\nType: {content_type}"
                key_slide = generate_slide_with_ai("key_point", key_point_context, categories, content_type, slides)
                key_slide["order"] = len(slides) + 1
                key_slide["background_color"] = background_color
                slides.append(key_slide)
        
        # 4. Conclusion slide if space
        if len(slides) < slide_count and ai_generated_content.get("conclusion"):
            conclusion_context = f"Create a conclusion slide for: {title}\nConclusion: {ai_generated_content.get('conclusion', '')}\nKey Points Covered: {', '.join(key_points[:3])}\nType: {content_type}"
            conclusion_slide = generate_slide_with_ai("conclusion", conclusion_context, categories, content_type, slides)
            conclusion_slide["order"] = len(slides) + 1
            conclusion_slide["background_color"] = background_color
            slides.append(conclusion_slide)
        
        # 5. Fill remaining slots
        while len(slides) < slide_count:
            insight_context = f"Create an additional insights slide for: {title}\nAvailable Content: Summary - {ai_generated_content.get('summary', '')[:200]}...\nRemaining Key Points: {', '.join(key_points[len(slides)-2:]) if len(slides)-2 < len(key_points) else 'Various important aspects'}\nType: {content_type}"
            insight_slide = generate_slide_with_ai("additional_insights", insight_context, categories, content_type, slides)
            insight_slide["order"] = len(slides) + 1
            insight_slide["background_color"] = background_color
            slides.append(insight_slide)
        
        logger.info(f"Successfully generated {len(slides)} slides for '{title}'")
        return slides[:slide_count]
        
    except HTTPException as e:
        # If OpenAI fails, create minimal slides
        logger.error(f"Error in generate_slides_with_ai for '{title}': {e.detail}")
        
        # Create minimal fallback slides
        fallback_slides = [
            {
                "order": 1,
                "title": title,
                "body": "Feed generation requires OpenAI API access. Please check your API key and quota.",
                "bullets": ["AI processing failed"],
                "background_color": "#FFFFFF",
                "source_refs": [],
                "render_markdown": True
            }
        ]
        
        if slide_count > 1:
            fallback_slides.append({
                "order": 2,
                "title": "Content Unavailable",
                "body": "The content could not be processed due to API limitations.",
                "bullets": ["Check OpenAI API configuration"],
                "background_color": "#FFFFFF",
                "source_refs": [],
                "render_markdown": True
            })
        
        return fallback_slides[:slide_count]
    except Exception as e:
        logger.error(f"Unexpected error in generate_slides_with_ai for '{title}': {str(e)}")
        raise

def generate_feed_background_color(title: str, content_type: str, categories: List[str]) -> str:
    """Generate background color from predefined distinct colors."""
    import hashlib
    
    distinct_colors = [
        "#1e3a8a", "#166534", "#581c87", "#991b1b", "#9a3412",
        "#115e59", "#831843", "#854d0e", "#3730a3", "#064e3b",
        "#1d4ed8", "#15803d", "#7c3aed", "#dc2626", "#ea580c",
        "#0f766e", "#be185d", "#ca8a04", "#4f46e5", "#047857",
    ]
    
    seed_string = f"{title}_{content_type}_{'_'.join(sorted(categories))}"
    hash_object = hashlib.md5(seed_string.encode())
    hash_int = int(hash_object.hexdigest()[:8], 16)
    
    color_index = hash_int % len(distinct_colors)
    color_hex = distinct_colors[color_index]
    
    logger.info(f"Generated predefined distinct color: {color_hex} for '{title}'")
    return color_hex

# ------------------ Core Feed Creation Functions ------------------

def create_feed_from_blog(db: Session, blog: Blog):
    """Generate feed and slides from a blog using AI."""
    try:
        # Check if feed already exists for this blog
        existing_feed = db.query(Feed).filter(Feed.blog_id == blog.id).first()
        if existing_feed:
            db.query(Slide).filter(Slide.feed_id == existing_feed.id).delete()
            db.flush()
        
        # Create new feed with enhanced metadata
        admin_categories = [c.name for c in db.query(Category).filter(Category.is_active == True).all()]
        if not admin_categories:
            # Use fallback categories
            admin_categories = ["Uncategorized"]
        
        try:
            categories, skills, tools, roles, content_type = categorize_content_with_openai(blog.content, admin_categories)
            ai_generated_content = generate_feed_content_with_ai(blog.title, blog.content, categories, "blog")
            slides_data = generate_slides_with_ai(blog.title, blog.content, ai_generated_content, categories, "blog")
            
            feed_title = ai_generated_content.get("title", blog.title)
            status = "ready"
        except HTTPException as e:
            # If OpenAI fails, create minimal feed
            logger.warning(f"OpenAI failed for blog {blog.id}, creating minimal feed: {e.detail}")
            categories = ["Uncategorized"]
            skills = []
            tools = []
            roles = []
            content_type = ContentType.BLOG
            ai_generated_content = {
                "title": blog.title,
                "summary": "Content processing requires OpenAI API access. Please check your API key and quota.",
                "key_points": ["AI processing not available"],
                "conclusion": "Unable to generate content summary due to API limitations."
            }
            slides_data = generate_slides_with_ai(blog.title, blog.content, ai_generated_content, categories, "blog")
            feed_title = blog.title
            status = "partial"
        
        feed = Feed(
            blog_id=blog.id, 
            title=feed_title,
            categories=categories, 
            skills=skills,
            tools=tools,
            roles=roles,
            content_type=content_type,
            status=status,
            ai_generated_content=ai_generated_content,
            image_generation_enabled=False,
            source_type="blog",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(feed)
        db.flush()
        
        for slide_data in slides_data:
            slide = Slide(
                feed_id=feed.id,
                order=slide_data["order"],
                title=slide_data["title"],
                body=slide_data["body"],
                bullets=slide_data.get("bullets"),
                background_color=slide_data.get("background_color", "#FFFFFF"),
                source_refs=slide_data.get("source_refs", []),
                render_markdown=int(slide_data.get("render_markdown", True)),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.add(slide)
        
        db.commit()
        db.refresh(feed)
        logger.info(f"Successfully created AI-generated feed {feed.id} for blog {blog.id}")
        return feed
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating AI-generated feed for blog {blog.id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create feed for blog: {str(e)}")

def create_feed_from_transcript(db: Session, transcript: Transcript, overwrite: bool = False):
    """Generate feed and slides from a YouTube transcript using AI."""
    try:
        existing_feed = db.query(Feed).filter(Feed.transcript_id == transcript.transcript_id).first()
        
        if existing_feed and not overwrite:
            logger.info(f"Feed already exists for transcript {transcript.transcript_id}, skipping")
            return existing_feed
            
        admin_categories = [c.name for c in db.query(Category).filter(Category.is_active == True).all()]
        if not admin_categories:
            admin_categories = ["Uncategorized"]
        
        try:
            categories, skills, tools, roles, content_type = categorize_content_with_openai(transcript.transcript_text, admin_categories)
            ai_generated_content = generate_feed_content_with_ai(transcript.title, transcript.transcript_text, categories, "transcript")
            slides_data = generate_slides_with_ai(transcript.title, transcript.transcript_text, ai_generated_content, categories, "transcript")
            
            feed_title = ai_generated_content.get("title", transcript.title)
            status = "ready"
        except HTTPException as e:
            # If OpenAI fails, create minimal feed
            logger.warning(f"OpenAI failed for transcript {transcript.transcript_id}, creating minimal feed: {e.detail}")
            categories = ["Uncategorized"]
            skills = []
            tools = []
            roles = []
            content_type = ContentType.VIDEO
            ai_generated_content = {
                "title": transcript.title,
                "summary": "Content processing requires OpenAI API access. Please check your API key and quota.",
                "key_points": ["AI processing not available"],
                "conclusion": "Unable to generate content summary due to API limitations."
            }
            slides_data = generate_slides_with_ai(transcript.title, transcript.transcript_text, ai_generated_content, categories, "transcript")
            feed_title = transcript.title
            status = "partial"
        
        if existing_feed and overwrite:
            # UPDATE existing feed
            existing_feed.title = feed_title
            existing_feed.categories = categories
            existing_feed.skills = skills
            existing_feed.tools = tools
            existing_feed.roles = roles
            existing_feed.content_type = content_type
            existing_feed.status = status
            existing_feed.ai_generated_content = ai_generated_content
            existing_feed.updated_at = datetime.utcnow()
            
            # Delete old slides
            db.query(Slide).filter(Slide.feed_id == existing_feed.id).delete()
            db.flush()
            
            # Create new slides
            for slide_data in slides_data:
                slide = Slide(
                    feed_id=existing_feed.id,
                    order=slide_data["order"],
                    title=slide_data["title"],
                    body=slide_data["body"],
                    bullets=slide_data.get("bullets"),
                    background_color=slide_data.get("background_color", "#FFFFFF"),
                    source_refs=slide_data.get("source_refs", []),
                    render_markdown=int(slide_data.get("render_markdown", True)),
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                db.add(slide)
            
            db.commit()
            db.refresh(existing_feed)
            logger.info(f"Successfully UPDATED AI-generated feed {existing_feed.id} for transcript {transcript.transcript_id}")
            return existing_feed
        else:
            # CREATE new feed
            feed = Feed(
                transcript_id=transcript.transcript_id,
                title=feed_title,
                categories=categories,
                skills=skills,
                tools=tools,
                roles=roles,
                content_type=content_type,
                status=status,
                ai_generated_content=ai_generated_content,
                image_generation_enabled=False,
                source_type="youtube",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.add(feed)
            db.flush()
            
            # Create slides
            for slide_data in slides_data:
                slide = Slide(
                    feed_id=feed.id,
                    order=slide_data["order"],
                    title=slide_data["title"],
                    body=slide_data["body"],
                    bullets=slide_data.get("bullets"),
                    background_color=slide_data.get("background_color", "#FFFFFF"),
                    source_refs=slide_data.get("source_refs", []),
                    render_markdown=int(slide_data.get("render_markdown", True)),
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                db.add(slide)
            
            db.commit()
            db.refresh(feed)
            logger.info(f"Successfully CREATED AI-generated feed {feed.id} for transcript {transcript.transcript_id}")
            return feed
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating/updating AI-generated feed for transcript {transcript.transcript_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create feed for transcript: {str(e)}")

def get_feed_metadata(feed: Feed, db: Session) -> Dict[str, Any]:
    """Extract proper metadata for feeds including YouTube channel names and correct URLs."""
    if feed.source_type == "youtube":
        # Get the transcript to access YouTube-specific data
        transcript = db.query(Transcript).filter(Transcript.transcript_id == feed.transcript_id).first()
        
        if transcript:
            # Extract video ID from transcript data
            video_id = getattr(transcript, 'video_id', None)
            if not video_id:
                # Try to extract from transcript_id or other fields
                video_id = getattr(transcript, 'youtube_video_id', feed.transcript_id)
            
            # Get original title from transcript
            original_title = transcript.title if transcript else feed.title
            
            # Get channel information from YouTube API
            channel_info = get_youtube_channel_info(video_id)
            channel_name = channel_info.get("channel_name", "YouTube Creator")
            
            # Construct proper YouTube URL
            source_url = f"https://www.youtube.com/watch?v={video_id}"
            
            return {
                "title": feed.title,
                "original_title": original_title,
                "author": channel_name,
                "source_url": source_url,
                "source_type": "youtube",
                "channel_name": channel_name,
                "channel_id": channel_info.get("channel_id"),
                "video_id": video_id,
                "channel_info": channel_info
            }
        else:
            # Fallback if transcript not found
            video_id = feed.transcript_id
            channel_info = get_youtube_channel_info(video_id)
            channel_name = channel_info.get("channel_name", "YouTube Creator")
            
            return {
                "title": feed.title,
                "original_title": feed.title,
                "author": channel_name,
                "source_url": f"https://www.youtube.com/watch?v={video_id}",
                "source_type": "youtube",
                "channel_name": channel_name,
                "channel_id": channel_info.get("channel_id"),
                "video_id": video_id,
                "channel_info": channel_info
            }
    
    else:  # blog source type
        blog = feed.blog
        if blog:
            website_name = blog.website.replace("https://", "").replace("http://", "").split("/")[0]
            author = getattr(blog, 'author', 'Admin') or 'Admin'
            
            return {
                "title": feed.title,
                "original_title": blog.title,
                "author": author,
                "source_url": blog.url,
                "source_type": "blog",
                "website_name": website_name,
                "website": blog.website
            }
        else:
            # Fallback if blog not found
            return {
                "title": feed.title,
                "original_title": "Unknown",
                "author": "Admin",
                "source_url": "#",
                "source_type": "blog",
                "website_name": "Unknown",
                "website": "Unknown"
            }

def process_blog_feeds_creation(blogs: List[Blog], website: str, overwrite: bool = False, use_ai: bool = True):
    """Background task to process blog feed creation."""
    from database import SessionLocal
    db = SessionLocal()
    try:
        created_count = 0
        skipped_count = 0
        error_count = 0
        openai_error_count = 0
        error_messages = []
        openai_errors = []
        
        for blog in blogs:
            try:
                existing_feed = db.query(Feed).filter(Feed.blog_id == blog.id).first()
                if existing_feed and not overwrite:
                    skipped_count += 1
                    continue
                
                if existing_feed and overwrite:
                    # Delete existing slides to regenerate
                    db.query(Slide).filter(Slide.feed_id == existing_feed.id).delete()
                    db.flush()
                
                # Create feed with AI
                feed = create_feed_from_blog(db, blog)
                
                if feed:
                    created_count += 1
                    if feed.status == "partial":
                        openai_error_count += 1
                        openai_errors.append(f"Blog {blog.id}: OpenAI quota/configuration issue")
                    
            except HTTPException as e:
                if "quota" in str(e.detail).lower() or "OpenAI" in str(e.detail):
                    openai_error_count += 1
                    openai_errors.append(f"Blog {blog.id}: {e.detail}")
                else:
                    error_count += 1
                    error_messages.append(f"Blog {blog.id}: {e.detail}")
                logger.error(f"Error processing blog {blog.id}: {e.detail}")
                continue
            except Exception as e:
                error_count += 1
                error_messages.append(f"Blog {blog.id}: {str(e)}")
                logger.error(f"Error processing blog {blog.id}: {e}")
                continue
        
        summary = f"Completed blog feed creation for {website}: {created_count} created"
        if openai_error_count > 0:
            summary += f", {openai_error_count} with OpenAI issues"
        if error_count > 0:
            summary += f", {error_count} errors"
        if skipped_count > 0:
            summary += f", {skipped_count} skipped"
        
        logger.info(summary)
        
        # Log errors for debugging
        if openai_errors:
            logger.warning(f"OpenAI errors for {website}: {openai_errors[:5]}")  # Log first 5
            
    finally:
        db.close()

def process_transcript_feeds_creation(transcripts: List[Transcript], job_id: str, overwrite: bool = False, use_ai: bool = True):
    """Background task to process transcript feed creation."""
    from database import SessionLocal
    db = SessionLocal()
    try:
        created_count = 0
        skipped_count = 0
        error_count = 0
        openai_error_count = 0
        error_messages = []
        openai_errors = []
        
        for transcript in transcripts:
            try:
                existing_feed = db.query(Feed).filter(Feed.transcript_id == transcript.transcript_id).first()
                if existing_feed and not overwrite:
                    skipped_count += 1
                    continue
                
                # Create feed with AI
                feed = create_feed_from_transcript(db, transcript, overwrite)
                
                if feed:
                    created_count += 1
                    if feed.status == "partial":
                        openai_error_count += 1
                        openai_errors.append(f"Transcript {transcript.transcript_id}: OpenAI quota/configuration issue")
                    
            except HTTPException as e:
                if "quota" in str(e.detail).lower() or "OpenAI" in str(e.detail):
                    openai_error_count += 1
                    openai_errors.append(f"Transcript {transcript.transcript_id}: {e.detail}")
                else:
                    error_count += 1
                    error_messages.append(f"Transcript {transcript.transcript_id}: {e.detail}")
                logger.error(f"Error processing transcript {transcript.transcript_id}: {e.detail}")
                continue
            except Exception as e:
                error_count += 1
                error_messages.append(f"Transcript {transcript.transcript_id}: {str(e)}")
                logger.error(f"Error processing transcript {transcript.transcript_id}: {e}")
                continue
        
        summary = f"Completed transcript feed creation for job {job_id}: {created_count} created"
        if openai_error_count > 0:
            summary += f", {openai_error_count} with OpenAI issues"
        if error_count > 0:
            summary += f", {error_count} errors"
        if skipped_count > 0:
            summary += f", {skipped_count} skipped"
        
        logger.info(summary)
        
        # Log errors for debugging
        if openai_errors:
            logger.warning(f"OpenAI errors for job {job_id}: {openai_errors[:5]}")  # Log first 5
            
    finally:
        db.close()

# ------------------ Endpoints ------------------

@router.get("/sources/with-ids", response_model=List[dict])
def get_sources_with_ids(
    page: int = 1,
    limit: int = 50,
    active_only: bool = True,
    db: Session = Depends(get_db)
):
    """Get all sources with their IDs and feed counts."""
    try:
        query = db.query(Source)
        
        if active_only:
            query = query.filter(Source.is_active == True)
        
        total = query.count()
        sources = query.offset((page - 1) * limit).limit(limit).all()
        
        sources_with_counts = []
        for source in sources:
            # Count ALL feeds for this source (not just published)
            if source.source_type == "blog":
                feed_count = db.query(Feed).join(Blog).filter(
                    Blog.website == source.website
                ).count()
            else:  # youtube
                feed_count = db.query(Feed).filter(
                    Feed.source_type == "youtube"
                ).count()
            
            sources_with_counts.append({
                "id": source.id,
                "name": source.name,
                "website": source.website,
                "source_type": source.source_type,
                "feed_count": feed_count,
                "follower_count": source.follower_count,
                "is_active": source.is_active
            })
        
        return sources_with_counts
        
    except Exception as e:
        logger.error(f"Error fetching sources with IDs: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch sources")

@router.get("/source/{source_id}/feeds", response_model=dict)
def get_feeds_by_source_id(
    source_id: int,
    page: int = 1,
    limit: int = 20,
    content_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get all feeds (both published and unpublished) by source ID."""
    try:
        source = db.query(Source).filter(Source.id == source_id).first()
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        
        # Build query based on source type
        if source.source_type == "blog":
            query = db.query(Feed).options(
                joinedload(Feed.slides),
                joinedload(Feed.blog)
            ).join(Blog).filter(
                Blog.website == source.website
            )
        else:
            query = db.query(Feed).options(
                joinedload(Feed.slides)
            ).filter(
                Feed.source_type == "youtube"
            )
        
        # Apply content type filter with safe handling
        if content_type:
            try:
                # Convert string to FilterType enum
                content_type_enum = FilterType(content_type)
                query = query.filter(Feed.content_type == content_type_enum)
            except ValueError:
                # If invalid content_type provided, ignore the filter
                logger.warning(f"Invalid content_type provided: {content_type}")
        
        query = query.order_by(Feed.created_at.desc())
        
        total = query.count()
        feeds = query.offset((page - 1) * limit).limit(limit).all()
        
        # Format response with safe attribute access
        feeds_data = []
        for feed in feeds:
            # Check if published
            is_published = db.query(PublishedFeed).filter(
                PublishedFeed.feed_id == feed.id,
                PublishedFeed.is_active == True
            ).first() is not None
            
            # Safely get content_type
            feed_content_type = feed.content_type.value if feed.content_type else FilterType.BLOG.value
            
            feeds_data.append({
                "id": feed.id,
                "title": feed.title,
                "categories": feed.categories or [],
                "content_type": feed_content_type,
                "skills": getattr(feed, 'skills', []) or [],
                "tools": getattr(feed, 'tools', []) or [],
                "roles": getattr(feed, 'roles', []) or [],
                "status": feed.status,
                "is_published": is_published,
                "slides_count": len(feed.slides) if feed.slides else 0,
                "created_at": feed.created_at.isoformat() if feed.created_at else None,
                "source_type": feed.source_type
            })
        
        return {
            "source": {
                "id": source.id,
                "name": source.name,
                "website": source.website,
                "source_type": source.source_type
            },
            "feeds": feeds_data,
            "total": total,
            "page": page,
            "limit": limit,
            "has_more": (page * limit) < total
        }
        
    except Exception as e:
        logger.error(f"Error fetching feeds for source {source_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch feeds for source")

@router.get("/all", response_model=dict)
def get_all_feeds(
    response: Response, 
    page: int = 1, 
    limit: int = 20, 
    category: Optional[str] = None,
    status: Optional[str] = None,
    source_type: Optional[str] = None,
    is_published: Optional[bool] = None,
    content_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get all feeds (both published and unpublished) with filtering."""
    query = db.query(Feed).options(joinedload(Feed.blog))
    
    if category:
        query = query.filter(Feed.categories.contains([category]))
    if status:
        query = query.filter(Feed.status == status)
    if source_type:
        query = query.filter(Feed.source_type == source_type)
    if content_type:
        try:
            content_type_enum = FilterType(content_type)
            query = query.filter(Feed.content_type == content_type_enum)
        except ValueError:
            logger.warning(f"Invalid content_type provided: {content_type}")
    if is_published is not None:
        if is_published:
            published_feed_ids = db.query(PublishedFeed.feed_id).filter(PublishedFeed.is_active == True)
            query = query.filter(Feed.id.in_(published_feed_ids))
        else:
            published_feed_ids = db.query(PublishedFeed.feed_id).filter(PublishedFeed.is_active == True)
            query = query.filter(~Feed.id.in_(published_feed_ids))
    
    query = query.order_by(Feed.created_at.desc())
    total = query.count()
    feeds = query.offset((page - 1) * limit).limit(limit).all()

    items = []
    for feed in feeds:
        is_published_status = db.query(PublishedFeed).filter(
            PublishedFeed.feed_id == feed.id,
            PublishedFeed.is_active == True
        ).first() is not None
        
        feed_content_type = feed.content_type.value if feed.content_type else FilterType.BLOG.value
        
        items.append({
            "id": feed.id,
            "blog_id": feed.blog_id,
            "transcript_id": feed.transcript_id,
            "title": feed.title,
            "categories": feed.categories,
            "content_type": feed_content_type,
            "skills": getattr(feed, 'skills', []) or [],
            "tools": getattr(feed, 'tools', []) or [],
            "roles": getattr(feed, 'roles', []) or [],
            "status": feed.status,
            "source_type": feed.source_type or "blog",
            "is_published": is_published_status,
            "slides_count": len(feed.slides) if feed.slides else 0,
            "created_at": feed.created_at.isoformat() if feed.created_at else None,
            "updated_at": feed.updated_at.isoformat() if feed.updated_at else None,
            "ai_generated": feed.ai_generated_content is not None,
        })

    has_more = (page * limit) < total
    response.headers["X-Total-Count"] = str(total)
    response.headers["X-Page"] = str(page)
    response.headers["X-Limit"] = str(limit)
    
    return {
        "items": items,
        "page": page, 
        "limit": limit, 
        "total": total, 
        "has_more": has_more
    }

@router.get("/{feed_id}", response_model=dict)
def get_feed_by_id(feed_id: int, db: Session = Depends(get_db)):
    """Get full AI-generated feed with slides and is_published status."""
    feed = db.query(Feed).options(
        joinedload(Feed.blog), 
        joinedload(Feed.slides),
        joinedload(Feed.published_feed)
    ).filter(Feed.id == feed_id).first()
    
    if not feed:
        raise HTTPException(status_code=404, detail="Feed not found")

    ai_content = getattr(feed, 'ai_generated_content', {})
    
    # Determine if the feed is published
    is_published = feed.published_feed is not None
    
    # Get proper metadata
    meta = get_feed_metadata(feed, db)
    
    return {
        "id": feed.id,
        "blog_id": feed.blog_id,
        "transcript_id": feed.transcript_id,
        "title": feed.title,
        "categories": feed.categories,
        "status": feed.status,
        "source_type": feed.source_type or "blog",
        "ai_generated_content": ai_content,
        "is_published": is_published,
        "published_at": feed.published_feed.published_at.isoformat() if is_published else None,
        "meta": meta,
        "slides": sorted([
            {
                "id": s.id,
                "order": s.order,
                "title": s.title,
                "body": s.body,
                "bullets": s.bullets,
                "background_color": s.background_color,
                "background_image_prompt": None,
                "source_refs": s.source_refs,
                "render_markdown": bool(s.render_markdown),
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "updated_at": s.updated_at.isoformat() if s.updated_at else None
            } for s in feed.slides
        ], key=lambda x: x["order"]),
        "created_at": feed.created_at.isoformat() if feed.created_at else None,
        "updated_at": feed.updated_at.isoformat() if feed.updated_at else None,
        "ai_generated": hasattr(feed, 'ai_generated_content') and feed.ai_generated_content is not None,
        "images_generated": False
    }

@router.post("/feeds", response_model=dict)
def create_feeds_from_website(
    request: FeedRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Create feeds for all blogs from a website."""
    blogs = db.query(Blog).filter(Blog.website == request.website).all()
    if not blogs:
        raise HTTPException(status_code=404, detail="No blogs found for this website")
    
    # CHECK OPENAI AVAILABILITY BEFORE STARTING
    openai_check = check_openai_availability()
    if not openai_check["available"]:
        return {
            "website": request.website,
            "total_blogs": len(blogs),
            "use_ai": True,
            "generate_images": False,
            "source_type": "blog",
            "message": f"Cannot start feed creation: {openai_check['reason']}",
            "status": "failed",
            "openai_available": False,
            "openai_reason": openai_check["reason"]
        }

    background_tasks.add_task(
        process_blog_feeds_creation,
        blogs,
        request.website,
        request.overwrite,
        True,
    )

    return {
        "website": request.website,
        "total_blogs": len(blogs),
        "use_ai": True,
        "generate_images": False,
        "source_type": "blog",
        "message": "Blog feed creation process started in background",
        "status": "processing",
        "openai_available": True,
        "warning": "If OpenAI API fails during processing, feeds will be created with minimal content"
    }

@router.post("/feeds/youtube", response_model=dict)
def create_feeds_from_youtube(
    request: YouTubeFeedRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Create feeds from YouTube transcripts."""
    transcripts = []
    job_identifier = "all_transcripts"
    
    if request.job_id:
        transcript_job = db.query(TranscriptJob).filter(TranscriptJob.job_id == request.job_id).first()
        if transcript_job:
            transcripts = db.query(Transcript).filter(Transcript.job_id == transcript_job.id).all()
            job_identifier = f"job_{request.job_id}"
        else:
            video_transcript = db.query(Transcript).filter(Transcript.video_id == request.job_id).first()
            if video_transcript:
                transcripts = [video_transcript]
                job_identifier = f"video_{request.job_id}"
            else:
                raise HTTPException(
                    status_code=404, 
                    detail=f"No transcripts found for job ID: {request.job_id}"
                )
    elif request.video_id:
        transcripts = db.query(Transcript).filter(Transcript.video_id == request.video_id).all()
        job_identifier = f"video_{request.video_id}"
        if not transcripts:
            raise HTTPException(status_code=404, detail=f"No transcript found for video ID: {request.video_id}")
    else:
        transcripts = db.query(Transcript).all()
        if not transcripts:
            raise HTTPException(status_code=404, detail="No transcripts found in database")

    if not transcripts:
        raise HTTPException(status_code=404, detail="No transcripts found for the given criteria")
    
    if not transcripts:
        raise HTTPException(status_code=404, detail="No transcripts found for the given criteria")
    
    # CHECK OPENAI AVAILABILITY BEFORE STARTING
    openai_check = check_openai_availability()
    if not openai_check["available"]:
        return {
            "job_id": request.job_id,
            "video_id": request.video_id,
            "total_transcripts": len(transcripts),
            "use_ai": True,
            "generate_images": False,
            "source_type": "youtube",
            "message": f"Cannot start feed creation: {openai_check['reason']}",
            "status": "failed",
            "openai_available": False,
            # "openai_reason": openai_check["reason"]
        }

    background_tasks.add_task(
        process_transcript_feeds_creation,
        transcripts,
        job_identifier,
        request.overwrite,
        True,
    )

    return {
        "job_id": request.job_id,
        "video_id": request.video_id,
        "total_transcripts": len(transcripts),
        "use_ai": True,
        "generate_images": False,
        "source_type": "youtube",
        "message": f"YouTube transcript feed creation process started for {len(transcripts)} transcripts",
        "status": "processing",
        "openai_available": True,
        "warning": "If OpenAI API fails during processing, feeds will be created with minimal content"
    }


@router.get("/source/{website}/categorized", response_model=dict)
def get_categorized_feeds_by_source(
    website: str,
    response: Response, 
    page: int = 1, 
    limit: int = 20,
    exclude_uncategorized: bool = True,
    db: Session = Depends(get_db)
):
    """Get categorized feeds from a specific source URL/website."""
    # First, verify the website exists and get its blogs
    blogs = db.query(Blog).filter(Blog.website == website).all()
    if not blogs:
        raise HTTPException(status_code=404, detail=f"No blogs found for website: {website}")
    
    # Get blog IDs for this website
    blog_ids = [blog.id for blog in blogs]
    
    # Query feeds for these blog IDs with category filters
    query = db.query(Feed).options(joinedload(Feed.blog)).filter(Feed.blog_id.in_(blog_ids))
    
    # Filter feeds that have categories
    query = query.filter(Feed.categories.isnot(None))
    
    if exclude_uncategorized:
        # Exclude feeds that contain "Uncategorized" in their categories
        query = query.filter(~Feed.categories.contains(["Uncategorized"]))
    
    # Order by creation date (newest first)
    query = query.order_by(Feed.created_at.desc())
    
    # Get total count and paginated results
    total = query.count()
    feeds = query.offset((page - 1) * limit).limit(limit).all()

    # Format the response
    items = []
    for f in feeds:
        # Additional validation to ensure meaningful categories
        if f.categories and (not exclude_uncategorized or "Uncategorized" not in f.categories):
            # Get proper metadata
            meta = get_feed_metadata(f, db)
            
            items.append({
                "id": f.id,
                "blog_id": f.blog_id,
                "title": f.title,
                "categories": f.categories,
                "status": f.status,
                "slides_count": len(f.slides),
                "meta": meta,
                "created_at": f.created_at.isoformat() if f.created_at else None,
                "updated_at": f.updated_at.isoformat() if f.updated_at else None,
                "ai_generated": hasattr(f, 'ai_generated_content') and f.ai_generated_content is not None,
                "images_generated": False
            })

    has_more = (page * limit) < total
    response.headers["X-Total-Count"] = str(total)
    response.headers["X-Page"] = str(page)
    response.headers["X-Limit"] = str(limit)

    return {
        "website": website,
        "items": items, 
        "page": page, 
        "limit": limit, 
        "total": total, 
        "has_more": has_more,
        "filters": {
            "exclude_uncategorized": exclude_uncategorized
        }
    }