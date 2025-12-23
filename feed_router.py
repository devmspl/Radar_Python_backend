from fastapi import APIRouter, Depends, HTTPException, Response, BackgroundTasks, Query
from sqlalchemy.orm import Session, joinedload
from datetime import datetime
import os
import re
import logging
import json
from typing import List, Optional, Dict, Any
from database import get_db
from models import Blog, Category, Feed, Slide, Transcript, TranscriptJob, Source, PublishedFeed, FilterType,SubCategory,ScrapeJob,Topic
from openai import OpenAI, APIError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential
from schemas import FeedRequest, DeleteSlideRequest, YouTubeFeedRequest,FilterFeedsRequest
from sqlalchemy import or_, String
import requests
from enum import Enum
from sqlalchemy import func, or_, and_, asc, desc
from models import Domain, Concept, FeedConcept, DomainConcept, ContentList

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
                        # "description": channel_info.get("description", ""),
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
    """Categorize content using OpenAI and extract skills, tools, roles, and concepts."""
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
                    6. Extract key CONCEPTS discussed (3-5 main concepts)
                    7. Suggest 1-2 DOMAINS this belongs to (from: Technology, Business, Science, Arts, Health, Education, etc.)
                    
                    Return JSON with this structure:
                    {
                        "categories": ["category1", "category2"],
                        "skills": ["skill1", "skill2"],
                        "tools": ["tool1", "tool2"],
                        "roles": ["role1", "role2"],
                        "content_type": "Blog/Video/Podcast/Webinar",
                        "concepts": ["concept1", "concept2", "concept3"],
                        "domains": ["domain1", "domain2"]
                    }"""
                },
                {
                    "role": "user",
                    "content": f"Available categories: {', '.join(admin_categories)}.\n\nContent:\n{truncated_content}\n\nReturn JSON with categories, skills, tools, roles, content_type, concepts, and domains."
                }
            ],
            temperature=0.1,
            max_tokens=500,
            response_format={ "type": "json_object" }
        )
        
        analysis = json.loads(response.choices[0].message.content.strip())
        
        # Extract all components
        categories = analysis.get("categories", [])
        skills = analysis.get("skills", [])
        tools = analysis.get("tools", [])
        roles = analysis.get("roles", [])
        concepts = analysis.get("concepts", [])
        domains = analysis.get("domains", [])
        
        # Validate and process categories
        matched_categories = []
        for item in categories:
            for category in admin_categories:
                if (item.lower() == category.lower() or 
                    item.lower() in category.lower() or 
                    category.lower() in item.lower()):
                    matched_categories.append(category)
                    break
        
        # Remove duplicates
        seen = set()
        unique_categories = []
        for cat in matched_categories:
            if cat not in seen:
                seen.add(cat)
                unique_categories.append(cat)
        
        if not unique_categories:
            unique_categories = [admin_categories[0]] if admin_categories else ["Uncategorized"]
        
        # Determine content type
        content_type_str = analysis.get("content_type", "Blog")
        content_type = ContentType.BLOG
        
        if "video" in content_type_str.lower() or "youtube" in content_type_str.lower():
            content_type = ContentType.VIDEO
        elif "podcast" in content_type_str.lower():
            content_type = ContentType.PODCAST
        elif "webinar" in content_type_str.lower():
            content_type = ContentType.WEBINAR
        
        return (
            unique_categories[:3], 
            skills[:10], 
            tools[:10], 
            roles[:10], 
            content_type,
            concepts[:5],  # Limit to 5 concepts
            domains[:2]   # Limit to 2 domains
        )
        
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
        # 0. COVER/SUMMARY SLIDE (NEW - added at position 1)
        cover_context = f"""Create a comprehensive cover slide that summarizes all the key points from this content.
        Title: {title}
        Content Type: {content_type}
        Categories: {', '.join(categories)}
        
        Available Content:
        - Summary: {ai_generated_content.get('summary', '')[:200]}...
        - Key Points: {', '.join(ai_generated_content.get('key_points', [])[:5])}
        - Conclusion: {ai_generated_content.get('conclusion', '')[:150]}...
        
        Create an engaging cover slide that provides an overview of what will be covered in all slides."""
        
        cover_slide = generate_slide_with_ai("cover", cover_context, categories, content_type)
        cover_slide["order"] = 1  # First position
        cover_slide["title"] = f"Overview: {title}"  # Enhanced title
        cover_slide["background_color"] = background_color
        slides.append(cover_slide)
        
        # 1. Original title slide (now becomes slide 2)
        title_context = f"Create an engaging title slide for: {title}\nSummary: {ai_generated_content.get('summary', '')}\nCategories: {', '.join(categories)}\nType: {content_type}"
        title_slide = generate_slide_with_ai("title", title_context, categories, content_type, slides)
        title_slide["order"] = 2  # Changed from 1 to 2
        title_slide["background_color"] = background_color
        slides.append(title_slide)
        
        if slide_count == 1:
            return slides[:1]  # Return only cover slide if count is 1
        
        # 2. Summary slide (now becomes slide 3 if slide_count > 1)
        if slide_count > 2:
            summary_context = f"Create a comprehensive summary slide for: {title}\nFull Summary: {ai_generated_content.get('summary', '')}\nType: {content_type}"
            summary_slide = generate_slide_with_ai("summary", summary_context, categories, content_type, slides)
            summary_slide["order"] = 3  # Changed from 2 to 3
            summary_slide["background_color"] = background_color
            slides.append(summary_slide)
        
        if slide_count <= 2:
            return slides[:slide_count]
        
        # Adjust remaining slides calculation (since we added a cover slide)
        remaining_slides = slide_count - len(slides)
        
        # 3. Key point slides
        key_points = ai_generated_content.get("key_points", [])
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
            insight_context = f"Create an additional insights slide for: {title}\nAvailable Content: Summary - {ai_generated_content.get('summary', '')[:200]}...\nRemaining Key Points: {', '.join(key_points[len(slides)-3:]) if len(slides)-3 < len(key_points) else 'Various important aspects'}\nType: {content_type}"
            insight_slide = generate_slide_with_ai("additional_insights", insight_context, categories, content_type, slides)
            insight_slide["order"] = len(slides) + 1
            insight_slide["background_color"] = background_color
            slides.append(insight_slide)
        
        logger.info(f"Successfully generated {len(slides)} slides for '{title}' (including cover slide)")
        return slides[:slide_count]
        
    except HTTPException as e:
        # If OpenAI fails, create minimal slides
        logger.error(f"Error in generate_slides_with_ai for '{title}': {e.detail}")
        
        # Create minimal fallback slides with cover
        fallback_slides = [
            {
                "order": 1,
                "title": f"Cover: {title}",
                "body": "Overview of all key points...",
                "bullets": ["AI processing failed for detailed content"],
                "background_color": "#FFFFFF",
                "source_refs": [],
                "render_markdown": True
            }
        ]
        
        if slide_count > 1:
            fallback_slides.append({
                "order": 2,
                "title": title,
                "body": "Feed generation requires OpenAI API access. Please check your API key and quota.",
                "bullets": ["AI processing failed"],
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

# ------------------ Helper Functions for New Models ------------------

# ------------------ Enhanced Helper Functions ------------------

def extract_clean_source_name(website_url: str) -> str:
    """Extract clean source name using OpenAI or basic parsing."""
    try:
        if not website_url or website_url in ["Unknown", "#"]:
            return "Unknown Source"
        
        # Basic cleaning
        clean_url = website_url.replace("https://", "").replace("http://", "").split("/")[0]
        
        # Try OpenAI for better naming
        if client:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a naming expert. Extract a clean, professional source name from a domain.
                        Return ONLY the name, no explanations.
                        Examples:
                        - "blog.hubspot.com" -> "HubSpot Blog"
                        - "news.ycombinator.com" -> "Hacker News" 
                        - "techcrunch.com" -> "TechCrunch"
                        - "medium.com" -> "Medium"
                        - "towardsdatascience.com" -> "Towards Data Science"
                        - If unsure, return the main domain name without TLD."""
                    },
                    {
                        "role": "user",
                        "content": f"Extract a clean source name from: {clean_url}"
                    }
                ],
                temperature=0.1,
                max_tokens=50
            )
            
            source_name = response.choices[0].message.content.strip()
            source_name = source_name.replace('"', '').replace("'", "").strip()
            
            if source_name and len(source_name) > 2:
                return source_name
        
        # Fallback to basic domain extraction
        parts = clean_url.split('.')
        if len(parts) > 2:
            # Remove www and TLD
            return parts[1].title() if parts[0] == 'www' else parts[0].title()
        elif len(parts) == 2:
            return parts[0].title()
        else:
            return clean_url
            
    except Exception as e:
        logger.error(f"Source name extraction failed for {website_url}: {e}")
        # Fallback to basic domain extraction
        clean_url = website_url.replace("https://", "").replace("http://", "").split("/")[0]
        return clean_url


def get_or_create_concepts(db: Session, concept_names: List[str]) -> List[Concept]:
    """Get or create concept objects."""
    concepts = []
    for name in concept_names:
        if not name or len(name.strip()) < 2:
            continue
            
        concept = db.query(Concept).filter(
            func.lower(Concept.name) == func.lower(name.strip())
        ).first()
        
        if not concept:
            concept = Concept(
                name=name.strip(),
                description=f"Concept for {name.strip()}",
                is_active=True
            )
            db.add(concept)
            db.flush()
        
        concepts.append(concept)
    
    return concepts

def get_or_create_domains(db: Session, domain_names: List[str]) -> List[Domain]:
    """Get or create domain objects."""
    domains = []
    for name in domain_names:
        if not name or len(name.strip()) < 2:
            continue
            
        domain = db.query(Domain).filter(
            func.lower(Domain.name) == func.lower(name.strip())
        ).first()
        
        if not domain:
            domain = Domain(
                name=name.strip(),
                description=f"Domain for {name.strip()}",
                is_active=True
            )
            db.add(domain)
            db.flush()
        
        domains.append(domain)
    
    return domains

def get_or_assign_category(db: Session, categories: List[str]) -> tuple:
    """Get or assign category and subcategory."""
    category_obj = None
    subcategory_obj = None
    
    if categories and len(categories) > 0:
        # Get the first category
        category_name = categories[0]
        category_obj = db.query(Category).filter(
            Category.name.ilike(f"%{category_name}%")
        ).first()
        
        # If category found, get its first subcategory
        if category_obj:
            subcategory_obj = db.query(SubCategory).filter(
                SubCategory.category_id == category_obj.id
            ).first()
    
    return category_obj, subcategory_obj


def get_topic_descriptions(categories: List[str], db: Session) -> List[Dict[str, Any]]:
    """Get topic descriptions for given categories."""
    topic_descriptions = []
    
    for category_name in categories:
        topic = db.query(Topic).filter(
            Topic.name == category_name,
            Topic.is_active == True
        ).first()
        
        if topic:
            # Count feeds for this topic
            feed_count = db.query(Feed).filter(
                Feed.status == "ready",
                Feed.categories.isnot(None),
                Feed.categories.contains([category_name])
            ).count()
            
            topic_descriptions.append({
                "name": topic.name,
                "description": topic.description,
                "id": topic.id,
                "feed_count": feed_count,
                "follower_count": topic.follower_count
            })
        else:
            # Create a basic description if topic doesn't exist
            topic_descriptions.append({
                "name": category_name,
                "description": f"Content related to {category_name}",
                "id": None,
                "feed_count": 0,
                "follower_count": 0
            })
    
    return topic_descriptions

def create_or_update_content_list(playlist_id: str, feed_id: int, db: Session):
    """Create or update content list for YouTube playlist."""
    # Find existing content list for this playlist
    content_list = db.query(ContentList).filter(
        ContentList.source_type == "youtube",
        ContentList.source_id == playlist_id
    ).first()
    
    if content_list:
        # Add feed_id to existing list if not already present
        current_feed_ids = content_list.feed_ids or []
        if feed_id not in current_feed_ids:
            content_list.feed_ids = current_feed_ids + [feed_id]
            content_list.updated_at = datetime.utcnow()
            logger.info(f"Updated list '{content_list.name}' with feed {feed_id}")
    else:
        # Try to get playlist info from YouTube API
        playlist_name = f"YouTube Playlist {playlist_id}"
        
        # Try to get playlist name from transcript job
        if feed_id:
            feed = db.query(Feed).filter(Feed.id == feed_id).first()
            if feed and feed.transcript_id:
                transcript = db.query(Transcript).filter(
                    Transcript.transcript_id == feed.transcript_id
                ).first()
                if transcript:
                    # Try to find the job that created this transcript
                    job = db.query(TranscriptJob).filter(
                        TranscriptJob.id == transcript.job_id
                    ).first()
                    if job and job.playlists:
                        try:
                            playlists = json.loads(job.playlists)
                            for pl in playlists:
                                if pl.get("id") == playlist_id:
                                    playlist_name = pl.get("name", playlist_name)
                                    break
                        except json.JSONDecodeError:
                            pass
        
        # Create new list
        content_list = ContentList(
            name=playlist_name,
            description=f"Auto-generated from YouTube playlist: {playlist_name}",
            source_type="youtube",
            source_id=playlist_id,
            feed_ids=[feed_id],
            is_active=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(content_list)
        logger.info(f"Created new list '{playlist_name}' for playlist {playlist_id} with feed {feed_id}")
    
    db.flush()
    return content_list
def auto_create_lists_from_existing_youtube_feeds(db: Session):
    """Auto-create ContentList entries from existing YouTube feeds with playlist_id."""
    try:
        logger.info("Starting auto-creation of lists from existing YouTube feeds...")
        
        # Find all YouTube feeds with transcripts that have playlist_id
        youtube_feeds = db.query(Feed).filter(
            Feed.source_type == "youtube",
            Feed.status == "ready"
        ).all()
        
        list_created = 0
        list_updated = 0
        
        for feed in youtube_feeds:
            # Get the transcript for this feed
            transcript = None
            if feed.transcript_id:
                transcript = db.query(Transcript).filter(
                    Transcript.transcript_id == feed.transcript_id
                ).first()
            
            if transcript and transcript.playlist_id:
                # Create or update content list for this playlist
                try:
                    content_list = create_or_update_content_list(transcript.playlist_id, feed.id, db)
                    # Check if this was a new list or updated list
                    if content_list.created_at == content_list.updated_at:
                        list_created += 1
                    else:
                        list_updated += 1
                except Exception as e:
                    logger.error(f"Error creating list for feed {feed.id}: {e}")
                    continue
        
        # Also create channel-based lists
        channel_groups = {}
        for feed in youtube_feeds:
            meta = get_feed_metadata(feed, db)
            channel_id = meta.get("channel_id")
            channel_name = meta.get("channel_name")
            
            if channel_id and channel_name:
                if channel_id not in channel_groups:
                    channel_groups[channel_id] = {
                        "name": f"{channel_name} Videos",
                        "channel_name": channel_name,
                        "feeds": []
                    }
                channel_groups[channel_id]["feeds"].append(feed.id)
        
        # Create channel-based lists
        for channel_id, data in channel_groups.items():
            if len(data["feeds"]) >= 3:
                # Check if list exists
                existing_list = db.query(ContentList).filter(
                    ContentList.source_type == "youtube",
                    ContentList.source_id == f"channel_{channel_id}"
                ).first()
                
                if not existing_list:
                    new_list = ContentList(
                        name=data["name"],
                        description=f"Videos from {data['channel_name']} YouTube channel",
                        source_type="youtube",
                        source_id=f"channel_{channel_id}",
                        feed_ids=data["feeds"],
                        is_active=True,
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                    db.add(new_list)
                    list_created += 1
        
        db.commit()
        logger.info(f"Auto-created {list_created} new lists and updated {list_updated} existing lists from YouTube feeds")
        return {
            "created": list_created,
            "updated": list_updated,
            "total": list_created + list_updated
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error auto-creating lists from feeds: {e}")
        return {"created": 0, "updated": 0, "total": 0, "error": str(e)}


# ------------------ Updated Core Feed Creation Functions ------------------

def create_feed_from_blog(db: Session, blog: Blog):
    """Generate feed and slides from a blog using AI with full metadata."""
    try:
        # Check if feed already exists for this blog
        existing_feed = db.query(Feed).filter(Feed.blog_id == blog.id).first()
        
        # Delete old slides if updating
        if existing_feed:
            # Delete existing concept relationships
            db.query(FeedConcept).filter(FeedConcept.feed_id == existing_feed.id).delete()
            # Delete slides
            db.query(Slide).filter(Slide.feed_id == existing_feed.id).delete()
            db.flush()
        
        # Get admin categories for classification
        admin_categories = [c.name for c in db.query(Category).filter(Category.is_active == True).all()]
        if not admin_categories:
            admin_categories = ["Uncategorized"]
        
        try:
            # Enhanced categorization with concepts and domains
            categories, skills, tools, roles, content_type, concepts, domains = categorize_content_with_openai(
                blog.content, admin_categories
            )
            
            # Get or assign category and subcategory
            category_obj, subcategory_obj = get_or_assign_category(db, categories)
            
            # Get or create concepts
            concept_objects = get_or_create_concepts(db, concepts)
            
            # Get or create domains
            domain_objects = get_or_create_domains(db, domains)
            
            # Link concepts to domains
            for concept in concept_objects:
                for domain in domain_objects:
                    # Check if relationship already exists
                    existing_link = db.query(DomainConcept).filter(
                        DomainConcept.domain_id == domain.id,
                        DomainConcept.concept_id == concept.id
                    ).first()
                    
                    if not existing_link:
                        domain_concept = DomainConcept(
                            domain_id=domain.id,
                            concept_id=concept.id,
                            relevance_score=1.0
                        )
                        db.add(domain_concept)
            
            # Generate AI content
            ai_generated_content = generate_feed_content_with_ai(blog.title, blog.content, categories, "blog")
            slides_data = generate_slides_with_ai(blog.title, blog.content, ai_generated_content, categories, "blog")
            
            feed_title = ai_generated_content.get("title", blog.title)
            status = "ready"
            
        except HTTPException as e:
            # Fallback minimal feed
            logger.warning(f"OpenAI failed for blog {blog.id}, creating minimal feed: {e.detail}")
            categories = ["Uncategorized"]
            skills = []
            tools = []
            roles = []
            concepts = []
            domains = []
            content_type = ContentType.BLOG
            category_obj = None
            subcategory_obj = None
            concept_objects = []
            domain_objects = []
            
            ai_generated_content = {
                "title": blog.title,
                "summary": "Content processing requires OpenAI API access. Please check your API key and quota.",
                "key_points": ["AI processing not available"],
                "conclusion": "Unable to generate content summary due to API limitations."
            }
            slides_data = generate_slides_with_ai(blog.title, blog.content, ai_generated_content, categories, "blog")
            feed_title = blog.title
            status = "partial"
        
        # Create feed data
        feed_data = {
            "blog_id": blog.id, 
            "title": feed_title,
            "categories": categories, 
            "skills": skills,
            "tools": tools,
            "roles": roles,
            "content_type": content_type,
            "status": status,
            "ai_generated_content": ai_generated_content,
            "image_generation_enabled": False,
            "source_type": "blog",
            "category_id": category_obj.id if category_obj else None,
            "subcategory_id": subcategory_obj.id if subcategory_obj else None,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        if existing_feed:
            # Update existing feed
            for key, value in feed_data.items():
                setattr(existing_feed, key, value)
            feed = existing_feed
        else:
            # Create new feed
            feed = Feed(**feed_data)
            db.add(feed)
        
        db.flush()  # Get feed ID
        
        # Link concepts to feed
        for concept in concept_objects:
            feed_concept = FeedConcept(
                feed_id=feed.id,
                concept_id=concept.id,
                confidence_score=0.8  # Default confidence
            )
            db.add(feed_concept)
        
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
        
        # Create or update source
        if blog.website:
            source_name = extract_clean_source_name(blog.website)
            source = db.query(Source).filter(
                Source.website == blog.website,
                Source.source_type == "blog"
            ).first()
            
            if not source:
                source = Source(
                    name=source_name,
                    website=blog.website,
                    source_type="blog",
                    is_active=True
                )
                db.add(source)
                db.flush()
            
            # Update source popularity
            source.feed_count = db.query(Feed).join(Blog).filter(
                Blog.website == blog.website
            ).count()
        
        db.commit()
        db.refresh(feed)
        logger.info(f"Successfully created AI-generated feed {feed.id} for blog {blog.id} with {len(concept_objects)} concepts")
        return feed
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating AI-generated feed for blog {blog.id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create feed for blog: {str(e)}")

def create_feed_from_transcript(db: Session, transcript: Transcript, overwrite: bool = False):
    """Generate feed and slides from a YouTube transcript using AI with full metadata."""
    try:
        existing_feed = db.query(Feed).filter(Feed.transcript_id == transcript.transcript_id).first()
        
        if existing_feed and not overwrite:
            logger.info(f"Feed already exists for transcript {transcript.transcript_id}, skipping")
            return existing_feed
        
        # Delete old slides if updating
        if existing_feed and overwrite:
            # Delete existing concept relationships
            db.query(FeedConcept).filter(FeedConcept.feed_id == existing_feed.id).delete()
            # Delete slides
            db.query(Slide).filter(Slide.feed_id == existing_feed.id).delete()
            db.flush()
        
        # Get admin categories for classification
        admin_categories = [c.name for c in db.query(Category).filter(Category.is_active == True).all()]
        if not admin_categories:
            admin_categories = ["Uncategorized"]
        
        try:
            # Enhanced categorization with concepts and domains
            categories, skills, tools, roles, content_type, concepts, domains = categorize_content_with_openai(
                transcript.transcript_text, admin_categories
            )
            
            # Get or assign category and subcategory
            category_obj, subcategory_obj = get_or_assign_category(db, categories)
            
            # Get or create concepts
            concept_objects = get_or_create_concepts(db, concepts)
            
            # Get or create domains
            domain_objects = get_or_create_domains(db, domains)
            
            # Link concepts to domains
            for concept in concept_objects:
                for domain in domain_objects:
                    existing_link = db.query(DomainConcept).filter(
                        DomainConcept.domain_id == domain.id,
                        DomainConcept.concept_id == concept.id
                    ).first()
                    
                    if not existing_link:
                        domain_concept = DomainConcept(
                            domain_id=domain.id,
                            concept_id=concept.id,
                            relevance_score=1.0
                        )
                        db.add(domain_concept)
            
            # Generate AI content
            ai_generated_content = generate_feed_content_with_ai(
                transcript.title, transcript.transcript_text, categories, "transcript"
            )
            slides_data = generate_slides_with_ai(
                transcript.title, transcript.transcript_text, ai_generated_content, categories, "transcript"
            )
            
            feed_title = ai_generated_content.get("title", transcript.title)
            status = "ready"
            
        except HTTPException as e:
            # Fallback minimal feed
            logger.warning(f"OpenAI failed for transcript {transcript.transcript_id}, creating minimal feed: {e.detail}")
            categories = ["Uncategorized"]
            skills = []
            tools = []
            roles = []
            concepts = []
            domains = []
            content_type = ContentType.VIDEO
            category_obj = None
            subcategory_obj = None
            concept_objects = []
            domain_objects = []
            
            ai_generated_content = {
                "title": transcript.title,
                "summary": "Content processing requires OpenAI API access. Please check your API key and quota.",
                "key_points": ["AI processing not available"],
                "conclusion": "Unable to generate content summary due to API limitations."
            }
            slides_data = generate_slides_with_ai(
                transcript.title, transcript.transcript_text, ai_generated_content, categories, "transcript"
            )
            feed_title = transcript.title
            status = "partial"
        
        # Prepare feed data
        feed_data = {
            "transcript_id": transcript.transcript_id,
            "title": feed_title,
            "categories": categories,
            "skills": skills,
            "tools": tools,
            "roles": roles,
            "content_type": content_type,
            "status": status,
            "ai_generated_content": ai_generated_content,
            "image_generation_enabled": False,
            "source_type": "youtube",
            "category_id": category_obj.id if category_obj else None,
            "subcategory_id": subcategory_obj.id if subcategory_obj else None,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # Get YouTube video ID and channel info
        youtube_video_id = transcript.video_id
        
        # Store video ID in feed
        if youtube_video_id:
            feed_data["youtube_video_id"] = youtube_video_id
        
        if existing_feed and overwrite:
            # Update existing feed attributes
            for key, value in feed_data.items():
                setattr(existing_feed, key, value)
            
            feed = existing_feed
        else:
            # Create new feed
            feed = Feed(**feed_data)
            db.add(feed)
        
        db.flush()  # Get feed ID
        
        # Link concepts to feed
        for concept in concept_objects:
            feed_concept = FeedConcept(
                feed_id=feed.id,
                concept_id=concept.id,
                confidence_score=0.8
            )
            db.add(feed_concept)
        
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
        
        # Create or update source (YouTube channel)
        channel_info = get_youtube_channel_info(transcript.video_id)
        channel_name = channel_info.get("channel_name", "YouTube Creator")
        channel_id = channel_info.get("channel_id")
        
        # Store channel info in feed metadata
        if channel_id:
            # Update the feed with channel info
            if not feed.youtube_channel_id:
                feed.youtube_channel_id = channel_id
            if not feed.youtube_channel_name:
                feed.youtube_channel_name = channel_name
        
        if channel_name != "YouTube Creator":
            source = db.query(Source).filter(
                or_(
                    Source.name == channel_name,
                    Source.website.like(f"%{channel_id}%") if channel_id else None
                ),
                Source.source_type == "youtube"
            ).first()
            
            if not source:
                website = f"https://www.youtube.com/channel/{channel_id}" if channel_id else "https://www.youtube.com"
                source = Source(
                    name=channel_name,
                    website=website,
                    source_type="youtube",
                    is_active=True,
                    metadata=channel_info
                )
                db.add(source)
                db.flush()
            
            # Update source popularity
            source.feed_count = db.query(Feed).filter(
                Feed.source_type == "youtube",
                Feed.youtube_channel_id == channel_id
            ).count()
        
        # CREATE OR UPDATE CONTENT LIST IF THIS IS FROM A PLAYLIST
        if transcript.playlist_id:
            # Find existing content list for this playlist
            content_list = db.query(ContentList).filter(
                ContentList.source_type == "youtube",
                ContentList.source_id == transcript.playlist_id
            ).first()
            
            if content_list:
                # Add feed_id to existing list if not already present
                current_feed_ids = content_list.feed_ids or []
                if feed.id not in current_feed_ids:
                    content_list.feed_ids = current_feed_ids + [feed.id]
                    content_list.updated_at = datetime.utcnow()
                    logger.info(f"Updated list '{content_list.name}' with feed {feed.id}")
            else:
                # Try to get playlist info from transcript job
                playlist_name = f"YouTube Playlist {transcript.playlist_id}"
                
                # Try to get playlist name from transcript job
                if transcript.job_id:
                    job = db.query(TranscriptJob).filter(
                        TranscriptJob.id == transcript.job_id
                    ).first()
                    if job and job.playlists:
                        try:
                            playlists = json.loads(job.playlists)
                            for pl in playlists:
                                if pl.get("id") == transcript.playlist_id:
                                    playlist_name = pl.get("name", playlist_name)
                                    break
                        except json.JSONDecodeError:
                            pass
                
                # Create new list
                content_list = ContentList(
                    name=playlist_name,
                    description=f"Auto-generated from YouTube playlist: {playlist_name}",
                    source_type="youtube",
                    source_id=transcript.playlist_id,
                    feed_ids=[feed.id],
                    is_active=True,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                db.add(content_list)
                logger.info(f"Created new list '{playlist_name}' for playlist {transcript.playlist_id} with feed {feed.id}")
        
        db.commit()
        db.refresh(feed)
        
        action = "UPDATED" if existing_feed and overwrite else "CREATED"
        logger.info(f"Successfully {action} AI-generated feed {feed.id} for transcript {transcript.transcript_id}")
        return feed
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating/updating AI-generated feed for transcript {transcript.transcript_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create feed for transcript: {str(e)}")

def get_website_favicon(website_url: str) -> str:
    """Get favicon URL for a website."""
    try:
        # Clean and format the URL
        if not website_url.startswith(('http://', 'https://')):
            website_url = f"https://{website_url}"
        
        # Extract domain
        from urllib.parse import urlparse
        parsed_url = urlparse(website_url)
        domain = parsed_url.netloc
        
        if not domain:
            # If parsing failed, try to extract domain manually
            domain = website_url.replace('https://', '').replace('http://', '').split('/')[0]
        
        # Common favicon locations
        favicon_urls = [
            f"https://{domain}/favicon.ico",
            f"https://{domain}/favicon.png",
            f"https://www.{domain}/favicon.ico",
            f"https://www.{domain}/favicon.png",
            f"https://{domain}/apple-touch-icon.png",
            f"https://www.{domain}/apple-touch-icon.png"
        ]
        
        # Try to check if favicon exists (optional)
        # For simplicity, we'll just return the most likely URL
        return favicon_urls[0]
        
    except Exception:
        # Fallback to a generic favicon
        return "https://www.google.com/s2/favicons?domain=" + (website_url.split('//')[-1].split('/')[0] if '//' in website_url else website_url)


def get_feed_metadata(feed: Feed, db: Session) -> Dict[str, Any]:
    """Extract proper metadata for feeds including YouTube channel names, correct URLs, favicons, and topic descriptions."""
    # Initialize base metadata
    base_meta = {}
    concepts = []
    
    # Get concepts for this feed
    if hasattr(feed, 'concepts') and feed.concepts:
        for concept in feed.concepts:
            concepts.append({
                "id": concept.id,
                "name": concept.name,
                "description": concept.description
            })
    
    if feed.source_type == "youtube" and feed.transcript_id:
        # Get the transcript to access YouTube-specific data
        transcript = db.query(Transcript).filter(
            Transcript.transcript_id == feed.transcript_id
        ).first()
        
        if transcript:
            video_id = getattr(transcript, 'video_id', None)
            if not video_id:
                video_id = feed.transcript_id
            
            original_title = transcript.title if transcript else feed.title
            
            # Get channel information from YouTube API
            channel_info = get_youtube_channel_info(video_id)
            channel_name = channel_info.get("channel_name", "YouTube Creator")
            
            source_url = f"https://www.youtube.com/watch?v={video_id}"
            
            # Use YouTube favicon
            favicon = "https://www.youtube.com/s/desktop/12d6b690/img/favicon.ico"
            
            base_meta = {
                "title": feed.title,
                "original_title": original_title,
                "author": channel_name,
                "source_url": source_url,
                "source_type": "youtube",
                "channel_name": channel_name,
                "channel_id": channel_info.get("channel_id"),
                "video_id": video_id,
                "favicon": favicon,
                "channel_info": channel_info,
                "concepts": concepts
            }
    
    # For blogs
    elif feed.source_type == "blog" and feed.blog:
        blog = feed.blog
        website_name = blog.website.replace("https://", "").replace("http://", "").split("/")[0]
        author = getattr(blog, 'author', 'Admin') or 'Admin'
        
        # Get favicon URL
        favicon = get_website_favicon(blog.website)
        
        base_meta = {
            "title": feed.title,
            "original_title": blog.title,
            "author": author,
            "source_url": blog.url,
            "source_type": "blog",
            "website_name": website_name,
            "website": blog.website,
            "favicon": favicon,
            "concepts": concepts
        }
    
    else:
        # Fallback
        base_meta = {
            "title": feed.title,
            "original_title": feed.title,
            "author": "Unknown",
            "source_url": "#",
            "source_type": feed.source_type or "blog",
            "website_name": "Unknown",
            "website": "Unknown",
            "favicon": get_website_favicon("Unknown"),
            "concepts": concepts
        }
    
    # ADD TOPIC DESCRIPTIONS if feed has categories
    if feed.categories:
        topics = []
        for category_name in feed.categories:
            # Get topic from database
            topic = db.query(Topic).filter(
                Topic.name == category_name,
                Topic.is_active == True
            ).first()
            
            if topic:
                topics.append({
                    "name": topic.name,
                    "description": topic.description,
                    "id": topic.id
                })
            else:
                # If topic doesn't exist, create a basic one
                topics.append({
                    "name": category_name,
                    "description": f"Content related to {category_name}",
                    "id": None
                })
        
        base_meta["topics"] = topics
    
    return base_meta


def process_blog_feeds_creation(blogs: List[Blog], website: str, overwrite: bool = False):
    """Enhanced background task to process blog feed creation with full metadata."""
    from database import SessionLocal
    db = SessionLocal()
    try:
        created_count = 0
        updated_count = 0
        skipped_count = 0
        error_count = 0
        openai_error_count = 0
        error_messages = []
        
        for blog in blogs:
            try:
                existing_feed = db.query(Feed).filter(Feed.blog_id == blog.id).first()
                
                if existing_feed and not overwrite:
                    skipped_count += 1
                    continue
                
                # Create or update feed with full metadata
                feed = create_feed_from_blog(db, blog)
                
                if feed:
                    if existing_feed and overwrite:
                        updated_count += 1
                    else:
                        created_count += 1
                    
                    if feed.status == "partial":
                        openai_error_count += 1
                    
            except HTTPException as e:
                if "quota" in str(e.detail).lower() or "OpenAI" in str(e.detail):
                    openai_error_count += 1
                    error_messages.append(f"Blog {blog.id}: {e.detail}")
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
        
        summary = f"Completed blog feed creation for {website}: {created_count} created, {updated_count} updated"
        if openai_error_count > 0:
            summary += f", {openai_error_count} with OpenAI issues"
        if error_count > 0:
            summary += f", {error_count} errors"
        if skipped_count > 0:
            summary += f", {skipped_count} skipped"
        
        logger.info(summary)
        
        # Log first 5 errors for debugging
        if error_messages:
            logger.warning(f"Errors for {website}: {error_messages[:5]}")
            
    finally:
        db.close()

def process_transcript_feeds_creation(transcripts: List[Transcript], job_id: str, overwrite: bool = False):
    """Enhanced background task to process transcript feed creation with full metadata."""
    from database import SessionLocal
    db = SessionLocal()
    try:
        created_count = 0
        updated_count = 0
        skipped_count = 0
        error_count = 0
        openai_error_count = 0
        error_messages = []
        
        for transcript in transcripts:
            try:
                existing_feed = db.query(Feed).filter(Feed.transcript_id == transcript.transcript_id).first()
                
                if existing_feed and not overwrite:
                    skipped_count += 1
                    continue
                
                # Create or update feed with full metadata
                feed = create_feed_from_transcript(db, transcript, overwrite)
                
                if feed:
                    if existing_feed and overwrite:
                        updated_count += 1
                    else:
                        created_count += 1
                    
                    if feed.status == "partial":
                        openai_error_count += 1
                    
            except HTTPException as e:
                if "quota" in str(e.detail).lower() or "OpenAI" in str(e.detail):
                    openai_error_count += 1
                    error_messages.append(f"Transcript {transcript.transcript_id}: {e.detail}")
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
        
        summary = f"Completed transcript feed creation for job {job_id}: {created_count} created, {updated_count} updated"
        if openai_error_count > 0:
            summary += f", {openai_error_count} with OpenAI issues"
        if error_count > 0:
            summary += f", {error_count} errors"
        if skipped_count > 0:
            summary += f", {skipped_count} skipped"
        
        logger.info(summary)
        
        # Log first 5 errors for debugging
        if error_messages:
            logger.warning(f"Errors for job {job_id}: {error_messages[:5]}")
            
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
    # Update query to include category and subcategory
    query = db.query(Feed).options(
        joinedload(Feed.blog),
        joinedload(Feed.category),
        joinedload(Feed.subcategory)
    )
    
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
        
        # Get metadata for each feed
        meta = get_feed_metadata(feed, db)
        
        # Get topic descriptions
        topic_descriptions = []
        if feed.categories:
            topic_descriptions = get_topic_descriptions(feed.categories, db)
        
        # NEW: Get category and subcategory names
        category_name = feed.category.name if feed.category else None
        subcategory_name = feed.subcategory.name if feed.subcategory else None
        
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
            # NEW: Add category and subcategory info
            "category_name": category_name,
            "subcategory_name": subcategory_name,
            "category_id": feed.category_id,
            "subcategory_id": feed.subcategory_id,
            "category_display": f"{category_name} {{ {subcategory_name} }}" if category_name and subcategory_name else category_name,
            "meta": meta,
            "topics": topic_descriptions  # NEW: Add topic descriptions
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


# ------------------ Feeds by Subcategory API ------------------

@router.get("/feeds-by-subcategory", response_model=Dict[str, Any])
def get_feeds_by_subcategory(
    subcategory_name: str = Query(..., description="Subcategory name to search for feeds"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Results per page"),
    include_slides: bool = Query(False, description="Include slide data in response"),
    db: Session = Depends(get_db)
):
    """
    Get all feeds related to a specific subcategory by name.
    
    This API:
    - Takes subcategory name as input
    - Returns all feeds belonging to that subcategory
    - Returns feed count and subcategory details
    - Sorted by latest feeds first (created_at descending)
    
    Authentication: Not required (Public API)
    """
    
    # Step 1: Find subcategory by name (case-insensitive search)
    subcategory = db.query(SubCategory).filter(
        SubCategory.name.ilike(f"%{subcategory_name.strip()}%"),
        SubCategory.is_active == True
    ).first()
    
    if not subcategory:
        raise HTTPException(
            status_code=404,
            detail=f"Subcategory '{subcategory_name}' not found"
        )
    
    # Step 2: Get category info
    category = db.query(Category).filter(
        Category.id == subcategory.category_id
    ).first()
    
    # Step 3: Query feeds for this subcategory
    query_obj = db.query(Feed).filter(
        Feed.subcategory_id == subcategory.id,
        Feed.status == "ready"
    )
    
    if include_slides:
        query_obj = query_obj.options(joinedload(Feed.slides))
    
    # Step 4: Get total count
    total_count = query_obj.count()
    
    # Step 5: Sort by created_at descending and paginate
    query_obj = query_obj.order_by(Feed.created_at.desc())
    feeds = query_obj.offset((page - 1) * limit).limit(limit).all()
    
    # Step 6: Format feeds
    formatted_feeds = []
    for feed in feeds:
        # Get feed metadata
        feed_meta = get_feed_metadata(feed, db)
        
        feed_data = {
            "id": feed.id,
            "title": feed.title,
            "categories": feed.categories,
            "content_type": feed.content_type.value if feed.content_type else "Blog",
            "source_type": feed.source_type,
            "skills": feed.skills or [],
            "tools": feed.tools or [],
            "roles": feed.roles or [],
            "status": feed.status,
            "category_id": feed.category_id,
            "subcategory_id": feed.subcategory_id,
            "created_at": feed.created_at.isoformat() if feed.created_at else None,
            "updated_at": feed.updated_at.isoformat() if feed.updated_at else None,
            "metadata": feed_meta
        }
        
        # Add AI generated content summary if available
        if feed.ai_generated_content:
            feed_data["summary"] = feed.ai_generated_content.get("summary", "")[:200] + "..." if len(feed.ai_generated_content.get("summary", "")) > 200 else feed.ai_generated_content.get("summary", "")
            feed_data["key_points"] = feed.ai_generated_content.get("key_points", [])[:5]
        
        # Add slides if requested
        if include_slides and feed.slides:
            feed_data["slides"] = [
                {
                    "id": slide.id,
                    "order": slide.order,
                    "title": slide.title,
                    "body": slide.body,
                    "bullets": slide.bullets,
                    "background_color": slide.background_color
                }
                for slide in sorted(feed.slides, key=lambda s: s.order)
            ]
            feed_data["slide_count"] = len(feed.slides)
        else:
            feed_data["slide_count"] = len(feed.slides) if feed.slides else 0
        
        formatted_feeds.append(feed_data)
    
    return {
        "subcategory": {
            "id": subcategory.id,
            "uuid": subcategory.uuid,
            "name": subcategory.name,
            "description": subcategory.description,
            "category_id": subcategory.category_id,
            "category_name": category.name if category else "Unknown",
            "is_active": subcategory.is_active,
            "created_at": subcategory.created_at.isoformat() if subcategory.created_at else None,
            "updated_at": subcategory.updated_at.isoformat() if subcategory.updated_at else None
        },
        "feeds": {
            "items": formatted_feeds,
            "page": page,
            "limit": limit,
            "total": total_count,
            "has_more": (page * limit) < total_count
        },
        "feed_count": total_count
    }

    
@router.get("/{feed_id}", response_model=dict)
def get_feed_by_id(feed_id: int, db: Session = Depends(get_db)):
    """Get full AI-generated feed with slides and is_published status."""
    # Update the query to include category and subcategory joins
    feed = db.query(Feed).options(
        joinedload(Feed.blog), 
        joinedload(Feed.slides),
        joinedload(Feed.published_feed),
        joinedload(Feed.category),
        joinedload(Feed.subcategory),
        joinedload(Feed.concepts)  # Load concepts
    ).filter(Feed.id == feed_id).first()
    
    if not feed:
        raise HTTPException(status_code=404, detail="Feed not found")

    ai_content = getattr(feed, 'ai_generated_content', {})
    
    # Determine if the feed is published
    is_published = feed.published_feed is not None
    
    # Get proper metadata
    meta = get_feed_metadata(feed, db)
    
    # Get topic descriptions
    topic_descriptions = []
    if feed.categories:
        topic_descriptions = get_topic_descriptions(feed.categories, db)
    
    # NEW: Get category and subcategory names
    category_name = feed.category.name if feed.category else None
    subcategory_name = feed.subcategory.name if feed.subcategory else None
    
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
        "topics": topic_descriptions,  # NEW: Add topic descriptions
        # NEW: Add category and subcategory info
        "category_name": category_name,
        "subcategory_name": subcategory_name,
        "category_id": feed.category_id,
        "subcategory_id": feed.subcategory_id,
        # Display as requested format: "category_name { subcategory_name }"
        "category_display": f"{category_name} {{ {subcategory_name} }}" if category_name and subcategory_name else category_name,
        
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

@router.get("/feeds/job-id", response_model=dict)
def get_feeds_by_job_id(
    job_id: Optional[str] = None,
    website_uuid: Optional[str] = None,
    page: int = 1,
    limit: int = 20,
    content_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get feeds by transcript job ID OR website UUID.
    
    For YouTube: Uses job_id to find transcripts, then feeds via transcript_id
    For Blogs: Uses website_uuid to find blogs, then feeds via blog_id
    """
    
    if not job_id and not website_uuid:
        raise HTTPException(
            status_code=400,
            detail="Either job_id or website_uuid must be provided"
        )
    
    if job_id and website_uuid:
        raise HTTPException(
            status_code=400,
            detail="Provide only one of job_id or website_uuid, not both"
        )
    
    try:
        # Base query for feeds
        query = db.query(Feed).options(
            joinedload(Feed.blog),
            joinedload(Feed.slides),
            joinedload(Feed.category),
            joinedload(Feed.subcategory)
        )
        
        source_info = {}
        source_type = None
        
        if job_id:
            # Handle YouTube transcript job ID
            logger.info(f"Searching for YouTube feeds by job ID: {job_id}")
            
            # First, find the transcript job
            transcript_job = db.query(TranscriptJob).filter(
                TranscriptJob.job_id == job_id
            ).first()
            
            if not transcript_job:
                # Try finding a single video by video_id if job_id is a video ID
                video_transcript = db.query(Transcript).filter(
                    Transcript.video_id == job_id
                ).first()
                
                if video_transcript:
                    # Query feeds for this specific transcript
                    query = query.filter(Feed.transcript_id == video_transcript.transcript_id)
                    source_info = {
                        "type": "youtube_video",
                        "identifier": job_id,
                        "title": video_transcript.title,
                        "video_id": video_transcript.video_id,
                        "transcript_id": video_transcript.transcript_id
                    }
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f"No transcript job or video found with ID: {job_id}"
                    )
            else:
                # Get all transcript IDs for this job
                transcripts = db.query(Transcript).filter(
                    Transcript.job_id == transcript_job.id
                ).all()
                
                transcript_ids = [t.transcript_id for t in transcripts]
                
                if not transcript_ids:
                    raise HTTPException(
                        status_code=404,
                        detail=f"No transcripts found for job: {job_id}"
                    )
                
                # Query feeds for these transcript IDs
                query = query.filter(Feed.transcript_id.in_(transcript_ids))
                source_info = {
                    "type": "youtube_job",
                    "identifier": job_id,
                    "job_id": transcript_job.job_id,
                    "url": transcript_job.url,
                    "content_type": transcript_job.type.value if transcript_job.type else "video",
                    "content_name": transcript_job.content_name,
                    "description": transcript_job.description,
                    "total_items": transcript_job.total_items,
                    "processed_items": transcript_job.processed_items,
                    "transcript_count": len(transcripts)
                }
            
            # Filter by source type
            query = query.filter(Feed.source_type == "youtube")
            source_type = "youtube"
        
        elif website_uuid:
            # Handle website UUID - could be website, scrape job UID, or source name
            logger.info(f"Searching for blog feeds by identifier: {website_uuid}")
            
            # FIRST: Check if this is a scrape job UID
            scrape_job = db.query(ScrapeJob).filter(
                ScrapeJob.uid == website_uuid
            ).first()
            
            if scrape_job:
                # This is a scrape job UID
                logger.info(f"Identifier {website_uuid} is a scrape job UID")
                
                # Find blogs for this website from the scrape job
                blogs = db.query(Blog).filter(
                    Blog.website == scrape_job.website
                ).all()
                
                blog_ids = [blog.id for blog in blogs]
                
                if not blog_ids:
                    raise HTTPException(
                        status_code=404,
                        detail=f"No blogs found for scrape job website: {scrape_job.website}"
                    )
                
                # Query feeds for these blog IDs
                query = query.filter(Feed.blog_id.in_(blog_ids))
                source_info = {
                    "type": "scrape_job",
                    "identifier": website_uuid,
                    "website": scrape_job.website,
                    "url": scrape_job.url,
                    "status": scrape_job.status,
                    "items_processed": getattr(scrape_job, 'items_processed', None),
                    "created_at": scrape_job.created_at.isoformat() if scrape_job.created_at else None,
                    "blog_count": len(blogs)
                }
            
            else:
                # Not a scrape job UID, check if it's a website URL/name
                # Look for blogs with this website
                blogs = db.query(Blog).filter(
                    Blog.website.ilike(f"%{website_uuid}%")
                ).all()
                
                if blogs:
                    # Found blogs by website
                    blog_ids = [blog.id for blog in blogs]
                    query = query.filter(Feed.blog_id.in_(blog_ids))
                    source_info = {
                        "type": "blog_website",
                        "identifier": website_uuid,
                        "website": blogs[0].website if blogs else website_uuid,
                        "blog_count": len(blogs)
                    }
                
                else:
                    # Check if it's a Source name
                    source = db.query(Source).filter(
                        or_(
                            Source.website.ilike(f"%{website_uuid}%"),
                            Source.name.ilike(f"%{website_uuid}%")
                        )
                    ).first()
                    
                    if source:
                        # Get blogs for this source website
                        blogs = db.query(Blog).filter(
                            Blog.website == source.website
                        ).all()
                        
                        blog_ids = [blog.id for blog in blogs]
                        
                        if not blog_ids:
                            raise HTTPException(
                                status_code=404,
                                detail=f"No blogs found for source: {source.website}"
                            )
                        
                        query = query.filter(Feed.blog_id.in_(blog_ids))
                        source_info = {
                            "type": "source",
                            "identifier": website_uuid,
                            "website": source.website,
                            "source_name": source.name,
                            "blog_count": len(blogs)
                        }
                    
                    else:
                        # Nothing found
                        raise HTTPException(
                            status_code=404,
                            detail=f"No website, source, or scrape job found with identifier: {website_uuid}"
                        )
            
            # Filter by source type
            query = query.filter(Feed.source_type == "blog")
            source_type = "blog"
        
        # Apply content type filter if provided
        if content_type:
            try:
                content_type_enum = FilterType(content_type)
                query = query.filter(Feed.content_type == content_type_enum)
            except ValueError:
                logger.warning(f"Invalid content_type provided: {content_type}")
        
        # Apply ordering and pagination
        query = query.order_by(Feed.created_at.desc())
        total = query.count()
        feeds = query.offset((page - 1) * limit).limit(limit).all()
        
        # Format the response
        feeds_data = []
        for feed in feeds:
            # Check if published
            is_published = db.query(PublishedFeed).filter(
                PublishedFeed.feed_id == feed.id,
                PublishedFeed.is_active == True
            ).first() is not None
            
            # Get proper metadata
            meta = get_feed_metadata(feed, db)
            
            # Get category and subcategory names
            category_name = feed.category.name if feed.category else None
            subcategory_name = feed.subcategory.name if feed.subcategory else None
            
            # Format content type
            feed_content_type = feed.content_type.value if feed.content_type else FilterType.BLOG.value
            
            # Get source-specific details
            feed_details = {
                "id": feed.id,
                "title": feed.title,
                "categories": feed.categories or [],
                "content_type": feed_content_type,
                "skills": getattr(feed, 'skills', []) or [],
                "tools": getattr(feed, 'tools', []) or [],
                "roles": getattr(feed, 'roles', []) or [],
                "status": feed.status,
                "source_type": feed.source_type or "blog",
                "is_published": is_published,
                "slides_count": len(feed.slides) if feed.slides else 0,
                "created_at": feed.created_at.isoformat() if feed.created_at else None,
                "updated_at": feed.updated_at.isoformat() if feed.updated_at else None,
                "ai_generated": feed.ai_generated_content is not None,
                # Category info
                "category_name": category_name,
                "subcategory_name": subcategory_name,
                "category_id": feed.category_id,
                "subcategory_id": feed.subcategory_id,
                "category_display": f"{category_name} {{ {subcategory_name} }}" if category_name and subcategory_name else category_name,
                "meta": meta
            }
            
            # Add source-specific IDs
            if feed.source_type == "blog":
                feed_details["blog_id"] = feed.blog_id
                if feed.blog:
                    feed_details["website"] = feed.blog.website
            elif feed.source_type == "youtube":
                feed_details["transcript_id"] = feed.transcript_id
                # Try to get video_id from transcript
                if feed.transcript_id:
                    transcript = db.query(Transcript).filter(
                        Transcript.transcript_id == feed.transcript_id
                    ).first()
                    if transcript:
                        feed_details["video_id"] = transcript.video_id
            
            feeds_data.append(feed_details)
        
        return {
            "source": source_info,
            "feeds": feeds_data,
            "total": total,
            "page": page,
            "limit": limit,
            "has_more": (page * limit) < total
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching feeds by source: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch feeds: {str(e)}"
        )
@router.get("/debug/youtube-lists")
def debug_youtube_lists(db: Session = Depends(get_db)):
    """Debug endpoint to check YouTube feeds and their playlist status."""
    
    # Get all YouTube feeds
    youtube_feeds = db.query(Feed).filter(
        Feed.source_type == "youtube",
        Feed.status == "ready"
    ).all()
    
    feed_info = []
    for feed in youtube_feeds:
        transcript = None
        playlist_id = None
        
        if feed.transcript_id:
            transcript = db.query(Transcript).filter(
                Transcript.transcript_id == feed.transcript_id
            ).first()
            if transcript:
                playlist_id = transcript.playlist_id
        
        feed_info.append({
            "feed_id": feed.id,
            "title": feed.title,
            "transcript_id": feed.transcript_id,
            "playlist_id": playlist_id,
            "has_transcript": transcript is not None,
            "has_playlist": bool(playlist_id)
        })
    
    # Check existing ContentList entries
    content_lists = db.query(ContentList).filter(
        ContentList.source_type == "youtube"
    ).all()
    
    list_info = []
    for clist in content_lists:
        list_info.append({
            "id": clist.id,
            "name": clist.name,
            "source_id": clist.source_id,
            "feed_count": len(clist.feed_ids) if clist.feed_ids else 0,
            "is_active": clist.is_active
        })
    
    # Find feeds with playlist_id but no ContentList
    feeds_with_playlists = [f for f in feed_info if f["has_playlist"]]
    missing_lists = []
    
    for feed in feeds_with_playlists:
        playlist_id = feed["playlist_id"]
        # Check if list exists for this playlist
        existing_list = db.query(ContentList).filter(
            ContentList.source_type == "youtube",
            ContentList.source_id == playlist_id
        ).first()
        
        if not existing_list:
            missing_lists.append({
                "feed_id": feed["feed_id"],
                "playlist_id": playlist_id,
                "title": feed["title"]
            })
    
    return {
        "total_youtube_feeds": len(youtube_feeds),
        "feeds_with_playlists": len(feeds_with_playlists),
        "existing_content_lists": len(content_lists),
        "missing_lists_count": len(missing_lists),
        "missing_lists": missing_lists[:10],  # First 10
        "feeds_sample": feed_info[:5],
        "lists_sample": list_info[:5]
    }

@router.post("/create-missing-lists")
def create_missing_youtube_lists(db: Session = Depends(get_db)):
    """Endpoint to create missing ContentList entries from YouTube feeds."""
    result = auto_create_lists_from_existing_youtube_feeds(db)
    return {
        "message": "Auto-creation completed",
        "result": result
    }

@router.get("/trigger-list-creation")
def trigger_list_creation(
    source_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Manually trigger ContentList creation from existing feeds."""
    try:
        if source_type == "youtube" or not source_type:
            result = auto_create_lists_from_existing_youtube_feeds(db)
            return {
                "status": "success",
                "message": f"Created {result.get('created', 0)} new lists and updated {result.get('updated', 0)} from YouTube feeds",
                "details": result
            }
        else:
            return {
                "status": "error",
                "message": f"Unsupported source type: {source_type}"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to create lists: {str(e)}"
        }

@router.post("/feeds/filtered", response_model=dict)
def get_filtered_feeds_post(
    response: Response,
    page: int = 1,
    limit: int = 20,
    published_status: Optional[str] = None,
    category_ids: Optional[str] = None,
    subcategory_ids: Optional[str] = None,
    search_query: Optional[str] = None,
    source_type: Optional[str] = None,
    content_type: Optional[str] = None,
    sort_by: str = "created_at",
    sort_order: str = "desc",
    date_field: str = "created_at",
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get feeds with advanced filtering using POST method with query parameters.
    """
    try:
        # Parse comma-separated IDs to lists
        category_ids_list = []
        if category_ids:
            try:
                category_ids_list = [int(id.strip()) for id in category_ids.split(',') if id.strip().isdigit()]
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid category_ids format. Use comma-separated integers."
                )
        
        subcategory_ids_list = []
        if subcategory_ids:
            try:
                subcategory_ids_list = [int(id.strip()) for id in subcategory_ids.split(',') if id.strip().isdigit()]
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid subcategory_ids format. Use comma-separated integers."
                )
        
        # Base query with joins for all related data
        query = db.query(Feed).options(
            joinedload(Feed.blog),
            joinedload(Feed.slides),
            joinedload(Feed.category),
            joinedload(Feed.subcategory),
            joinedload(Feed.concepts)
        )
        
        # Filter by published status
        if published_status:
            published_feed_ids = db.query(PublishedFeed.feed_id).filter(
                PublishedFeed.is_active == True
            )
            
            if published_status.lower() == "published":
                query = query.filter(Feed.id.in_(published_feed_ids))
            elif published_status.lower() == "unpublished":
                query = query.filter(~Feed.id.in_(published_feed_ids))
        
        # Filter by category IDs
        if category_ids_list:
            query = query.filter(Feed.category_id.in_(category_ids_list))
        
        # Filter by subcategory IDs
        if subcategory_ids_list:
            query = query.filter(Feed.subcategory_id.in_(subcategory_ids_list))
        
        # Filter by source type
        if source_type:
            query = query.filter(Feed.source_type == source_type)
        
        # Filter by content type
        if content_type:
            try:
                content_type_enum = FilterType(content_type)
                query = query.filter(Feed.content_type == content_type_enum)
            except ValueError:
                logger.warning(f"Invalid content_type provided: {content_type}")
        
        # DATE RANGE FILTERING
        if from_date or to_date:
            date_column = Feed.updated_at if date_field == "updated_at" else Feed.created_at
            
            if from_date:
                try:
                    from_datetime = datetime.fromisoformat(from_date.replace('Z', '+00:00'))
                except ValueError:
                    try:
                        from_datetime = datetime.strptime(from_date, "%Y-%m-%d")
                    except ValueError:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid from_date format: {from_date}. Use YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS"
                        )
                query = query.filter(date_column >= from_datetime)
            
            if to_date:
                try:
                    to_datetime = datetime.fromisoformat(to_date.replace('Z', '+00:00'))
                except ValueError:
                    try:
                        to_datetime = datetime.strptime(to_date, "%Y-%m-%d")
                        to_datetime = to_datetime.replace(hour=23, minute=59, second=59)
                    except ValueError:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid to_date format: {to_date}. Use YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS"
                        )
                query = query.filter(date_column <= to_datetime)
        
        # **FIXED SEARCH LOGIC**
        if search_query and len(search_query.strip()) >= 2:
            search_term = f"%{search_query.strip().lower()}%"
            
            # Create a subquery for feeds that match the search criteria
            search_subquery = db.query(Feed.id).distinct()
            
            # Apply search filters to the subquery
            search_filters = []
            
            # 1. Search in Feed table fields
            search_filters.extend([
                Feed.title.ilike(search_term),
                Feed.status.ilike(search_term),
            ])
            
            # 2. Search in JSON arrays (categories, skills, tools, roles)
            # For PostgreSQL JSONB fields, we need to use different approach
            # Assuming your database is PostgreSQL with JSONB columns
            search_filters.extend([
                Feed.categories.cast(String).ilike(search_term),
                Feed.skills.cast(String).ilike(search_term),
                Feed.tools.cast(String).ilike(search_term),
                Feed.roles.cast(String).ilike(search_term),
            ])
            
            # 3. Search in related Blog content
            search_filters.append(
                db.query(Blog.id).filter(
                    Blog.id == Feed.blog_id,
                    Blog.content.ilike(search_term)
                ).exists()
            )
            
            # 4. Search in related Transcript content
            search_filters.append(
                db.query(Transcript.id).filter(
                    Transcript.transcript_id == Feed.transcript_id,
                    Transcript.transcript_text.ilike(search_term)
                ).exists()
            )
            
            # 5. Search in related Concepts
            search_filters.append(
                db.query(Concept.id).join(FeedConcept).filter(
                    FeedConcept.feed_id == Feed.id,
                    Concept.name.ilike(search_term)
                ).exists()
            )
            
            # Apply all OR conditions to the subquery
            search_subquery = search_subquery.filter(or_(*search_filters))
            
            # Get the feed IDs that match the search
            matching_feed_ids = [row[0] for row in search_subquery.all()]
            
            # If no matches found, return empty result early
            if not matching_feed_ids:
                return {
                    "feeds": [],
                    "pagination": {
                        "page": page,
                        "limit": limit,
                        "total": 0,
                        "total_pages": 0,
                        "has_more": False
                    },
                    "filters_applied": {
                        "search_query": search_query,
                        "published_status": published_status,
                        "category_ids": category_ids,
                        "subcategory_ids": subcategory_ids,
                        "source_type": source_type,
                        "content_type": content_type,
                        "sort_by": sort_by,
                        "sort_order": sort_order,
                        "date_field": date_field,
                        "from_date": from_date,
                        "to_date": to_date
                    }
                }
            
            # Filter the main query to only include feeds that match the search
            query = query.filter(Feed.id.in_(matching_feed_ids))
        
        # Apply sorting
        if sort_by == "title":
            sort_column = Feed.title
        elif sort_by == "updated_at":
            sort_column = Feed.updated_at
        else:
            sort_column = Feed.created_at
        
        if sort_order.lower() == "asc":
            query = query.order_by(asc(sort_column))
        else:
            query = query.order_by(desc(sort_column))
        
        # Get total count before pagination
        total = query.count()
        
        # Apply pagination
        feeds = query.offset((page - 1) * limit).limit(limit).all()
        
        # Format the response
        feeds_data = []
        for feed in feeds:
            # Check if published
            is_published = db.query(PublishedFeed).filter(
                PublishedFeed.feed_id == feed.id,
                PublishedFeed.is_active == True
            ).first() is not None
            
            # Get proper metadata
            meta = get_feed_metadata(feed, db)
            
            # Get topic descriptions
            topic_descriptions = []
            if feed.categories:
                topic_descriptions = get_topic_descriptions(feed.categories, db)
            
            # Get category and subcategory names
            category_name = feed.category.name if feed.category else None
            subcategory_name = feed.subcategory.name if feed.subcategory else None
            
            # Format content type
            feed_content_type = feed.content_type.value if feed.content_type else FilterType.BLOG.value
            
            # Get concepts
            concepts = []
            if hasattr(feed, 'concepts') and feed.concepts:
                for concept in feed.concepts:
                    concepts.append({
                        "id": concept.id,
                        "name": concept.name,
                        "description": concept.description
                    })
            
            # Get all slides for this feed
            slides_data = []
            if feed.slides:
                sorted_slides = sorted(feed.slides, key=lambda x: x.order)
                for slide in sorted_slides:
                    slides_data.append({
                        "id": slide.id,
                        "order": slide.order,
                        "title": slide.title,
                        "body": slide.body,
                        "bullets": slide.bullets or [],
                        "background_color": slide.background_color,
                        "source_refs": slide.source_refs or [],
                        "render_markdown": bool(slide.render_markdown),
                        "created_at": slide.created_at.isoformat() if slide.created_at else None,
                        "updated_at": slide.updated_at.isoformat() if slide.updated_at else None
                    })
            
            # AI generated content
            ai_content = {}
            if hasattr(feed, 'ai_generated_content') and feed.ai_generated_content:
                ai_content = feed.ai_generated_content
            elif feed.title and feed.slides:
                ai_content = {
                    "title": feed.title,
                    "summary": feed.slides[0].body if feed.slides and feed.slides[0].body else "",
                    "key_points": [],
                    "conclusion": feed.slides[-1].body if feed.slides and feed.slides[-1].body else ""
                }
            
            feed_data = {
                "id": feed.id,
                "title": feed.title,
                "categories": feed.categories or [],
                "content_type": feed_content_type,
                "skills": getattr(feed, 'skills', []) or [],
                "tools": getattr(feed, 'tools', []) or [],
                "roles": getattr(feed, 'roles', []) or [],
                "status": feed.status,
                "source_type": feed.source_type or "blog",
                "is_published": is_published,
                "slides_count": len(feed.slides) if feed.slides else 0,
                "slides": slides_data,
                "ai_generated_content": ai_content,
                "created_at": feed.created_at.isoformat() if feed.created_at else None,
                "updated_at": feed.updated_at.isoformat() if feed.updated_at else None,
                "ai_generated": feed.ai_generated_content is not None,
                "category_name": category_name,
                "subcategory_name": subcategory_name,
                "category_id": feed.category_id,
                "subcategory_id": feed.subcategory_id,
                "category_display": f"{category_name} {{ {subcategory_name} }}" if category_name and subcategory_name else category_name,
                "meta": meta,
                "topics": topic_descriptions,
                "concepts": concepts
            }
            
            # Add source-specific IDs
            if feed.source_type == "blog":
                feed_data["blog_id"] = feed.blog_id
                if feed.blog:
                    feed_data["website"] = feed.blog.website
                    feed_data["author"] = getattr(feed.blog, 'author', 'Unknown')
            elif feed.source_type == "youtube":
                feed_data["transcript_id"] = feed.transcript_id
                if feed.transcript_id:
                    transcript = db.query(Transcript).filter(
                        Transcript.transcript_id == feed.transcript_id
                    ).first()
                    if transcript:
                        feed_data["video_id"] = transcript.video_id
                        feed_data["channel_name"] = meta.get("channel_name", "YouTube Creator")
            
            feeds_data.append(feed_data)
        
        # Add response headers
        response.headers["X-Total-Count"] = str(total)
        response.headers["X-Page"] = str(page)
        response.headers["X-Limit"] = str(limit)
        response.headers["X-Total-Pages"] = str((total + limit - 1) // limit)
        
        has_more = (page * limit) < total
        
        return {
            "feeds": feeds_data,
            "pagination": {
                "page": page,
                "limit": limit,
                "total": total,
                "total_pages": (total + limit - 1) // limit,
                "has_more": has_more
            },
            "filters_applied": {
                "published_status": published_status,
                "category_ids": category_ids,
                "subcategory_ids": subcategory_ids,
                "search_query": search_query,
                "source_type": source_type,
                "content_type": content_type,
                "sort_by": sort_by,
                "sort_order": sort_order,
                "date_field": date_field,
                "from_date": from_date,
                "to_date": to_date
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching filtered feeds: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch filtered feeds: {str(e)}"
        )

@router.delete("/{feed_id}", response_model=dict)
def delete_feed_by_id(feed_id: int, db: Session = Depends(get_db)):
    """
    Delete a feed by its ID along with all related data.
    This will delete:
    1. The feed record itself
    2. All slides associated with the feed
    3. Feed-Concept relationships
    4. PublishedFeed entry (if exists)
    5. ContentList associations (removes feed ID from lists)
    """
    try:
        # Start a transaction
        db.begin()
        
        # First, check if the feed exists
        feed = db.query(Feed).filter(Feed.id == feed_id).first()
        if not feed:
            raise HTTPException(
                status_code=404,
                detail=f"Feed with ID {feed_id} not found"
            )
        
        logger.info(f"Starting deletion of feed ID {feed_id} - Title: '{feed.title}'")
        
        # 1. Delete feed-concept relationships first
        concept_deleted = db.query(FeedConcept).filter(
            FeedConcept.feed_id == feed_id
        ).delete(synchronize_session=False)
        logger.info(f"Deleted {concept_deleted} concept relationships for feed {feed_id}")
        
        # 2. Delete all slides associated with this feed
        slides_deleted = db.query(Slide).filter(
            Slide.feed_id == feed_id
        ).delete(synchronize_session=False)
        logger.info(f"Deleted {slides_deleted} slides for feed {feed_id}")
        
        # 3. Delete published feed entry if exists
        published_feed_deleted = db.query(PublishedFeed).filter(
            PublishedFeed.feed_id == feed_id
        ).delete(synchronize_session=False)
        if published_feed_deleted:
            logger.info(f"Deleted published feed entry for feed {feed_id}")
        
        # 4. Remove this feed from any ContentLists it's part of
        content_lists = db.query(ContentList).filter(
            ContentList.feed_ids.isnot(None),
            ContentList.feed_ids.contains([feed_id])
        ).all()
        
        lists_updated = 0
        for content_list in content_lists:
            if content_list.feed_ids and feed_id in content_list.feed_ids:
                # Remove the feed ID from the list
                updated_feed_ids = [fid for fid in content_list.feed_ids if fid != feed_id]
                content_list.feed_ids = updated_feed_ids
                content_list.updated_at = datetime.utcnow()
                lists_updated += 1
                logger.info(f"Removed feed {feed_id} from content list '{content_list.name}'")
        
        # 5. Store feed info for logging before deletion
        feed_info = {
            "id": feed.id,
            "title": feed.title,
            "source_type": feed.source_type,
            "created_at": feed.created_at,
            "status": feed.status
        }
        
        # 6. Finally, delete the feed itself
        db.delete(feed)
        
        # Commit the transaction
        db.commit()
        
        logger.info(f"Successfully deleted feed ID {feed_id}")
        
        return {
            "status": "success",
            "message": f"Feed '{feed_info['title']}' (ID: {feed_id}) has been deleted",
            "details": {
                "feed_id": feed_info["id"],
                "title": feed_info["title"],
                "slides_deleted": slides_deleted,
                "concept_relationships_deleted": concept_deleted,
                "published_feed_deleted": bool(published_feed_deleted),
                "content_lists_updated": lists_updated,
                "source_type": feed_info["source_type"],
                "deleted_at": datetime.utcnow().isoformat()
            }
        }
        
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting feed {feed_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete feed: {str(e)}"
        )
@router.post("/delete/batch", response_model=dict)
def delete_feeds_in_batch(
    feed_ids: List[int],
    soft_delete: bool = False,
    db: Session = Depends(get_db)
):
    """
    Delete multiple feeds at once.
    
    Args:
        feed_ids: List of feed IDs to delete
        soft_delete: If True, marks feeds as deleted instead of hard delete
    """
    if not feed_ids:
        raise HTTPException(
            status_code=400,
            detail="No feed IDs provided"
        )
    
    results = {
        "successful": [],
        "failed": [],
        "not_found": []
    }
    
    try:
        for feed_id in feed_ids:
            try:
                if soft_delete:
                    # Soft delete logic
                    feed = db.query(Feed).filter(Feed.id == feed_id).first()
                    if feed:
                        feed.status = "deleted"
                        feed.is_deleted = True
                        feed.deleted_at = datetime.utcnow()
                        results["successful"].append(feed_id)
                    else:
                        results["not_found"].append(feed_id)
                else:
                    # Hard delete logic
                    feed = db.query(Feed).filter(Feed.id == feed_id).first()
                    if feed:
                        # Delete related data
                        db.query(FeedConcept).filter(
                            FeedConcept.feed_id == feed_id
                        ).delete(synchronize_session=False)
                        
                        db.query(Slide).filter(
                            Slide.feed_id == feed_id
                        ).delete(synchronize_session=False)
                        
                        db.query(PublishedFeed).filter(
                            PublishedFeed.feed_id == feed_id
                        ).delete(synchronize_session=False)
                        
                        # Delete the feed
                        db.delete(feed)
                        results["successful"].append(feed_id)
                    else:
                        results["not_found"].append(feed_id)
                        
            except Exception as e:
                logger.error(f"Error deleting feed {feed_id}: {e}")
                results["failed"].append({
                    "feed_id": feed_id,
                    "error": str(e)
                })
        
        db.commit()
        
        return {
            "status": "partial_success" if results["failed"] else "success",
            "message": f"Processed {len(feed_ids)} feeds",
            "results": results,
            "summary": {
                "total_requested": len(feed_ids),
                "successful": len(results["successful"]),
                "failed": len(results["failed"]),
                "not_found": len(results["not_found"])
            }
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error in batch delete: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch delete failed: {str(e)}"
        )
