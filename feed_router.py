from fastapi import APIRouter, Depends, HTTPException, Response, BackgroundTasks
from sqlalchemy.orm import Session, joinedload
from datetime import datetime
import os
import re
import logging
import json
from typing import List, Optional, Dict, Any
from database import get_db
from models import Blog, Category, Feed, Slide, Transcript, TranscriptJob
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from schemas import FeedRequest, DeleteSlideRequest, YouTubeFeedRequest

router = APIRouter(prefix="/get", tags=["Feeds"])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------ AI Categorization Function ------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def categorize_content_with_openai(content: str, admin_categories: list) -> list:
    """Categorize content (blog or transcript) using OpenAI into admin-defined categories."""
    try:
        truncated_content = content[:4000] + "..." if len(content) > 4000 else content
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a content categorization assistant. Analyze the content and categorize it into the provided categories. Return only the category names that best match the content as a comma-separated list. Choose at most 3 most relevant categories."
                },
                {
                    "role": "user",
                    "content": f"Available categories: {', '.join(admin_categories)}.\n\nContent:\n{truncated_content}\n\nReturn only the category names as a comma-separated list."
                }
            ],
            temperature=0.1,
            max_tokens=100
        )
        
        text = response.choices[0].message.content.strip()
        text_items = [x.strip().lower() for x in re.split(r'[,;|]', text) if x.strip()]
        
        matched_categories = []
        for item in text_items:
            for category in admin_categories:
                if (item == category.lower() or item in category.lower() or category.lower() in item):
                    matched_categories.append(category)
                    break
        
        seen = set()
        unique_categories = []
        for cat in matched_categories:
            if cat not in seen:
                seen.add(cat)
                unique_categories.append(cat)
        
        return unique_categories[:3] if unique_categories else ["Uncategorized"]
    
    except Exception as e:
        logger.error(f"OpenAI categorization error: {e}")
        return ["Uncategorized"]

# ------------------ AI Content Generation Functions ------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_slide_with_ai(slide_type: str, context: str, categories: List[str], content_type: str = "blog", previous_slides: List[Dict] = None) -> Dict[str, Any]:
    """Generate a specific slide type using AI (background color will be overridden later)."""
    try:
        content_type_context = f"This is based on {content_type} content."
        
        # Simplified system prompt - we'll handle background color separately
        system_prompt = {
            "role": "system", 
            "content": """You are a presentation design expert. Create engaging, concise slides. 
            Return JSON with: title, body, bullets (array).
            Focus on creating clear, informative content. The background color will be handled separately."""
        }
        
        messages = [
            system_prompt,
            {
                "role": "user",
                "content": f"Slide Type: {slide_type}\n{content_type_context}\nContext: {context}\nCategories: {', '.join(categories)}\nGenerate slide content in JSON format."
            }
        ]
        
        if previous_slides:
            messages[1]["content"] += f"\nPrevious Slides: {json.dumps(previous_slides[-2:])}"
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=1200
        )
        content = response.choices[0].message.content
        slide_data = json.loads(content)
        
        return {
            "title": slide_data.get("title", f"Slide {slide_type}"),
            "body": slide_data.get("body", ""),
            "bullets": slide_data.get("bullets", []),
            "background_color": "#FFFFFF",  # Default, will be overridden
            "source_refs": [],
            "render_markdown": True
        }
    except Exception as e:
        logger.error(f"OpenAI slide generation error: {e}")
        return generate_fallback_slide(slide_type, context, categories, content_type)

def generate_slides_with_ai(title: str, content: str, ai_generated_content: Dict[str, Any], categories: List[str], content_type: str = "blog") -> List[Dict]:
    """Generate random number of presentation slides (1-10) using AI with same background color for all slides."""
    import random
    
    # Determine slide count based on content richness (1-10 slides)
    content_length = len(content)
    key_points_count = len(ai_generated_content.get("key_points", []))
    summary_length = len(ai_generated_content.get("summary", ""))
    
    # Calculate content richness score
    richness_score = min(
        (content_length / 1000) +  # Content length factor
        (key_points_count * 0.5) +  # Number of key points
        (summary_length / 500),     # Summary completeness
        10.0  # Cap at 10
    )
    
    # Randomize slide count based on richness, but ensure at least 1 slide
    base_slides = max(1, min(10, int(richness_score) + random.randint(-1, 2)))
    slide_count = max(1, min(10, base_slides))  # Ensure between 1-10
    
    logger.info(f"Generating {slide_count} slides for '{title}' (richness: {richness_score:.2f})")
    
    # Generate a single background color for all slides in this feed
    background_color = generate_unified_background_color(content_type, categories)
    
    slides = []
    
    # Always start with title slide
    title_context = f"Content: {title}\nSummary: {ai_generated_content.get('summary', '')}\nType: {content_type}"
    title_slide = generate_slide_with_ai("title", title_context, categories, content_type)
    title_slide["order"] = 1
    title_slide["background_color"] = background_color  # Override with unified color
    slides.append(title_slide)
    
    # If only 1 slide needed, return just title
    if slide_count == 1:
        return slides
    
    # Add summary slide if we have at least 2 slides
    summary_context = f"Summary: {ai_generated_content.get('summary', '')}\nContent: {title}\nType: {content_type}"
    summary_slide = generate_slide_with_ai("summary", summary_context, categories, content_type, slides)
    summary_slide["order"] = 2
    summary_slide["background_color"] = background_color  # Override with unified color
    slides.append(summary_slide)
    
    # If only 2 slides needed, return title + summary
    if slide_count == 2:
        return slides
    
    # Add key points based on available content and remaining slide count
    key_points = ai_generated_content.get("key_points", [])
    remaining_slides = slide_count - 2  # Already used title and summary
    
    # Determine how many key point slides to create
    key_point_slides_count = min(remaining_slides, len(key_points), 5)  # Max 5 key point slides
    
    for i in range(key_point_slides_count):
        if i < len(key_points):
            point = key_points[i]
            key_point_context = f"Key Point: {point}\nContent: {title}\nType: {content_type}"
            key_slide = generate_slide_with_ai("key_point", key_point_context, categories, content_type, slides)
            key_slide["order"] = len(slides) + 1
            key_slide["background_color"] = background_color  # Override with unified color
            slides.append(key_slide)
    
    # If we still have slides remaining, add conclusion
    if len(slides) < slide_count:
        conclusion_context = f"Conclusion: {ai_generated_content.get('conclusion', '')}\nKey Points: {', '.join(key_points[:3])}\nType: {content_type}"
        conclusion_slide = generate_slide_with_ai("conclusion", conclusion_context, categories, content_type, slides)
        conclusion_slide["order"] = len(slides) + 1
        conclusion_slide["background_color"] = background_color  # Override with unified color
        slides.append(conclusion_slide)
    
    # Fill remaining slots with additional insights
    while len(slides) < slide_count:
        remaining_count = slide_count - len(slides)
        insight_context = f"Additional insights from: {title}\nRemaining key points: {', '.join(key_points[len(slides)-2:]) if len(slides)-2 < len(key_points) else 'Various aspects'}\nType: {content_type}"
        insight_slide = generate_slide_with_ai("additional_insights", insight_context, categories, content_type, slides)
        insight_slide["order"] = len(slides) + 1
        insight_slide["background_color"] = background_color  # Override with unified color
        slides.append(insight_slide)
    
    # Ensure we have exactly the determined number of slides
    return slides[:slide_count]

def generate_unified_background_color(content_type: str, categories: List[str]) -> str:
    """Generate a unified background color for all slides in a feed based on content type and categories."""
    import random
    
    # Color palettes based on content type and categories
    color_palettes = {
        "blog": [
            "#1a365d",  # Dark blue - professional
            "#2d3748",  # Dark gray - formal
            "#2c5530",  # Dark green - growth/learning
            "#742a2a",  # Dark red - important/urgent
            "#553c9a",  # Purple - creative/innovative
        ],
        "transcript": [
            "#2c5282",  # Blue - video/content
            "#4a5568",  # Gray-blue - neutral
            "#2b6cb0",  # Light blue - engaging
            "#1a202c",  # Very dark - professional
            "#3182ce",  # Bright blue - dynamic
        ],
        "youtube": [
            "#c53030",  # Red - YouTube brand
            "#2b6cb0",  # Blue - professional
            "#4a5568",  # Gray - neutral
            "#2d3748",  # Dark gray - formal
            "#744210",  # Brown - warm
        ]
    }
    
    # Category-based color mapping
    category_colors = {
        "technology": "#2b6cb0",      # Blue
        "business": "#2d3748",        # Dark gray
        "education": "#2c5530",       # Green
        "entertainment": "#b83280",   # Pink
        "news": "#c53030",            # Red
        "podcasts": "#744210",        # Brown
        "tutorials": "#3182ce",       # Light blue
        "reviews": "#553c9a",         # Purple
        "interviews": "#4a5568",      # Gray-blue
        "how-to": "#2c5282",          # Blue-gray
    }
    
    # Try to match categories first
    for category in categories:
        category_lower = category.lower()
        for cat_key, color in category_colors.items():
            if cat_key in category_lower:
                return color
    
    # Fall back to content type palette
    palette = color_palettes.get(content_type, color_palettes["blog"])
    return random.choice(palette)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_feed_content_with_ai(title: str, content: str, categories: List[str], content_type: str = "blog") -> Dict[str, Any]:
    """Generate engaging feed content using AI for both blogs and transcripts."""
    try:
        truncated_content = content[:6000] + "..." if len(content) > 6000 else content
        
        system_prompt = {
            "blog": """You are a content summarization and presentation expert. Create an engaging, structured feed from blog content. 
            
            Return JSON with the following structure:
            {
                "title": "Engaging title that captures the essence",
                "summary": "2-3 paragraph comprehensive summary of the main content",
                "key_points": ["Point 1", "Point 2", "Point 3", "Point 4", "Point 5"],
                "conclusion": "1-2 paragraph concluding thoughts or key takeaways"
            }
            
            Requirements:
            - Title should be compelling and under 80 characters
            - Summary should capture the main ideas comprehensively
            - Key points should be 5-7 bullet points highlighting most important aspects
            - Conclusion should provide clear takeaways
            - Maintain the original meaning and key insights""",
            
            "transcript": """You are a video content summarization expert. Create an engaging, structured feed from video transcript content. Focus on the main ideas, key takeaways, and actionable insights.
            
            Return JSON with the following structure:
            {
                "title": "Engaging title that captures the video's essence", 
                "summary": "2-3 paragraph summary of the video's main content and message",
                "key_points": ["Key insight 1", "Key insight 2", "Key insight 3", "Key insight 4", "Key insight 5"],
                "conclusion": "1-2 paragraph conclusion with main takeaways and actionable advice"
            }
            
            Requirements:
            - Title should be compelling and reflect the video's core message
            - Summary should capture the video's main narrative and purpose
            - Key points should highlight the most valuable insights from the video
            - Conclusion should provide clear takeaways and practical applications
            - Focus on actionable insights and main arguments"""
        }
        
        user_prompt = f"""
        Content Title: {title}
        Content Type: {content_type}
        Categories: {', '.join(categories)}
        
        Content:
        {truncated_content}
        
        Please generate engaging feed content in the specified JSON format.
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
        
        # Validate and ensure all required fields are present
        required_fields = ["title", "summary", "key_points", "conclusion"]
        for field in required_fields:
            if field not in ai_content:
                if field == "key_points":
                    ai_content[field] = []
                else:
                    ai_content[field] = f"Default {field} for {title}"
        
        # Ensure key_points is a list and has reasonable length
        if isinstance(ai_content["key_points"], list):
            ai_content["key_points"] = ai_content["key_points"][:7]  # Limit to 7 points max
        else:
            ai_content["key_points"] = []
        
        logger.info(f"Successfully generated AI content for {content_type}: {title}")
        return ai_content
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in AI response: {e}")
        logger.info(f"Raw AI response: {content}")
        return generate_fallback_feed_content(title, content, content_type)
    
    except Exception as e:
        logger.error(f"OpenAI feed generation error: {e}")
        return generate_fallback_feed_content(title, content, content_type)


def generate_fallback_feed_content(title: str, content: str, content_type: str = "blog") -> Dict[str, Any]:
    """Fallback content generation for both blogs and transcripts when AI fails."""
    logger.info(f"Using fallback content generation for: {title}")
    
    if content_type == "transcript":
        # For transcripts, split by sentences or natural breaks
        sentences = re.split(r'(?<=[.!?])\s+', content)
        # Filter out very short sentences and take meaningful ones
        meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
        
        if meaningful_sentences:
            summary = meaningful_sentences[0]
            # Use next 5-7 sentences as key points, ensuring they're substantial
            key_points = []
            for sentence in meaningful_sentences[1:6]:
                if len(sentence) > 40 and len(key_points) < 5:
                    key_points.append(sentence)
        else:
            summary = f"A comprehensive summary of the video: {title}"
            key_points = [
                f"Key insight about {title}",
                f"Important point from the video",
                f"Main takeaway from {title}",
                f"Valuable information shared",
                f"Core message of the content"
            ]
    else:
        # For blogs, use paragraph-based approach
        paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 50]
        
        if paragraphs:
            summary = paragraphs[0]
            # Use next paragraphs as key points, taking first sentence from each
            key_points = []
            for paragraph in paragraphs[1:6]:
                if len(paragraph) > 30 and len(key_points) < 5:
                    # Extract first sentence or first 100 characters
                    first_sentence = re.split(r'(?<=[.!?]) +', paragraph)[0] if re.search(r'[.!?]', paragraph) else paragraph[:120] + "..."
                    key_points.append(first_sentence)
        else:
            summary = f"An overview of the blog post: {title}"
            key_points = [
                f"Main idea from {title}",
                f"Important concept discussed",
                f"Key finding in the content",
                f"Primary argument made",
                f"Essential information shared"
            ]
    
    # Ensure we have at least 3 key points
    while len(key_points) < 3:
        key_points.append(f"Additional insight from {title}")
    
    return {
        "title": f"Summary: {title}",
        "summary": summary,
        "key_points": key_points[:7],  # Max 7 points
        "conclusion": f"Key insights and main takeaways from {title}. This content provides valuable information on {', '.join(key_points[:2])}."
    }

# ------------------ Core Feed Creation Functions ------------------

def create_feed_from_blog(db: Session, blog: Blog):
    """Generate feed and slides from a blog using AI and store in DB with random slide count (1-10)."""
    try:
        # Check if feed already exists for this blog
        existing_feed = db.query(Feed).filter(Feed.blog_id == blog.id).first()
        if existing_feed:
            # Delete existing slides to regenerate
            db.query(Slide).filter(Slide.feed_id == existing_feed.id).delete()
            db.flush()
        
        # Create new feed
        admin_categories = [c.name for c in db.query(Category).filter(Category.is_active == True).all()]
        categories = categorize_content_with_openai(blog.content, admin_categories)
        ai_generated_content = generate_feed_content_with_ai(blog.title, blog.content, categories, "blog")
        
        # Generate slides with random count between 1-10 based on content
        slides_data = generate_slides_with_ai(blog.title, blog.content, ai_generated_content, categories, "blog")
        
        feed_title = ai_generated_content.get("title", blog.title)
        feed = Feed(
            blog_id=blog.id, 
            title=feed_title,
            categories=categories, 
            status="ready",
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
        logger.info(f"Successfully created AI-generated feed {feed.id} for blog {blog.id} with {len(slides_data)} slides")
        return feed
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating AI-generated feed for blog {blog.id}: {e}")
        raise

def create_feed_from_transcript(db: Session, transcript: Transcript, overwrite: bool = False):
    """Generate feed and slides from a YouTube transcript using AI and store in DB with random slide count (1-10)."""
    try:
        # Check if feed already exists for this transcript
        existing_feed = db.query(Feed).filter(Feed.transcript_id == transcript.transcript_id).first()
        
        if existing_feed and not overwrite:
            logger.info(f"Feed already exists for transcript {transcript.transcript_id}, skipping")
            return existing_feed
            
        admin_categories = [c.name for c in db.query(Category).filter(Category.is_active == True).all()]
        categories = categorize_content_with_openai(transcript.transcript_text, admin_categories)
        ai_generated_content = generate_feed_content_with_ai(transcript.title, transcript.transcript_text, categories, "transcript")
        
        # Generate slides with random count between 1-10 based on content
        slides_data = generate_slides_with_ai(transcript.title, transcript.transcript_text, ai_generated_content, categories, "transcript")
        
        feed_title = ai_generated_content.get("title", transcript.title)
        
        if existing_feed and overwrite:
            # UPDATE existing feed instead of deleting
            existing_feed.title = feed_title
            existing_feed.categories = categories
            existing_feed.status = "ready"
            existing_feed.ai_generated_content = ai_generated_content
            existing_feed.updated_at = datetime.utcnow()
            
            # Delete old slides
            db.query(Slide).filter(Slide.feed_id == existing_feed.id).delete()
            db.flush()
            
            # Create new slides with random count
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
            logger.info(f"Successfully UPDATED AI-generated feed {existing_feed.id} for transcript {transcript.transcript_id} with {len(slides_data)} slides")
            return existing_feed
        else:
            # CREATE new feed
            feed = Feed(
                transcript_id=transcript.transcript_id,
                title=feed_title,
                categories=categories, 
                status="ready",
                ai_generated_content=ai_generated_content,
                image_generation_enabled=False,
                source_type="youtube",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.add(feed)
            db.flush()
            
            # Create slides with random count
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
            logger.info(f"Successfully CREATED AI-generated feed {feed.id} for transcript {transcript.transcript_id} with {len(slides_data)} slides")
            return feed
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating/updating AI-generated feed for transcript {transcript.transcript_id}: {e}")
        raise
# ------------------ Helper Functions ------------------

def generate_fallback_feed_content(title: str, content: str, content_type: str = "blog") -> Dict[str, Any]:
    """Fallback content generation for both blogs and transcripts."""
    if content_type == "transcript":
        # For transcripts, split by sentences or natural breaks
        sentences = re.split(r'(?<=[.!?])\s+', content)
        summary = sentences[0] if sentences else f"A summary of {title}"
        
        key_points = []
        for i, sentence in enumerate(sentences[1:6] if len(sentences) > 1 else sentences[:5]):
            if len(sentence) > 20:  # Only use substantial sentences
                key_points.append(sentence)
        
        while len(key_points) < 3:
            key_points.append(f"Key insight from {title}")
    else:
        # For blogs, use paragraph-based approach
        paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 50]
        summary = paragraphs[0] if paragraphs else f"An overview of {title}"
        
        key_points = []
        for i, p in enumerate(paragraphs[1:4] if len(paragraphs) > 1 else paragraphs[:3]):
            first_sentence = re.split(r'(?<=[.!?]) +', p)[0] if re.search(r'[.!?]', p) else p[:100] + "..."
            key_points.append(first_sentence)
    
    while len(key_points) < 3:
        key_points.append(f"Important aspect of {title}")
    
    return {
        "title": f"Summary: {title}",
        "summary": summary,
        "key_points": key_points[:5],
        "conclusion": f"Key insights from {title}"
    }

def generate_fallback_slide(slide_type: str, context: str, categories: List[str], content_type: str = "blog") -> Dict[str, Any]:
    """Fallback slide generation with background colors."""
    content_type_label = "Video" if content_type == "transcript" else "Blog"
    
    # Default background colors for fallback slides
    fallback_colors = {
        "title": "#1a365d",  # Dark blue
        "summary": "#2d3748",  # Dark gray
        "key_point": "#4a5568",  # Medium gray
        "conclusion": "#2c5282",  # Blue gray
        "additional_insights": "#4a5568"  # Medium gray
    }
    
    return {
        "title": f"{slide_type.replace('_', ' ').title()} - {content_type_label}",
        "body": context[:500] + "..." if len(context) > 500 else context,
        "bullets": None,
        "background_color": fallback_colors.get(slide_type, "#FFFFFF"),
        "source_refs": [],
        "render_markdown": True
    }

def create_basic_feed_from_blog(db: Session, blog: Blog):
    """Fallback method to create basic feed without AI - limit to 5 slides."""
    admin_categories = [c.name for c in db.query(Category).filter(Category.is_active == True).all()]
    categories = categorize_content_with_openai(blog.content, admin_categories)
    
    # Limit to 5 paragraphs/slides
    paragraphs = blog.content.split("\n\n")[:5]
    slides = []
    
    for idx, p in enumerate(paragraphs):
        slides.append({
            "order": idx + 1,
            "title": blog.title if idx == 0 else f"Key Point {idx}",
            "body": p[:500],  # Limit body length
            "bullets": None,
            # "background_image_url": None,
            "source_refs": [],
            "render_markdown": True
        })
    
    feed = Feed(
        blog_id=blog.id, 
        categories=categories, 
        status="ready", 
        title=blog.title,
        image_generation_enabled=False,
        source_type="blog"
    )
    db.add(feed)
    db.flush()
    
    for s in slides:
        slide = Slide(
            feed_id=feed.id,
            order=s["order"],
            title=s["title"],
            body=s["body"],
            bullets=s.get("bullets"),
            # background_image_url=None,
            source_refs=s.get("source_refs", []),
            render_markdown=int(s.get("render_markdown", True))
        )
        db.add(slide)
    
    db.commit()
    db.refresh(feed)
    return feed

def create_basic_feed_from_transcript(db: Session, transcript: Transcript):
    """Fallback method to create basic feed from transcript without AI - limit to 5 slides."""
    admin_categories = [c.name for c in db.query(Category).filter(Category.is_active == True).all()]
    categories = categorize_content_with_openai(transcript.transcript_text, admin_categories)
    
    # Split transcript into meaningful chunks and limit to 5
    chunks = re.split(r'(?<=[.!?])\s+', transcript.transcript_text)
    chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 30][:5]
    
    slides = []
    for idx, chunk in enumerate(chunks):
        slides.append({
            "order": idx + 1,
            "title": transcript.title if idx == 0 else f"Key Insight {idx}",
            "body": chunk[:500],  # Limit body length
            "bullets": None,
            # "background_image_url": None,
            "source_refs": [],
            "render_markdown": True
        })
    
    feed = Feed(
        transcript_id=transcript.transcript_id,
        categories=categories, 
        status="ready", 
        title=transcript.title,
        image_generation_enabled=False,
        source_type="youtube"
    )
    db.add(feed)
    db.flush()
    
    for s in slides:
        slide = Slide(
            feed_id=feed.id,
            order=s["order"],
            title=s["title"],
            body=s["body"],
            bullets=s.get("bullets"),
            # background_image_url=None,
            source_refs=s.get("source_refs", []),
            render_markdown=int(s.get("render_markdown", True))
        )
        db.add(slide)
    
    db.commit()
    db.refresh(feed)
    return feed

# ------------------ Background Processing Functions ------------------

def process_blog_feeds_creation(db: Session, blogs: List[Blog], website: str, overwrite: bool = False, use_ai: bool = True):
    """Background task to process blog feed creation."""
    from database import SessionLocal
    db = SessionLocal()
    try:
        created_count = 0
        skipped_count = 0
        error_count = 0
        
        for blog in blogs:
            try:
                existing_feed = db.query(Feed).filter(Feed.blog_id == blog.id).first()
                if existing_feed and not overwrite:
                    skipped_count += 1
                    continue
                
                if existing_feed and overwrite:
                    db.delete(existing_feed)
                    db.flush()
                
                if use_ai:
                    feed = create_feed_from_blog(db, blog)
                else:
                    feed = create_basic_feed_from_blog(db, blog)
                
                if feed:
                    created_count += 1
            except Exception as e:
                error_count += 1
                logger.error(f"Error processing blog {blog.id}: {e}")
                continue
        
        logger.info(f"Completed blog feed creation for {website}: {created_count} created, {skipped_count} skipped, {error_count} errors")
    finally:
        db.close()

def process_transcript_feeds_creation(db: Session, transcripts: List[Transcript], job_id: str, overwrite: bool = False, use_ai: bool = True):
    """Background task to process transcript feed creation."""
    from database import SessionLocal
    db = SessionLocal()
    try:
        created_count = 0
        skipped_count = 0
        error_count = 0
        
        for transcript in transcripts:
            try:
                existing_feed = db.query(Feed).filter(Feed.transcript_id == transcript.transcript_id).first()
                if existing_feed and not overwrite:
                    skipped_count += 1
                    continue
                
                if existing_feed and overwrite:
                    db.delete(existing_feed)
                    db.flush()
                
                if use_ai:
                    feed = create_feed_from_transcript(db, transcript)
                else:
                    feed = create_basic_feed_from_transcript(db, transcript)
                
                if feed:
                    created_count += 1
            except Exception as e:
                error_count += 1
                logger.error(f"Error processing transcript {transcript.transcript_id}: {e}")
                continue
        
        logger.info(f"Completed transcript feed creation for job {job_id}: {created_count} created, {skipped_count} skipped, {error_count} errors")
    finally:
        db.close()

# ------------------ API Endpoints ------------------

@router.get("/feeds/all", response_model=dict)
def get_all_feeds(
    response: Response, 
    page: int = 1, 
    limit: int = 20, 
    category: Optional[str] = None,
    status: Optional[str] = None,
    source_type: Optional[str] = None,
    is_published: Optional[bool] = None,  # New filter parameter
    db: Session = Depends(get_db)
):
    """Get all feeds summary with filtering options including is_published."""
    query = db.query(Feed).options(joinedload(Feed.blog), joinedload(Feed.published_feed))
    
    if category:
        query = query.filter(Feed.categories.contains([category]))
    if status:
        query = query.filter(Feed.status == status)
    if source_type:
        query = query.filter(Feed.source_type == source_type)
    if is_published is not None:
        if is_published:
            query = query.filter(Feed.published_feed != None)  # Has published_feed relationship
        else:
            query = query.filter(Feed.published_feed == None)  # No published_feed relationship
    
    query = query.order_by(Feed.created_at.desc())
    total = query.count()
    feeds = query.offset((page - 1) * limit).limit(limit).all()

    items = []
    for f in feeds:
        # Determine is_published status
        is_published_status = f.published_feed is not None
        
        # Determine source metadata based on source_type
        if f.source_type == "youtube":
            meta = {
                "title": f.title,
                "original_title": f.title,
                "author": "YouTube Creator",
                "source_url": f"https://www.youtube.com/watch?v={f.transcript_id}" if f.transcript_id else "#",
                "source_type": "youtube"
            }
        else:
            meta = {
                "title": f.title,
                "original_title": f.blog.title if f.blog else "Unknown",
                "author": getattr(f.blog, 'author', 'Admin'),
                "source_url": getattr(f.blog, 'url', '#'),
                "source_type": "blog"
            }

        items.append({
            "id": f.id,
            "blog_id": f.blog_id,
            "transcript_id": f.transcript_id,
            "title": f.title,
            "categories": f.categories,
            "status": f.status,
            "source_type": f.source_type or "blog",
            "is_published": is_published_status,  # New field
            "published_at": f.published_feed.published_at.isoformat() if is_published_status else None,
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
        "items": items,
        "page": page, 
        "limit": limit, 
        "total": total, 
        "has_more": has_more,
        "filters": {
            "category": category, 
            "status": status, 
            "source_type": source_type,
            "is_published": is_published
        }
    }
@router.get("/feeds/{feed_id}", response_model=dict)
def get_feed_by_id(feed_id: int, db: Session = Depends(get_db)):
    """Get full AI-generated feed with slides and is_published status."""
    feed = db.query(Feed).options(
        joinedload(Feed.blog), 
        joinedload(Feed.slides),
        joinedload(Feed.published_feed)  # Eager load published_feed relationship
    ).filter(Feed.id == feed_id).first()
    
    if not feed:
        raise HTTPException(status_code=404, detail="Feed not found")

    ai_content = getattr(feed, 'ai_generated_content', {})
    
    # Determine if the feed is published
    is_published = feed.published_feed is not None
    
    # Determine source metadata based on source_type
    if feed.source_type == "youtube":
        meta = {
            "title": feed.title,
            "original_title": feed.title,
            "author": "YouTube Creator",
            "source_url": f"https://www.youtube.com/watch?v={feed.transcript_id}" if feed.transcript_id else "#",
            "source_type": "youtube"
        }
    else:
        meta = {
            "title": feed.title,
            "original_title": feed.blog.title if feed.blog else "Unknown",
            "author": getattr(feed.blog, 'author', 'Admin'),
            "source_url": getattr(feed.blog, 'url', '#'),
            "source_type": "blog"
        }
    
    return {
        "id": feed.id,
        "blog_id": feed.blog_id,
        "transcript_id": feed.transcript_id,
        "title": feed.title,
        "categories": feed.categories,
        "status": feed.status,
        "source_type": feed.source_type or "blog",
        "ai_generated_content": ai_content,
        "is_published": is_published,  # New field
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
    """Create feeds for all blogs from a website (AI data only, no images)."""
    blogs = db.query(Blog).filter(Blog.website == request.website).all()
    if not blogs:
        raise HTTPException(status_code=404, detail="No blogs found for this website")

    background_tasks.add_task(
        process_blog_feeds_creation,
        db,
        blogs,
        request.website,
        request.overwrite,
        request.use_ai,
    )

    return {
        "website": request.website,
        "total_blogs": len(blogs),
        "use_ai": request.use_ai,
        "generate_images": False,
        "source_type": "blog",
        "message": "Blog feed creation process started in background",
        "status": "processing"
    }

@router.post("/feeds/youtube", response_model=dict)
def create_feeds_from_youtube(
    request: YouTubeFeedRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Create feeds from YouTube transcripts (AI data only, no images)."""
    transcripts = []
    job_identifier = "all_transcripts"
    
    if request.job_id:
        # First try to find the TranscriptJob by its job_id string
        transcript_job = db.query(TranscriptJob).filter(TranscriptJob.job_id == request.job_id).first()
        
        if transcript_job:
            # If found, get all transcripts using the INTEGER id (foreign key)
            transcripts = db.query(Transcript).filter(Transcript.job_id == transcript_job.id).all()
            job_identifier = f"job_{request.job_id}"
        else:
            # If no job found, check if it's a video_id
            video_transcript = db.query(Transcript).filter(Transcript.video_id == request.job_id).first()
            if video_transcript:
                transcripts = [video_transcript]
                job_identifier = f"video_{request.job_id}"
            else:
                raise HTTPException(
                    status_code=404, 
                    detail=f"No transcripts found for job ID: {request.job_id}. Available jobs: {[job.job_id for job in db.query(TranscriptJob).all()]}"
                )
    
    elif request.video_id:
        # Get specific video transcript by video_id
        transcripts = db.query(Transcript).filter(Transcript.video_id == request.video_id).all()
        job_identifier = f"video_{request.video_id}"
        if not transcripts:
            raise HTTPException(status_code=404, detail=f"No transcript found for video ID: {request.video_id}")
    else:
        # Get all available transcripts
        transcripts = db.query(Transcript).all()
        if not transcripts:
            raise HTTPException(status_code=404, detail="No transcripts found in database")

    if not transcripts:
        raise HTTPException(status_code=404, detail="No transcripts found for the given criteria")

    background_tasks.add_task(
        process_transcript_feeds_creation,
        db,
        transcripts,
        job_identifier,
        request.overwrite,
        request.use_ai,
    )

    return {
        "job_id": request.job_id,
        "video_id": request.video_id,
        "total_transcripts": len(transcripts),
        "use_ai": request.use_ai,
        "generate_images": False,
        "source_type": "youtube",
        "message": f"YouTube transcript feed creation process started for {len(transcripts)} transcripts",
        "status": "processing"
    }

@router.delete("/feeds/slides", response_model=dict)
def delete_slide_from_feed(
    request: DeleteSlideRequest, 
    db: Session = Depends(get_db)
):
    """Delete a specific slide from a feed using request body."""
    feed_id = request.feed_id
    slide_id = request.slide_id

    # First verify the feed exists
    feed = db.query(Feed).filter(Feed.id == feed_id).first()
    if not feed:
        raise HTTPException(status_code=404, detail="Feed not found")
    
    # Find the specific slide
    slide = db.query(Slide).filter(
        Slide.id == slide_id, 
        Slide.feed_id == feed_id
    ).first()
    
    if not slide:
        raise HTTPException(status_code=404, detail="Slide not found in this feed")
    
    try:
        # Delete the slide
        db.delete(slide)
        db.commit()
        
        logger.info(f"Successfully deleted slide {slide_id} from feed {feed_id}")
        
        return {
            "message": "Slide deleted successfully",
            "feed_id": feed_id,
            "slide_id": slide_id,
            "deleted_slide_title": slide.title
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting slide {slide_id} from feed {feed_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete slide")

@router.get("/feeds/source/{website}/categorized", response_model=dict)
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
            items.append({
                "id": f.id,
                "blog_id": f.blog_id,
                "title": f.title,
                "categories": f.categories,
                "status": f.status,
                "slides_count": len(f.slides),
                "meta": {
                    "title": f.title,
                    "original_title": f.blog.title if f.blog else "Unknown",
                    "author": getattr(f.blog, 'author', 'Admin'),
                    "source_url": getattr(f.blog, 'url', '#'),
                    "website": website
                },
                "created_at": f.created_at.isoformat() if f.created_at else None,
                "updated_at": f.updated_at.isoformat() if f.updated_at else None,
                "ai_generated": hasattr(f, 'ai_generated_content') and f.ai_generated_content is not None,
                "images_generated": False  # Always false now
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