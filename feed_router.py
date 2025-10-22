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
    """Categorize content using OpenAI - no fallbacks, raises error on failure."""
    try:
        truncated_content = content[:4000] + "..." if len(content) > 4000 else content
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a content categorization assistant. Analyze the content and categorize it into the provided categories. Return only the category names that best match the content as a comma-separated list. Choose at most 3 most relevant categories. Do NOT return 'Uncategorized'."
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
        
        if not unique_categories:
            raise HTTPException(status_code=500, detail="OpenAI failed to categorize content into any available categories")
        
        return unique_categories[:3]
    
    except Exception as e:
        logger.error(f"OpenAI categorization error: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI categorization failed: {str(e)}")

# ------------------ AI Content Generation Functions ------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_feed_content_with_ai(title: str, content: str, categories: List[str], content_type: str = "blog") -> Dict[str, Any]:
    """Generate engaging feed content using AI - no fallbacks, raises error on failure."""
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
            - Title should be compelling and different from the original title
            - Summary should capture the main ideas comprehensively in 2-3 paragraphs
            - Key points should be 5-7 distinct bullet points highlighting most important aspects
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
            - Title should be compelling and reflect the video's core message (don't start with "Summary:")
            - Summary should capture the video's main narrative and purpose in 2-3 paragraphs
            - Key points should highlight the most valuable, distinct insights from the video (5-7 points)
            - Conclusion should provide clear takeaways and practical applications
            - Focus on actionable insights and main arguments, not transcript repetition"""
        }
        
        user_prompt = f"""
        Original Title: {title}
        Content Type: {content_type}
        Categories: {', '.join(categories)}
        
        Content:
        {truncated_content}
        
        Please generate engaging, structured feed content in the specified JSON format.
        Focus on creating unique, valuable content that summarizes and enhances the original.
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
        
        # Validate and ensure all required fields are present and meaningful
        required_fields = ["title", "summary", "key_points", "conclusion"]
        for field in required_fields:
            if field not in ai_content:
                raise HTTPException(status_code=500, detail=f"OpenAI response missing required field: {field}")
            
            # Validate content quality
            if field == "title" and ("summary:" in ai_content[field].lower() or ai_content[field].lower().startswith("summary")):
                raise HTTPException(status_code=500, detail="OpenAI generated poor quality title starting with 'Summary:'")
            
            if field == "key_points":
                if not isinstance(ai_content[field], list):
                    raise HTTPException(status_code=500, detail="OpenAI response key_points is not a list")
                if len(ai_content[field]) < 3:
                    raise HTTPException(status_code=500, detail="OpenAI generated insufficient key points")
                # Remove duplicate or low-quality key points
                unique_points = []
                for point in ai_content[field]:
                    point_str = str(point).strip()
                    if (len(point_str) > 20 and 
                        point_str not in unique_points and 
                        not point_str.lower().startswith("key insight from")):
                        unique_points.append(point_str)
                ai_content[field] = unique_points[:7]
                
            if field == "summary" and len(ai_content[field].split()) < 50:
                raise HTTPException(status_code=500, detail="OpenAI generated insufficient summary content")
        
        logger.info(f"Successfully generated AI content for {content_type}: {ai_content['title']}")
        return ai_content
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in AI response: {e}")
        logger.info(f"Raw AI response: {content}")
        raise HTTPException(status_code=500, detail=f"OpenAI returned invalid JSON: {str(e)}")
    
    except Exception as e:
        logger.error(f"OpenAI feed generation error: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI feed generation failed: {str(e)}")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_slide_with_ai(slide_type: str, context: str, categories: List[str], content_type: str = "blog", previous_slides: List[Dict] = None) -> Dict[str, Any]:
    """Generate a specific slide type using AI - no fallbacks, raises error on failure."""
    try:
        # Enhanced system prompt to ensure proper JSON response
        system_prompt = {
            "role": "system", 
            "content": f"""You are a presentation design expert. Create engaging, concise slides.
            You MUST return ONLY valid JSON with the following structure:
            {{
                "title": "Engaging slide title",
                "body": "Substantive body content that summarizes key points",
                "bullets": ["Bullet point 1", "Bullet point 2", "Bullet point 3"]
            }}
            
            CRITICAL REQUIREMENTS:
            - Return ONLY valid JSON, no other text
            - Create unique, engaging titles (NOT generic like "Title - Video")
            - Provide substantive body content (at least 50 words)
            - Use bullets array for key points
            - Avoid repeating content across slides
            
            Slide type: {slide_type}
            Content type: {content_type}
            """
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
            response_format={ "type": "json_object" }  # Force JSON response
        )
        
        content = response.choices[0].message.content
        
        # Log the raw response for debugging
        # logger.info(f"Raw OpenAI response for {slide_type}: {content[:200]}...")
        
        try:
            slide_data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error. Raw response: {content}")
            raise HTTPException(status_code=500, detail=f"OpenAI returned invalid JSON for {slide_type}: {str(e)}")
        
        # Validate and ensure we have proper content
        title = slide_data.get("title", "").strip()
        body = slide_data.get("body", "").strip()
        bullets = slide_data.get("bullets", [])
        
        # Validate content quality
        if not title or title.lower() in ["slide title", "title - video", "title - blog", "untitled"]:
            logger.error(f"Poor quality title generated: '{title}'")
            raise HTTPException(status_code=500, detail=f"OpenAI generated poor quality title: '{title}'")
        
        if not body or len(body) < 50:
            logger.error(f"Insufficient body content: '{body}'")
            raise HTTPException(status_code=500, detail=f"OpenAI generated insufficient body content for slide: {title}")
        
        # Ensure bullets is a list
        if not isinstance(bullets, list):
            bullets = []
        
        # logger.info(f"Successfully generated slide: {title}")
        return {
            "title": title,
            "body": body,
            "bullets": bullets,
            "background_color": "#FFFFFF",  # Will be overridden with feed's unique color
            "source_refs": [],
            "render_markdown": True
        }
        
    except HTTPException:
        # Re-raise HTTPExceptions
        raise
    except Exception as e:
        logger.error(f"OpenAI slide generation error for {slide_type}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OpenAI slide generation failed for {slide_type}: {str(e)}")

def generate_slides_with_ai(title: str, content: str, ai_generated_content: Dict[str, Any], categories: List[str], content_type: str = "blog") -> List[Dict]:
    """Generate random number of presentation slides (1-10) using AI with same color for all slides in a feed."""
    import random
    import hashlib
    
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
    
    # Generate a UNIQUE background color for THIS SPECIFIC FEED (same for all slides in this feed)
    background_color = generate_feed_background_color(title, content_type, categories)
    
    slides = []
    
    try:
        # 1. Title slide
        title_context = f"Create an engaging title slide for: {title}\nSummary: {ai_generated_content.get('summary', '')}\nCategories: {', '.join(categories)}\nType: {content_type}"
        title_slide = generate_slide_with_ai("title", title_context, categories, content_type)
        title_slide["order"] = 1
        title_slide["background_color"] = background_color
        slides.append(title_slide)
        
        # If only 1 slide needed, return just title
        if slide_count == 1:
            return slides
        
        # 2. Summary slide
        summary_context = f"Create a comprehensive summary slide for: {title}\nFull Summary: {ai_generated_content.get('summary', '')}\nType: {content_type}"
        summary_slide = generate_slide_with_ai("summary", summary_context, categories, content_type, slides)
        summary_slide["order"] = 2
        summary_slide["background_color"] = background_color
        slides.append(summary_slide)
        
        # If only 2 slides needed, return title + summary
        if slide_count == 2:
            return slides
        
        # 3. Key point slides
        key_points = ai_generated_content.get("key_points", [])
        remaining_slides = slide_count - 2  # Already used title and summary
        
        # Determine how many key point slides to create
        key_point_slides_count = min(remaining_slides, len(key_points), 5)  # Max 5 key point slides
        
        for i in range(key_point_slides_count):
            if i < len(key_points):
                point = key_points[i]
                key_point_context = f"Create a detailed key point slide for: {title}\nKey Point: {point}\nOther Key Points: {', '.join(key_points[:i] + key_points[i+1:][:2])}\nType: {content_type}"
                key_slide = generate_slide_with_ai("key_point", key_point_context, categories, content_type, slides)
                key_slide["order"] = len(slides) + 1
                key_slide["background_color"] = background_color
                slides.append(key_slide)
        
        # 4. Conclusion slide if we have space
        if len(slides) < slide_count and ai_generated_content.get("conclusion"):
            conclusion_context = f"Create a conclusion slide for: {title}\nConclusion: {ai_generated_content.get('conclusion', '')}\nKey Points Covered: {', '.join(key_points[:3])}\nType: {content_type}"
            conclusion_slide = generate_slide_with_ai("conclusion", conclusion_context, categories, content_type, slides)
            conclusion_slide["order"] = len(slides) + 1
            conclusion_slide["background_color"] = background_color
            slides.append(conclusion_slide)
        
        # 5. Fill remaining slots with additional insights
        while len(slides) < slide_count:
            insight_context = f"Create an additional insights slide for: {title}\nAvailable Content: Summary - {ai_generated_content.get('summary', '')[:200]}...\nRemaining Key Points: {', '.join(key_points[len(slides)-2:]) if len(slides)-2 < len(key_points) else 'Various important aspects'}\nType: {content_type}"
            insight_slide = generate_slide_with_ai("additional_insights", insight_context, categories, content_type, slides)
            insight_slide["order"] = len(slides) + 1
            insight_slide["background_color"] = background_color
            slides.append(insight_slide)
        
        logger.info(f"Successfully generated {len(slides)} slides for '{title}'")
        return slides[:slide_count]
        
    except Exception as e:
        logger.error(f"Error in generate_slides_with_ai for '{title}': {str(e)}")
        raise

def generate_feed_background_color(title: str, content_type: str, categories: List[str]) -> str:
    """Generate background color from a predefined set of distinct colors."""
    import hashlib
    
    # Predefined set of visually distinct colors
    distinct_colors = [
        "#1e3a8a",  # Dark blue
        "#166534",  # Dark green  
        "#581c87",  # Dark purple
        "#991b1b",  # Dark red
        "#9a3412",  # Dark orange
        "#115e59",  # Dark teal
        "#831843",  # Dark pink
        "#854d0e",  # Dark yellow
        "#3730a3",  # Dark indigo
        "#064e3b",  # Dark emerald
        "#1d4ed8",  # Medium blue
        "#15803d",  # Medium green
        "#7c3aed",  # Medium purple
        "#dc2626",  # Medium red
        "#ea580c",  # Medium orange
        "#0f766e",  # Medium teal
        "#be185d",  # Medium pink
        "#ca8a04",  # Medium yellow
        "#4f46e5",  # Medium indigo
        "#047857",  # Medium emerald
    ]
    
    # Create a unique seed
    seed_string = f"{title}_{content_type}_{'_'.join(sorted(categories))}"
    hash_object = hashlib.md5(seed_string.encode())
    hash_int = int(hash_object.hexdigest()[:8], 16)
    
    # Select color from the predefined list
    color_index = hash_int % len(distinct_colors)
    color_hex = distinct_colors[color_index]
    
    logger.info(f"Generated predefined distinct color: {color_hex} for '{title}' (index: {color_index})")
    return color_hex

# ------------------ Core Feed Creation Functions ------------------

def create_feed_from_blog(db: Session, blog: Blog):
    """Generate feed and slides from a blog using AI - no fallbacks, raises error on failure."""
    try:
        # Check if feed already exists for this blog
        existing_feed = db.query(Feed).filter(Feed.blog_id == blog.id).first()
        if existing_feed:
            # Delete existing slides to regenerate
            db.query(Slide).filter(Slide.feed_id == existing_feed.id).delete()
            db.flush()
        
        # Create new feed - NO FALLBACKS, will raise error if OpenAI fails
        admin_categories = [c.name for c in db.query(Category).filter(Category.is_active == True).all()]
        if not admin_categories:
            raise HTTPException(status_code=500, detail="No active categories found in database")
            
        categories = categorize_content_with_openai(blog.content, admin_categories)
        ai_generated_content = generate_feed_content_with_ai(blog.title, blog.content, categories, "blog")
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
        
    except HTTPException:
        # Re-raise HTTPExceptions
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating AI-generated feed for blog {blog.id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create feed for blog: {str(e)}")

def create_feed_from_transcript(db: Session, transcript: Transcript, overwrite: bool = False):
    """Generate feed and slides from a YouTube transcript using AI - no fallbacks, raises error on failure."""
    try:
        # Check if feed already exists for this transcript
        existing_feed = db.query(Feed).filter(Feed.transcript_id == transcript.transcript_id).first()
        
        if existing_feed and not overwrite:
            logger.info(f"Feed already exists for transcript {transcript.transcript_id}, skipping")
            return existing_feed
            
        # NO FALLBACKS - will raise error if OpenAI fails
        admin_categories = [c.name for c in db.query(Category).filter(Category.is_active == True).all()]
        if not admin_categories:
            raise HTTPException(status_code=500, detail="No active categories found in database")
            
        categories = categorize_content_with_openai(transcript.transcript_text, admin_categories)
        ai_generated_content = generate_feed_content_with_ai(transcript.title, transcript.transcript_text, categories, "transcript")
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
            logger.info(f"Successfully CREATED AI-generated feed {feed.id} for transcript {transcript.transcript_id} with {len(slides_data)} slides")
            return feed
        
    except HTTPException:
        # Re-raise HTTPExceptions
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating/updating AI-generated feed for transcript {transcript.transcript_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create feed for transcript: {str(e)}")

# ... (rest of the code remains the same for API endpoints)
# ------------------ Background Processing Functions ------------------

# ------------------ Background Processing Functions ------------------

def process_blog_feeds_creation(db: Session, blogs: List[Blog], website: str, overwrite: bool = False, use_ai: bool = True):
    """Background task to process blog feed creation - no fallbacks."""
    from database import SessionLocal
    db = SessionLocal()
    try:
        created_count = 0
        skipped_count = 0
        error_count = 0
        error_messages = []
        
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
                
                # Only AI generation is available now - no fallbacks
                feed = create_feed_from_blog(db, blog)
                
                if feed:
                    created_count += 1
                    
            except HTTPException as e:
                error_count += 1
                error_messages.append(f"Blog {blog.id}: {e.detail}")
                logger.error(f"OpenAI error processing blog {blog.id}: {e.detail}")
                continue
            except Exception as e:
                error_count += 1
                error_messages.append(f"Blog {blog.id}: {str(e)}")
                logger.error(f"Error processing blog {blog.id}: {e}")
                continue
        
        logger.info(f"Completed blog feed creation for {website}: {created_count} created, {skipped_count} skipped, {error_count} errors")
        
        # Log errors for debugging
        if error_messages:
            logger.warning(f"Errors encountered for {website}: {error_messages}")
            
    finally:
        db.close()

def process_transcript_feeds_creation(db: Session, transcripts: List[Transcript], job_id: str, overwrite: bool = False, use_ai: bool = True):
    """Background task to process transcript feed creation - no fallbacks."""
    from database import SessionLocal
    db = SessionLocal()
    try:
        created_count = 0
        skipped_count = 0
        error_count = 0
        error_messages = []
        
        for transcript in transcripts:
            try:
                existing_feed = db.query(Feed).filter(Feed.transcript_id == transcript.transcript_id).first()
                if existing_feed and not overwrite:
                    skipped_count += 1
                    continue
                
                # Only AI generation is available now - no fallbacks
                feed = create_feed_from_transcript(db, transcript, overwrite)
                
                if feed:
                    created_count += 1
                    
            except HTTPException as e:
                error_count += 1
                error_messages.append(f"Transcript {transcript.transcript_id}: {e.detail}")
                logger.error(f"OpenAI error processing transcript {transcript.transcript_id}: {e.detail}")
                continue
            except Exception as e:
                error_count += 1
                error_messages.append(f"Transcript {transcript.transcript_id}: {str(e)}")
                logger.error(f"Error processing transcript {transcript.transcript_id}: {e}")
                continue
        
        logger.info(f"Completed transcript feed creation for job {job_id}: {created_count} created, {skipped_count} skipped, {error_count} errors")
        
        # Log errors for debugging
        if error_messages:
            logger.warning(f"Errors encountered for job {job_id}: {error_messages}")
            
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
    """Create feeds for all blogs from a website - AI only, no fallbacks."""
    blogs = db.query(Blog).filter(Blog.website == request.website).all()
    if not blogs:
        raise HTTPException(status_code=404, detail="No blogs found for this website")

    background_tasks.add_task(
        process_blog_feeds_creation,
        db,
        blogs,
        request.website,
        request.overwrite,
        True,  # Always use AI now - no fallbacks
    )

    return {
        "website": request.website,
        "total_blogs": len(blogs),
        "use_ai": True,  # Always true now
        "generate_images": False,
        "source_type": "blog",
        "message": "Blog feed creation process started in background (AI-only, no fallbacks)",
        "status": "processing",
        # "warning": "If OpenAI API fails, feed creation will fail with clear error messages"
    }

@router.post("/feeds/youtube", response_model=dict)
def create_feeds_from_youtube(
    request: YouTubeFeedRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Create feeds from YouTube transcripts - AI only, no fallbacks."""
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

    background_tasks.add_task(
        process_transcript_feeds_creation,
        db,
        transcripts,
        job_identifier,
        request.overwrite,
        True,  # Always use AI now - no fallbacks
    )

    return {
        "job_id": request.job_id,
        "video_id": request.video_id,
        "total_transcripts": len(transcripts),
        "use_ai": True,  # Always true now
        "generate_images": False,
        "source_type": "youtube",
        "message": f"YouTube transcript feed creation process started for {len(transcripts)} transcripts (AI-only, no fallbacks)",
        "status": "processing",
        # "warning": "If OpenAI API fails, feed creation will fail with clear error messages"
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