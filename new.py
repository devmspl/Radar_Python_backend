#!/usr/bin/env python3
"""
Script to add AI-generated tags to existing feeds.
This will automatically analyze feed content and generate appropriate tags for filtering.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import openai
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import time
import logging

# Add parent directory to path to import models
sys.path.append(str(Path(__file__).parent.parent))

from models import Feed, Blog, Transcript, Slide, FilterType
from config import settings
from database import SessionLocal,engine
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tag_migration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load OpenAI API key from .env
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("❌ OPENAI_API_KEY not found in .env file")
    sys.exit(1)

openai.api_key = OPENAI_API_KEY

# Database setup
Database_URL = settings.DATABASE_URL
class FeedTagGenerator:
    """Generate tags for feeds using OpenAI API."""
    
    def __init__(self):
        self.api_key = OPENAI_API_KEY
        self.model = "gpt-3.5-turbo"  # You can use "gpt-4" for better results
        self.default_content_type = FilterType.BLOG
        
        # Template tags for fallback
        self.default_tags = {
            "WEBINAR": {
                "skills": ["Presentation", "Communication", "Public Speaking", "Teaching", "Knowledge Sharing"],
                "tools": ["Zoom", "Google Meet", "Microsoft Teams", "Slides", "Presentation Software"],
                "roles": ["Speaker", "Presenter", "Educator", "Trainer", "Workshop Facilitator"]
            },
            "BLOG": {
                "skills": ["Writing", "Research", "Content Creation", "SEO", "Editing"],
                "tools": ["WordPress", "Medium", "Google Docs", "Notion", "Markdown Editors"],
                "roles": ["Writer", "Blogger", "Content Creator", "Editor", "Journalist"]
            },
            "PODCAST": {
                "skills": ["Audio Editing", "Interviewing", "Storytelling", "Voice Modulation", "Script Writing"],
                "tools": ["Audacity", "GarageBand", "Podcast Hosting", "Microphones", "Audio Interfaces"],
                "roles": ["Podcaster", "Host", "Producer", "Audio Engineer", "Interviewer"]
            },
            "VIDEO": {
                "skills": ["Video Editing", "Script Writing", "Camerawork", "Lighting", "Directing"],
                "tools": ["Premiere Pro", "Final Cut Pro", "YouTube Studio", "Cameras", "Editing Software"],
                "roles": ["Video Creator", "Editor", "Director", "Content Creator", "YouTuber"]
            }
        }
    
    def analyze_feed_content(self, feed: Feed, db: Session) -> Dict[str, Any]:
        """Extract content from feed for AI analysis."""
        content_parts = []
        
        # Add feed title
        if feed.title:
            content_parts.append(f"Title: {feed.title}")
        
        # Add categories
        if feed.categories:
            content_parts.append(f"Categories: {', '.join(feed.categories[:5])}")
        
        # Get content from slides
        if feed.slides:
            sorted_slides = sorted(feed.slides, key=lambda x: x.order)
            slide_contents = []
            for slide in sorted_slides[:5]:  # First 5 slides
                slide_text = f"{slide.title}: {slide.body[:200]}"
                slide_contents.append(slide_text)
            content_parts.append(f"Slides Content: {' '.join(slide_contents)}")
        
        # Get content from source
        if feed.source_type == "blog" and feed.blog:
            blog = feed.blog
            if blog.content:
                content_parts.append(f"Blog Content: {blog.content[:500]}")
            if blog.title:
                content_parts.append(f"Blog Title: {blog.title}")
            if blog.description:
                content_parts.append(f"Blog Description: {blog.description[:300]}")
        
        elif feed.source_type == "youtube" and feed.transcript_id:
            transcript = db.query(Transcript).filter(
                Transcript.transcript_id == feed.transcript_id
            ).first()
            if transcript and transcript.transcript_text:
                content_parts.append(f"Transcript: {transcript.transcript_text[:500]}")
            if transcript and transcript.title:
                content_parts.append(f"Video Title: {transcript.title}")
        
        # Combine all content
        combined_content = "\n".join(content_parts)
        return {"content": combined_content, "title": feed.title or "Untitled Feed"}
    
    def generate_tags_with_ai(self, feed_content: Dict[str, Any]) -> Dict[str, Any]:
        """Use OpenAI to generate tags for the feed content."""
        content_preview = feed_content["content"][:1500]  # Limit content for token efficiency
        title = feed_content["title"]
        
        prompt = f"""
        Analyze this educational/professional content and suggest appropriate tags:
        
        Title: {title}
        
        Content Preview:
        {content_preview}
        
        Based on this content, please provide:
        1. Primary content type (choose ONE): ["Webinar", "Blog", "Podcast", "Video"]
        2. Relevant skills (3-5 specific skills mentioned or required)
        3. Relevant tools/software (3-5 tools mentioned or used)
        4. Relevant job roles (3-5 roles that would benefit from this content)
        
        Return ONLY a JSON object in this exact format:
        {{
            "content_type": "string (Webinar, Blog, Podcast, or Video)",
            "skills": ["skill1", "skill2", "skill3"],
            "tools": ["tool1", "tool2", "tool3"],
            "roles": ["role1", "role2", "role3"]
        }}
        
        Make the tags professional, specific, and relevant to the content.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional content analyst specializing in educational and professional development materials."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != 0:
                json_str = response_text[json_start:json_end]
                tags = json.loads(json_str)
            else:
                tags = json.loads(response_text)
            
            # Validate and clean tags
            tags = self._validate_and_clean_tags(tags)
            
            logger.info(f"✅ Generated tags: {tags['content_type']} with {len(tags['skills'])} skills")
            return tags
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse AI response: {e}. Using fallback tags.")
            return self._get_fallback_tags(title, content_preview)
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._get_fallback_tags(title, content_preview)
    
    def _validate_and_clean_tags(self, tags: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean AI-generated tags."""
        # Validate content type
        valid_content_types = ["Webinar", "Blog", "Podcast", "Video"]
        if tags.get("content_type") not in valid_content_types:
            tags["content_type"] = "Blog"
        
        # Clean lists
        for key in ["skills", "tools", "roles"]:
            if key in tags:
                items = tags[key]
                if isinstance(items, list):
                    # Remove empty strings, capitalize, deduplicate
                    cleaned = []
                    for item in items:
                        if isinstance(item, str) and item.strip():
                            # Capitalize each word
                            cleaned_item = ' '.join(word.capitalize() for word in item.strip().split())
                            cleaned.append(cleaned_item)
                    # Remove duplicates and limit to 5
                    tags[key] = list(dict.fromkeys(cleaned))[:5]
                else:
                    tags[key] = []
            else:
                tags[key] = []
        
        return tags
    
    def _get_fallback_tags(self, title: str, content_preview: str) -> Dict[str, Any]:
        """Get fallback tags based on content analysis."""
        content_lower = (title + " " + content_preview).lower()
        
        # Determine content type based on keywords
        if any(word in content_lower for word in ["webinar", "workshop", "seminar", "training", "lecture"]):
            content_type = "Webinar"
        elif any(word in content_lower for word in ["podcast", "audio", "episode", "interview", "show"]):
            content_type = "Podcast"
        elif any(word in content_lower for word in ["video", "youtube", "tutorial", "screen cast", "recording"]):
            content_type = "Video"
        else:
            content_type = "Blog"
        
        # Get appropriate default tags
        tags_key = content_type.upper()
        if tags_key in self.default_tags:
            return {
                "content_type": content_type,
                "skills": self.default_tags[tags_key]["skills"],
                "tools": self.default_tags[tags_key]["tools"],
                "roles": self.default_tags[tags_key]["roles"]
            }
        else:
            return {
                "content_type": content_type,
                "skills": ["Learning", "Research", "Professional Development"],
                "tools": ["Web Browser", "Note-taking App", "PDF Reader"],
                "roles": ["Learner", "Professional", "Student"]
            }
    
    def map_content_type_to_enum(self, content_type_str: str) -> FilterType:
        """Map string content type to FilterType enum."""
        content_type_str = content_type_str.lower()
        if content_type_str == "webinar":
            return FilterType.WEBINAR
        elif content_type_str == "blog":
            return FilterType.BLOG
        elif content_type_str == "podcast":
            return FilterType.PODCAST
        elif content_type_str == "video":
            return FilterType.VIDEO
        else:
            return FilterType.BLOG
    
    def update_feed_with_tags(self, feed: Feed, tags: Dict[str, Any], db: Session) -> bool:
        """Update feed with generated tags."""
        try:
            # Map content type to enum
            content_type_enum = self.map_content_type_to_enum(tags["content_type"])
            feed.content_type = content_type_enum
            
            # Update skills, tools, roles (only if they don't exist or are minimal)
            if not feed.skills or len(feed.skills) < 3:
                feed.skills = tags["skills"]
            else:
                # Merge existing with new tags
                existing = set(feed.skills)
                new = set(tags["skills"])
                feed.skills = list(existing.union(new))
            
            if not feed.tools or len(feed.tools) < 3:
                feed.tools = tags["tools"]
            else:
                existing = set(feed.tools)
                new = set(tags["tools"])
                feed.tools = list(existing.union(new))
            
            if not feed.roles or len(feed.roles) < 3:
                feed.roles = tags["roles"]
            else:
                existing = set(feed.roles)
                new = set(tags["roles"])
                feed.roles = list(existing.union(new))
            
            feed.updated_at = datetime.utcnow()
            return True
            
        except Exception as e:
            logger.error(f"Error updating feed {feed.id}: {e}")
            return False

def get_feeds_needing_tags(db: Session, limit: int = None, specific_ids: List[int] = None) -> List[Feed]:
    """Get feeds that need tag generation."""
    query = db.query(Feed).options(
        # joinedload(Feed.slides),
        # joinedload(Feed.blog)
    )
    
    if specific_ids:
        query = query.filter(Feed.id.in_(specific_ids))
    else:
        # Get feeds with minimal or no tags
        # You can adjust this condition based on your needs
        query = query.filter(
            (Feed.skills == None) | 
            (Feed.tools == None) | 
            (Feed.roles == None) |
            (Feed.content_type == FilterType.BLOG)  # Default value
        )
    
    if limit:
        query = query.limit(limit)
    
    return query.all()

def main():
    """Main function to run the tag migration."""
    print("=" * 60)
    print("FEED TAG MIGRATION SCRIPT")
    print("=" * 60)
    
    # Configuration
    BATCH_SIZE = 10  # Process feeds in batches to avoid rate limits
    DELAY_BETWEEN_BATCHES = 2  # Seconds to wait between batches
    LIMIT = None  # Set to None for all feeds, or specify a number
    SPECIFIC_FEED_IDS = None  # Set to [1, 2, 3] to process specific feeds only
    
    db = SessionLocal()
    tag_generator = FeedTagGenerator()
    
    try:
        # Get feeds that need tagging
        feeds = get_feeds_needing_tags(db, limit=LIMIT, specific_ids=SPECIFIC_FEED_IDS)
        
        if not feeds:
            print("✅ No feeds need tagging. All feeds already have tags!")
            return
        
        print(f"📊 Found {len(feeds)} feeds that need tagging")
        print("=" * 60)
        
        # Process in batches
        total_updated = 0
        total_failed = 0
        
        for i in range(0, len(feeds), BATCH_SIZE):
            batch = feeds[i:i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            total_batches = (len(feeds) + BATCH_SIZE - 1) // BATCH_SIZE
            
            print(f"\n🔄 Processing Batch {batch_num}/{total_batches} ({len(batch)} feeds)")
            print("-" * 40)
            
            batch_updated = 0
            batch_failed = 0
            
            for feed in batch:
                try:
                    print(f"\n📝 Feed #{feed.id}: {feed.title[:50]}...")
                    
                    # Skip if feed has comprehensive tags already
                    current_tags = {
                        "skills": len(feed.skills or []),
                        "tools": len(feed.tools or []),
                        "roles": len(feed.roles or [])
                    }
                    
                    if all(count >= 3 for count in current_tags.values()) and feed.content_type != FilterType.BLOG:
                        print(f"   ⏭️ Already has tags: {current_tags}")
                        continue
                    
                    # Analyze feed content
                    feed_content = tag_generator.analyze_feed_content(feed, db)
                    
                    # Generate tags with AI
                    print(f"   🤖 Generating tags with AI...")
                    tags = tag_generator.generate_tags_with_ai(feed_content)
                    
                    # Update feed
                    print(f"   📝 Applying tags...")
                    success = tag_generator.update_feed_with_tags(feed, tags, db)
                    
                    if success:
                        db.commit()
                        batch_updated += 1
                        print(f"   ✅ Updated: {tags['content_type']}")
                        print(f"      Skills: {', '.join(tags['skills'][:3])}")
                        print(f"      Tools: {', '.join(tags['tools'][:3])}")
                        print(f"      Roles: {', '.join(tags['roles'][:3])}")
                    else:
                        batch_failed += 1
                        print(f"   ❌ Failed to update")
                        
                except Exception as e:
                    batch_failed += 1
                    print(f"   ❌ Error processing feed {feed.id}: {str(e)[:100]}")
                    db.rollback()
                    continue
            
            total_updated += batch_updated
            total_failed += batch_failed
            
            print(f"\n📊 Batch {batch_num} Summary:")
            print(f"   ✅ Updated: {batch_updated}")
            print(f"   ❌ Failed: {batch_failed}")
            
            # Wait between batches to avoid rate limits
            if i + BATCH_SIZE < len(feeds):
                print(f"\n⏳ Waiting {DELAY_BETWEEN_BATCHES} seconds before next batch...")
                time.sleep(DELAY_BETWEEN_BATCHES)
        
        # Final statistics
        print("\n" + "=" * 60)
        print("📊 MIGRATION COMPLETE")
        print("=" * 60)
        print(f"Total feeds processed: {len(feeds)}")
        print(f"✅ Successfully updated: {total_updated}")
        print(f"❌ Failed: {total_failed}")
        
        # Show content type distribution
        print("\n📈 Content Type Distribution:")
        content_types = db.query(Feed.content_type, db.func.count(Feed.id)).group_by(Feed.content_type).all()
        for content_type, count in content_types:
            print(f"   {content_type.value if content_type else 'None'}: {count}")
        
        # Show sample of updated feeds
        print("\n🎯 Sample of Updated Feeds:")
        sample_feeds = db.query(Feed).filter(
            Feed.skills != None,
            Feed.tools != None
        ).order_by(Feed.updated_at.desc()).limit(5).all()
        
        for i, feed in enumerate(sample_feeds, 1):
            print(f"{i}. Feed #{feed.id}: {feed.title[:50]}...")
            print(f"   Type: {feed.content_type.value if feed.content_type else 'None'}")
            print(f"   Skills: {', '.join(feed.skills[:3] if feed.skills else [])}")
            print(f"   Tools: {', '.join(feed.tools[:3] if feed.tools else [])}")
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        db.close()
        print("\n🔚 Database connection closed.")

def verify_tags():
    """Verify that tags have been added correctly."""
    print("\n" + "=" * 60)
    print("🔍 VERIFYING TAGS")
    print("=" * 60)
    
    db = SessionLocal()
    
    try:
        # Count feeds with tags
        total_feeds = db.query(Feed).count()
        feeds_with_skills = db.query(Feed).filter(Feed.skills != None, Feed.skills != []).count()
        feeds_with_tools = db.query(Feed).filter(Feed.tools != None, Feed.tools != []).count()
        feeds_with_roles = db.query(Feed).filter(Feed.roles != None, Feed.roles != []).count()
        feeds_with_content_type = db.query(Feed).filter(Feed.content_type != None, Feed.content_type != FilterType.BLOG).count()
        
        print(f"Total feeds in database: {total_feeds}")
        print(f"Feeds with skills tags: {feeds_with_skills} ({feeds_with_skills/total_feeds*100:.1f}%)")
        print(f"Feeds with tools tags: {feeds_with_tools} ({feeds_with_tools/total_feeds*100:.1f}%)")
        print(f"Feeds with roles tags: {feeds_with_roles} ({feeds_with_roles/total_feeds*100:.1f}%)")
        print(f"Feeds with specific content type: {feeds_with_content_type} ({feeds_with_content_type/total_feeds*100:.1f}%)")
        
        # Show feeds still missing tags
        feeds_missing_tags = db.query(Feed).filter(
            (Feed.skills == None) | (Feed.skills == []) |
            (Feed.tools == None) | (Feed.tools == []) |
            (Feed.roles == None) | (Feed.roles == [])
        ).count()
        
        print(f"\nFeeds still missing some tags: {feeds_missing_tags}")
        
        if feeds_missing_tags > 0:
            print("\nSample of feeds missing tags:")
            missing_feeds = db.query(Feed).filter(
                (Feed.skills == None) | (Feed.skills == [])
            ).limit(5).all()
            
            for i, feed in enumerate(missing_feeds, 1):
                print(f"{i}. Feed #{feed.id}: {feed.title[:50]}...")
        
    finally:
        db.close()

if __name__ == "__main__":
    print("🚀 Starting Feed Tag Migration")
    print("Make sure your .env file has OPENAI_API_KEY and database is accessible")
    print("-" * 60)
    
    # Run the main migration
    main()
    
    # Verify the results
    verify_tags()
    
    print("\n✅ Script completed!")
    print("\n💡 Next steps:")
    print("1. Test your filter endpoints with the new tags")
    print("2. Use: POST /publish/feeds/advanced-filter-with-tags")
    print("3. Try filtering by content_type: 'Podcast', 'Webinar', etc.")
    print("4. Try filtering by skills/tools/roles")