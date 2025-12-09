import sys
import os
import json
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
from datetime import datetime
from typing import List, Tuple, Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from openai import OpenAI
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database setup
from database import Base, get_db
from models import Blog, Category, SubCategory, Feed, Transcript

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

client = OpenAI(api_key=OPENAI_API_KEY)


class FeedCategoryMigrator:
    """Migrator to fill missing category and subcategory IDs in existing feeds."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.openai_client = client
        self.stats = {
            "total_feeds": 0,
            "processed": 0,
            "updated": 0,
            "skipped": 0,
            "errors": 0,
            "no_content": 0
        }
    
    def get_feed_content(self, feed: Feed) -> Optional[str]:
        """Extract content from feed based on source type."""
        try:
            if feed.source_type == "blog" and feed.blog_id:
                blog = self.db.query(Blog).filter(Blog.id == feed.blog_id).first()
                if blog:
                    return blog.content
            
            elif feed.source_type == "youtube" and feed.transcript_id:
                transcript = self.db.query(Transcript).filter(
                    Transcript.transcript_id == feed.transcript_id
                ).first()
                if transcript:
                    return transcript.transcript_text
            
            return None
        except Exception as e:
            logger.error(f"Error getting content for feed {feed.id}: {e}")
            return None
    
    def get_all_admin_categories(self) -> List[str]:
        """Get all active categories from database."""
        categories = self.db.query(Category).filter(Category.is_active == True).all()
        return [category.name for category in categories]
    
    def categorize_with_openai(self, content: str, admin_categories: List[str]) -> Tuple[List[str], str, str]:
        """
        Categorize content using OpenAI and return:
        - Matched categories list
        - Primary category name (for category_id lookup)
        - Content type
        """
        try:
            truncated_content = content[:4000] + "..." if len(content) > 4000 else content
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": """You are a content analysis assistant. Analyze the content and:
                        1. Categorize it into the provided categories
                        2. Determine content type (Blog, Video, Podcast, Webinar)
                        
                        Return JSON with this structure:
                        {
                            "categories": ["category1", "category2"],
                            "content_type": "Blog/Video/Podcast/Webinar"
                        }"""
                    },
                    {
                        "role": "user",
                        "content": f"Available categories: {', '.join(admin_categories)}.\n\nContent:\n{truncated_content}\n\nReturn JSON with categories and content_type."
                    }
                ],
                temperature=0.1,
                max_tokens=300,
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
                unique_categories = [admin_categories[0]] if admin_categories else ["Uncategorized"]
            
            # Determine content type
            content_type_str = analysis.get("content_type", "Blog")
            
            # Return first category as primary, all categories, and content type
            primary_category = unique_categories[0] if unique_categories else None
            return unique_categories[:3], primary_category, content_type_str
            
        except Exception as e:
            logger.error(f"OpenAI categorization error: {e}")
            return [], None, "Blog"
    
    def find_category_and_subcategory(self, category_name: str) -> Tuple[Optional[int], Optional[str]]:
        """Find category and subcategory IDs based on category name."""
        try:
            # Find category
            category = self.db.query(Category).filter(
                Category.name.ilike(f"%{category_name}%"),
                Category.is_active == True
            ).first()
            
            if not category:
                logger.warning(f"Category not found for name: {category_name}")
                return None, None
            
            category_id = category.id
            
            # Find first subcategory for this category
            subcategory = self.db.query(SubCategory).filter(
                SubCategory.category_id == category_id
            ).first()
            
            subcategory_id = subcategory.id if subcategory else None
            
            return category_id, subcategory_id
            
        except Exception as e:
            logger.error(f"Error finding category/subcategory for '{category_name}': {e}")
            return None, None
    
    def migrate_single_feed(self, feed: Feed) -> bool:
        """Migrate a single feed by filling missing category/subcategory."""
        try:
            # Check if feed already has category_id and subcategory_id
            if feed.category_id is not None and feed.subcategory_id is not None:
                logger.info(f"Feed {feed.id} already has category_id={feed.category_id}, subcategory_id={feed.subcategory_id} - skipping")
                self.stats["skipped"] += 1
                return True
            
            # Get content for categorization
            content = self.get_feed_content(feed)
            if not content:
                logger.warning(f"No content found for feed {feed.id} - skipping")
                self.stats["no_content"] += 1
                return False
            
            # Get admin categories
            admin_categories = self.get_all_admin_categories()
            if not admin_categories:
                logger.error("No admin categories found in database")
                return False
            
            # Categorize using OpenAI
            logger.info(f"Categorizing feed {feed.id} with OpenAI...")
            categories, primary_category, content_type = self.categorize_with_openai(content, admin_categories)
            
            if not primary_category:
                logger.warning(f"No primary category found for feed {feed.id}")
                return False
            
            # Find category and subcategory IDs
            category_id, subcategory_id = self.find_category_and_subcategory(primary_category)
            
            if not category_id:
                logger.warning(f"Could not find category ID for '{primary_category}'")
                return False
            
            # Update feed with new values
            logger.info(f"Updating feed {feed.id}: category_id={category_id}, subcategory_id={subcategory_id}")
            
            # Update the feed
            feed.category_id = category_id
            feed.subcategory_id = subcategory_id
            
            # Also update categories array if needed
            if categories and not feed.categories:
                feed.categories = categories
            
            # Update content_type if needed
            if content_type and not feed.content_type:
                # Convert to appropriate enum value
                content_type_lower = content_type.lower()
                if "video" in content_type_lower:
                    feed.content_type = "Video"
                elif "podcast" in content_type_lower:
                    feed.content_type = "Podcast"
                elif "webinar" in content_type_lower:
                    feed.content_type = "Webinar"
                else:
                    feed.content_type = "Blog"
            
            self.db.commit()
            self.stats["updated"] += 1
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error migrating feed {feed.id}: {e}")
            self.stats["errors"] += 1
            return False
    
    def migrate_all_feeds(self, batch_size: int = 50, limit: Optional[int] = None):
        """Migrate all feeds with missing category/subcategory."""
        try:
            # Query feeds with missing category_id or subcategory_id
            query = self.db.query(Feed).filter(
                (Feed.category_id.is_(None)) | (Feed.subcategory_id.is_(None))
            ).order_by(Feed.id)
            
            if limit:
                query = query.limit(limit)
            
            feeds = query.all()
            self.stats["total_feeds"] = len(feeds)
            
            logger.info(f"Found {self.stats['total_feeds']} feeds with missing category/subcategory")
            
            if not feeds:
                logger.info("No feeds need migration")
                return
            
            # Process in batches
            for i in range(0, len(feeds), batch_size):
                batch = feeds[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(feeds)-1)//batch_size + 1}")
                
                for feed in batch:
                    self.stats["processed"] += 1
                    self.migrate_single_feed(feed)
                    
                    # Log progress
                    if self.stats["processed"] % 10 == 0:
                        logger.info(f"Progress: {self.stats['processed']}/{self.stats['total_feeds']} feeds processed")
            
            # Log final statistics
            logger.info("\n" + "="*50)
            logger.info("MIGRATION COMPLETE - STATISTICS")
            logger.info("="*50)
            logger.info(f"Total feeds with missing data: {self.stats['total_feeds']}")
            logger.info(f"Feeds processed: {self.stats['processed']}")
            logger.info(f"Feeds updated: {self.stats['updated']}")
            logger.info(f"Feeds skipped (already have data): {self.stats['skipped']}")
            logger.info(f"Feeds with no content: {self.stats['no_content']}")
            logger.info(f"Errors: {self.stats['errors']}")
            logger.info("="*50)
            
        except Exception as e:
            logger.error(f"Error in migrate_all_feeds: {e}")
            raise


def main():
    """Main function to run the migration."""
    # Database connection
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL not found in environment variables")
    
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    try:
        # Create a session
        db = SessionLocal()
        
        logger.info("Starting feed category migration...")
        
        # Initialize migrator
        migrator = FeedCategoryMigrator(db)
        
        # Run migration
        migrator.migrate_all_feeds(batch_size=20)
        
        logger.info("Migration completed successfully!")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise
    finally:
        if 'db' in locals():
            db.close()


if __name__ == "__main__":
    main()