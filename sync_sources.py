import logging
from sqlalchemy.orm import Session
from database import get_db
import models
from publish_router import get_feed_metadata

# Configure logging
logger = logging.getLogger(__name__)

def sync_sources():
    """
    Synchronizes sources from feeds to the database.
    Checks all feeds and creates corresponding Source entries if they don't exist.
    """
    # Get a database session
    db = next(get_db())
    
    try:
        logger.info("Starting source synchronization...")
        feeds = db.query(models.Feed).all()
        created_count = 0
        
        for feed in feeds:
            try:
                # Logic to get metadata using the shared function
                meta = get_feed_metadata(db, feed, feed.blog)
                
                # Determine Name, Website, Type
                source_type = meta.get('source_type', 'blog')
                
                follower_count = 0
                
                if source_type == 'youtube':
                    name = meta.get('channel_name') or meta.get('author') or "YouTube Creator"
                    website = meta.get('source_url') or "https://www.youtube.com"
                    
                    # Extract subscriber count
                    channel_info = meta.get('channel_info') or {}
                    subs = channel_info.get('subscriber_count')
                    if subs:
                        try:
                            follower_count = int(subs)
                        except (ValueError, TypeError):
                            follower_count = 0
                            
                else:
                    name = meta.get('website_name') or meta.get('author')
                    website = meta.get('website') or meta.get('source_url')
                
                # Skip if we couldn't determine a name
                if not name:
                    continue

                # Check existence by name and type
                existing = db.query(models.Source).filter(
                    models.Source.name == name,
                    models.Source.source_type == source_type
                ).first()
                
                if existing:
                    # Update follower count if we have a valid one (YouTube subs)
                    # For blogs we might not have it, so we preserve what's there or 0
                    if follower_count > 0 and existing.follower_count != follower_count:
                        existing.follower_count = follower_count
                        # We can also update other fields if needed, but name/type are keys
                        if website and not existing.website:
                            existing.website = website
                        db.commit()
                        logger.info(f"Updated source {name} follower count to {follower_count}")
                else:
                    new_source = models.Source(
                        name=name,
                        website=website if website else "",
                        source_type=source_type,
                        is_active=True,
                        follower_count=follower_count
                    )
                    db.add(new_source)
                    db.commit()
                    created_count += 1
                    logger.info(f"Created new source: {name} ({source_type}) with {follower_count} followers")
                    
            except Exception as e:
                logger.error(f"Error processing feed {feed.id} for source sync: {e}")
                continue
        
        logger.info(f"Source synchronization completed. Created {created_count} new sources.")
            
    except Exception as e:
        logger.error(f"Error in source synchronization script: {e}")
    finally:
        db.close()
