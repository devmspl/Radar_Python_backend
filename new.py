# update_all_feeds.py
"""
Complete script to update ALL existing feeds with enhanced metadata for search functionality.
This script will process all feeds and extract concepts, domains, and sources.
"""

import sys
import logging
import time
from datetime import datetime
from typing import List, Dict, Any
import json

# Add your project to path
sys.path.append('.')

from sqlalchemy import create_engine, func, text
from sqlalchemy.orm import sessionmaker
from models import (
    Feed, Blog, Transcript, Category, SubCategory,
    Domain, Concept, FeedConcept, DomainConcept, Source, ContentList
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feed_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AllFeedUpdater:
    def __init__(self, database_url: str = None):
        """Initialize the feed updater."""
        try:
            if database_url:
                self.engine = create_engine(database_url)
            else:
                # Try to import from config
                try:
                    from config import settings
                    self.engine = create_engine(settings.DATABASE_URL)
                except ImportError:
                    # Default database URL
                    self.engine = create_engine('sqlite:///./test.db')
            
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            self.db = SessionLocal()
            
            logger.info("[OK] Database connection established")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize: {str(e)}")
            raise
    
    def count_all_feeds(self) -> Dict[str, int]:
        """Count all feeds by status and source type."""
        try:
            total = self.db.query(Feed).count()
            ready = self.db.query(Feed).filter(Feed.status == "ready").count()
            youtube = self.db.query(Feed).filter(Feed.source_type == "youtube").count()
            blog = self.db.query(Feed).filter(Feed.source_type == "blog").count()
            
            return {
                "total": total,
                "ready": ready,
                "youtube": youtube,
                "blog": blog
            }
        except Exception as e:
            logger.error(f"Error counting feeds: {str(e)}")
            return {}
    
    def check_dependencies(self) -> bool:
        """Check if required functions are available."""
        try:
            # Try to import required functions
            from feed_router import (
                categorize_content_with_openai,
                get_or_create_concepts,
                get_or_create_domains,
                get_or_assign_category,
                extract_clean_source_name,
                get_youtube_channel_info
            )
            
            self.categorize_content_with_openai = categorize_content_with_openai
            self.get_or_create_concepts = get_or_create_concepts
            self.get_or_create_domains = get_or_create_domains
            self.get_or_assign_category = get_or_assign_category
            self.extract_clean_source_name = extract_clean_source_name
            self.get_youtube_channel_info = get_youtube_channel_info
            
            # Check OpenAI client
            from feed_router import client
            if not client:
                logger.warning("OpenAI client not configured - AI features will be limited")
                return False
            
            logger.info("[OK] All dependencies loaded successfully")
            return True
            
        except ImportError as e:
            logger.error(f"[ERROR] Missing dependencies: {str(e)}")
            logger.info("Creating fallback functions...")
            self._create_fallback_functions()
            return False
    
    def _create_fallback_functions(self):
        """Create fallback functions if OpenAI is not available."""
        def fallback_categorize(content, categories):
            """Fallback categorization without OpenAI."""
            import random
            
            # Simple keyword-based categorization
            keywords = {
                "technology": ["tech", "software", "code", "programming", "ai", "machine learning"],
                "business": ["business", "marketing", "sales", "revenue", "strategy"],
                "education": ["learn", "education", "tutorial", "course", "training"],
                "health": ["health", "medical", "fitness", "wellness", "medicine"]
            }
            
            content_lower = content.lower()
            matched_categories = []
            
            for category, keyword_list in keywords.items():
                for keyword in keyword_list:
                    if keyword in content_lower:
                        matched_categories.append(category)
                        break
            
            if not matched_categories:
                matched_categories = ["General"]
            
            return (
                matched_categories[:3],  # categories
                [],  # skills
                [],  # tools
                [],  # roles
                "Blog",  # content_type
                ["General Topic"],  # concepts
                ["General"]  # domains
            )
        
        def fallback_create_concepts(db, concepts):
            """Fallback concept creation."""
            concept_objects = []
            for name in concepts:
                concept = db.query(Concept).filter(func.lower(Concept.name) == func.lower(name)).first()
                if not concept:
                    concept = Concept(name=name, description=f"Concept: {name}", is_active=True)
                    db.add(concept)
                    db.flush()
                concept_objects.append(concept)
            return concept_objects
        
        def fallback_create_domains(db, domains):
            """Fallback domain creation."""
            domain_objects = []
            for name in domains:
                domain = db.query(Domain).filter(func.lower(Domain.name) == func.lower(name)).first()
                if not domain:
                    domain = Domain(name=name, description=f"Domain: {name}", is_active=True)
                    db.add(domain)
                    db.flush()
                domain_objects.append(domain)
            return domain_objects
        
        self.categorize_content_with_openai = fallback_categorize
        self.get_or_create_concepts = fallback_create_concepts
        self.get_or_create_domains = fallback_create_domains
        
        # Simple fallbacks for other functions
        self.get_or_assign_category = lambda db, categories: (None, None)
        self.extract_clean_source_name = lambda url: url.split('//')[-1].split('/')[0] if '//' in url else url
        self.get_youtube_channel_info = lambda video_id: {"channel_name": "YouTube Creator", "available": False}
        
        logger.info("[OK] Fallback functions created")
    
    def process_all_feeds(self, batch_size: int = 10, delay: float = 1.0):
        """Process all ready feeds in batches."""
        try:
            # Get all ready feeds
            feeds = self.db.query(Feed).filter(Feed.status == "ready").order_by(Feed.id).all()
            total_feeds = len(feeds)
            
            logger.info(f"Found {total_feeds} ready feeds to process")
            
            stats = {
                "total": total_feeds,
                "processed": 0,
                "success": 0,
                "skipped": 0,
                "failed": 0,
                "concepts_created": 0,
                "domains_created": 0,
                "sources_created": 0
            }
            
            start_time = time.time()
            
            # Process in batches
            for i in range(0, total_feeds, batch_size):
                batch = feeds[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (total_feeds + batch_size - 1) // batch_size
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} feeds)")
                
                batch_stats = self._process_batch(batch, stats)
                stats.update(batch_stats)
                
                # Show progress
                progress = (i + len(batch)) / total_feeds * 100
                elapsed = time.time() - start_time
                estimated_total = elapsed / (progress / 100) if progress > 0 else 0
                remaining = estimated_total - elapsed
                
                logger.info(f"Progress: {progress:.1f}% | "
                          f"Success: {stats['success']} | "
                          f"Failed: {stats['failed']} | "
                          f"ETA: {remaining/60:.1f} min")
                
                # Add delay between batches
                if i + batch_size < total_feeds and delay > 0:
                    time.sleep(delay)
            
            # Final commit
            self.db.commit()
            
            elapsed_time = time.time() - start_time
            logger.info(f"\n{'='*60}")
            logger.info(f"UPDATE COMPLETED")
            logger.info(f"{'='*60}")
            logger.info(f"Total time: {elapsed_time/60:.1f} minutes")
            logger.info(f"Total feeds: {stats['total']}")
            logger.info(f"Successfully processed: {stats['success']}")
            logger.info(f"Skipped (no content): {stats['skipped']}")
            logger.info(f"Failed: {stats['failed']}")
            logger.info(f"Concepts created: {stats['concepts_created']}")
            logger.info(f"Domains created: {stats['domains_created']}")
            logger.info(f"Sources created: {stats['sources_created']}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error in process_all_feeds: {str(e)}")
            self.db.rollback()
            raise
    
    def _process_batch(self, batch: List[Feed], stats: Dict[str, Any]) -> Dict[str, Any]:
        """Process a batch of feeds."""
        batch_stats = stats.copy()
        
        for feed in batch:
            try:
                batch_stats['processed'] += 1
                feed_id = feed.id
                feed_title = feed.title[:50] + "..." if len(feed.title) > 50 else feed.title
                
                logger.info(f"  Processing feed {batch_stats['processed']}/{stats['total']}: "
                          f"ID={feed_id}, Title='{feed_title}'")
                
                # Check if already has concepts
                existing_concepts = self.db.query(FeedConcept).filter(
                    FeedConcept.feed_id == feed.id
                ).count()
                
                if existing_concepts > 0:
                    logger.info(f"    [SKIP] Already has {existing_concepts} concepts, skipping")
                    batch_stats['skipped'] += 1
                    continue
                
                # Process the feed
                result = self._process_single_feed(feed)
                
                if result['success']:
                    batch_stats['success'] += 1
                    batch_stats['concepts_created'] += result.get('concepts_created', 0)
                    batch_stats['domains_created'] += result.get('domains_created', 0)
                    batch_stats['sources_created'] += result.get('sources_created', 0)
                    
                    logger.info(f"    [OK] Success: {result.get('concepts_created', 0)} concepts, "
                              f"{result.get('domains_created', 0)} domains")
                else:
                    batch_stats['failed'] += 1
                    logger.warning(f"    [FAIL] Failed: {result.get('error', 'Unknown error')}")
                
            except Exception as e:
                batch_stats['failed'] += 1
                logger.error(f"    [ERROR] Error processing feed {feed.id}: {str(e)}")
                continue
        
        # Commit the batch
        try:
            self.db.commit()
            logger.info(f"    [SAVE] Batch committed successfully")
        except Exception as e:
            logger.error(f"    [ERROR] Batch commit failed: {str(e)}")
            self.db.rollback()
        
        return batch_stats
    
    def _process_single_feed(self, feed: Feed) -> Dict[str, Any]:
        """Process a single feed."""
        try:
            # Get content
            content = ""
            if feed.source_type == "blog" and feed.blog:
                content = feed.blog.content
            elif feed.source_type == "youtube" and feed.transcript_id:
                transcript = self.db.query(Transcript).filter(
                    Transcript.transcript_id == feed.transcript_id
                ).first()
                if transcript:
                    content = transcript.transcript_text
            
            if not content or len(content.strip()) < 50:
                return {
                    "success": False,
                    "error": "No content or content too short",
                    "feed_id": feed.id
                }
            
            # Get categories for classification
            admin_categories = [c.name for c in self.db.query(Category).filter(
                Category.is_active == True
            ).all()]
            
            if not admin_categories:
                admin_categories = ["Uncategorized"]
            
            # Categorize content
            categories, skills, tools, roles, content_type, concepts, domains = \
                self.categorize_content_with_openai(content, admin_categories)
            
            # Create concepts and domains
            concept_objects = self.get_or_create_concepts(self.db, concepts)
            domain_objects = self.get_or_create_domains(self.db, domains)
            
            # Link concepts to domains
            domain_concept_count = 0
            for concept in concept_objects:
                for domain in domain_objects:
                    existing = self.db.query(DomainConcept).filter(
                        DomainConcept.domain_id == domain.id,
                        DomainConcept.concept_id == concept.id
                    ).first()
                    if not existing:
                        self.db.add(DomainConcept(
                            domain_id=domain.id,
                            concept_id=concept.id,
                            relevance_score=1.0
                        ))
                        domain_concept_count += 1
            
            # Link concepts to feed
            feed_concept_count = 0
            for concept in concept_objects:
                existing = self.db.query(FeedConcept).filter(
                    FeedConcept.feed_id == feed.id,
                    FeedConcept.concept_id == concept.id
                ).first()
                if not existing:
                    self.db.add(FeedConcept(
                        feed_id=feed.id,
                        concept_id=concept.id,
                        confidence_score=0.8
                    ))
                    feed_concept_count += 1
            
            # Create or update source
            source_created = 0
            if feed.source_type == "blog" and feed.blog:
                source_created = self._create_blog_source(feed.blog)
            elif feed.source_type == "youtube" and feed.transcript_id:
                transcript = self.db.query(Transcript).filter(
                    Transcript.transcript_id == feed.transcript_id
                ).first()
                if transcript:
                    source_created = self._create_youtube_source(transcript)
            
            # Update feed timestamp
            feed.updated_at = datetime.utcnow()
            
            return {
                "success": True,
                "feed_id": feed.id,
                "concepts_created": len(concept_objects),
                "domains_created": len(domain_objects),
                "feed_concepts_linked": feed_concept_count,
                "domain_concepts_linked": domain_concept_count,
                "sources_created": source_created
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "feed_id": feed.id
            }
    
    def _create_blog_source(self, blog: Blog) -> int:
        """Create or update blog source."""
        try:
            if not blog.website:
                return 0
            
            source_name = self.extract_clean_source_name(blog.website)
            
            source = self.db.query(Source).filter(
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
                self.db.add(source)
                self.db.flush()
                return 1
            return 0
            
        except Exception as e:
            logger.error(f"Error creating blog source: {str(e)}")
            return 0
    
    def _create_youtube_source(self, transcript: Transcript) -> int:
        """Create or update YouTube source."""
        try:
            channel_info = self.get_youtube_channel_info(transcript.video_id)
            channel_name = channel_info.get("channel_name", "YouTube Creator")
            
            if channel_name == "YouTube Creator":
                return 0
            
            source = self.db.query(Source).filter(
                Source.name == channel_name,
                Source.source_type == "youtube"
            ).first()
            
            if not source:
                website = f"https://www.youtube.com/channel/{channel_info.get('channel_id', '')}" \
                         if channel_info.get('channel_id') else "https://www.youtube.com"
                source = Source(
                    name=channel_name,
                    website=website,
                    source_type="youtube",
                    is_active=True
                )
                self.db.add(source)
                self.db.flush()
                return 1
            return 0
            
        except Exception as e:
            logger.error(f"Error creating YouTube source: {str(e)}")
            return 0
    
    def show_statistics(self):
        """Show current database statistics."""
        try:
            logger.info("\nCURRENT DATABASE STATISTICS")
            logger.info("=" * 50)
            
            # Feed statistics
            feed_counts = self.count_all_feeds()
            logger.info(f"Feeds:")
            logger.info(f"  Total: {feed_counts.get('total', 0)}")
            logger.info(f"  Ready: {feed_counts.get('ready', 0)}")
            logger.info(f"  YouTube: {feed_counts.get('youtube', 0)}")
            logger.info(f"  Blog: {feed_counts.get('blog', 0)}")
            
            # Concept statistics
            concept_count = self.db.query(Concept).count()
            concepts_with_feeds = self.db.query(Concept).join(FeedConcept).distinct().count()
            logger.info(f"\nConcepts:")
            logger.info(f"  Total: {concept_count}")
            logger.info(f"  Linked to feeds: {concepts_with_feeds}")
            
            # Domain statistics
            domain_count = self.db.query(Domain).count()
            logger.info(f"\nDomains:")
            logger.info(f"  Total: {domain_count}")
            
            # Source statistics
            source_count = self.db.query(Source).count()
            logger.info(f"\nSources:")
            logger.info(f"  Total: {source_count}")
            
            # Feed-Concept relationships
            feed_concept_count = self.db.query(FeedConcept).count()
            feeds_with_concepts = self.db.query(Feed).join(FeedConcept).distinct().count()
            logger.info(f"\nRelationships:")
            logger.info(f"  Feed-Concept links: {feed_concept_count}")
            logger.info(f"  Feeds with concepts: {feeds_with_concepts}")
            
            # Top concepts
            top_concepts = self.db.query(
                Concept.name,
                func.count(FeedConcept.id).label('count')
            ).join(FeedConcept).group_by(Concept.id).order_by(
                func.count(FeedConcept.id).desc()
            ).limit(5).all()
            
            if top_concepts:
                logger.info(f"\nTop 5 Concepts:")
                for i, (name, count) in enumerate(top_concepts, 1):
                    logger.info(f"  {i}. {name}: {count} feeds")
            
        except Exception as e:
            logger.error(f"Error showing statistics: {str(e)}")
    
    def cleanup(self):
        """Clean up orphaned data."""
        try:
            logger.info("\nCleaning up orphaned data...")
            
            # Find and delete orphaned concepts
            orphaned_concepts = self.db.query(Concept).filter(
                ~Concept.id.in_(self.db.query(FeedConcept.concept_id).distinct())
            ).all()
            
            # Find and delete orphaned domains
            orphaned_domains = self.db.query(Domain).filter(
                ~Domain.id.in_(self.db.query(DomainConcept.domain_id).distinct())
            ).all()
            
            concept_count = len(orphaned_concepts)
            domain_count = len(orphaned_domains)
            
            for concept in orphaned_concepts:
                self.db.delete(concept)
            
            for domain in orphaned_domains:
                self.db.delete(domain)
            
            self.db.commit()
            
            logger.info(f"  Deleted {concept_count} orphaned concepts")
            logger.info(f"  Deleted {domain_count} orphaned domains")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            self.db.rollback()
    
    def close(self):
        """Close database connection."""
        try:
            self.db.close()
            logger.info("Database connection closed")
        except:
            pass

def main():
    """Main function to run the updater."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Update ALL feeds with enhanced metadata for search")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of feeds per batch")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between batches in seconds")
    parser.add_argument("--stats", action="store_true", help="Show statistics only")
    parser.add_argument("--cleanup", action="store_true", help="Clean up orphaned data only")
    parser.add_argument("--test", action="store_true", help="Test mode - process only first 5 feeds")
    
    args = parser.parse_args()
    
    updater = None
    try:
        print("\n" + "="*60)
        print("FEED UPDATER - Enhancing all feeds for search")
        print("="*60 + "\n")
        
        updater = AllFeedUpdater()
        
        # Check dependencies
        if not updater.check_dependencies():
            print("[WARNING] OpenAI not configured. Using fallback functions.")
            print("          Search quality will be limited.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Exiting...")
                return
        
        # Show current statistics
        updater.show_statistics()
        
        if args.stats:
            # Just show statistics and exit
            updater.close()
            return
        
        if args.cleanup:
            # Just cleanup and exit
            updater.cleanup()
            updater.close()
            return
        
        # Confirm before processing
        feed_counts = updater.count_all_feeds()
        ready_feeds = feed_counts.get('ready', 0)
        
        if ready_feeds == 0:
            print("[ERROR] No ready feeds found to process!")
            updater.close()
            return
        
        if args.test:
            print(f"\n[TEST MODE] Will process only first 5 feeds")
            ready_feeds = 5
        else:
            print(f"\nREADY TO PROCESS {ready_feeds} FEEDS")
        
        print(f"Batch size: {args.batch_size}")
        print(f"Delay between batches: {args.delay} seconds")
        
        if not args.test:
            response = input("\nContinue with update? (y/n): ")
            if response.lower() != 'y':
                print("Update cancelled.")
                updater.close()
                return
        
        # Process feeds
        print("\n" + "="*60)
        print("STARTING FEED UPDATE PROCESS")
        print("="*60 + "\n")
        
        if args.test:
            # Test mode - process limited feeds
            feeds = updater.db.query(Feed).filter(Feed.status == "ready").order_by(Feed.id).limit(5).all()
            batch_stats = updater._process_batch(feeds, {"total": 5, "processed": 0, "success": 0, "skipped": 0, "failed": 0})
            updater.db.commit()
            print(f"\n[OK] TEST COMPLETED")
            print(f"Processed: {batch_stats.get('processed', 0)}")
            print(f"Success: {batch_stats.get('success', 0)}")
            print(f"Failed: {batch_stats.get('failed', 0)}")
        else:
            # Full processing
            stats = updater.process_all_feeds(
                batch_size=args.batch_size,
                delay=args.delay
            )
            
            # Show final statistics
            print("\n" + "="*60)
            print("FINAL STATISTICS")
            print("="*60)
            updater.show_statistics()
            
            # Cleanup orphaned data
            print("\n" + "="*60)
            print("CLEANING UP ORPHANED DATA")
            print("="*60)
            updater.cleanup()
        
        print("\n" + "="*60)
        print("[OK] UPDATE PROCESS COMPLETED SUCCESSFULLY")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n[WARNING] Update interrupted by user")
        if updater:
            updater.db.rollback()
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if updater:
            updater.close()

if __name__ == "__main__":
    main()