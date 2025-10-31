# run_migrations.py
import sqlite3
import os
from sqlalchemy import create_engine, inspect, text
from database import engine

def run_migrations():
    """Run database migrations for new fields"""
    
    print("🔍 Starting database migrations...")
    
    # Get database path from engine
    database_url = str(engine.url)
    if database_url.startswith('sqlite:///'):
        db_path = database_url.replace('sqlite:///', '')
        print(f"📁 Database path: {db_path}")
    else:
        print(f"📁 Database URL: {database_url}")
    
    # Method 1: Using SQLAlchemy inspector
    try:
        inspector = inspect(engine)
        
        # Check blogs table
        if 'blogs' in inspector.get_table_names():
            existing_columns = [col['name'] for col in inspector.get_columns('blogs')]
            print(f"📋 Blogs table columns: {existing_columns}")
            
            if 'generate_feed' not in existing_columns:
                print("➕ Adding generate_feed to blogs table...")
                with engine.connect() as conn:
                    conn.execute(text("ALTER TABLE blogs ADD COLUMN generate_feed BOOLEAN DEFAULT 0"))
                    print("✅ Added generate_feed to blogs")
            
            if 'feed_generated' not in existing_columns:
                print("➕ Adding feed_generated to blogs table...")
                with engine.connect() as conn:
                    conn.execute(text("ALTER TABLE blogs ADD COLUMN feed_generated BOOLEAN DEFAULT 0"))
                    print("✅ Added feed_generated to blogs")
        
        # Check transcripts table
        if 'transcripts' in inspector.get_table_names():
            existing_columns = [col['name'] for col in inspector.get_columns('transcripts')]
            print(f"📋 Transcripts table columns: {existing_columns}")
            
            if 'generate_feed' not in existing_columns:
                print("➕ Adding generate_feed to transcripts table...")
                with engine.connect() as conn:
                    conn.execute(text("ALTER TABLE transcripts ADD COLUMN generate_feed BOOLEAN DEFAULT 0"))
                    print("✅ Added generate_feed to transcripts")
            
            if 'feed_generated' not in existing_columns:
                print("➕ Adding feed_generated to transcripts table...")
                with engine.connect() as conn:
                    conn.execute(text("ALTER TABLE transcripts ADD COLUMN feed_generated BOOLEAN DEFAULT 0"))
                    print("✅ Added feed_generated to transcripts")
                    
    except Exception as e:
        print(f"❌ SQLAlchemy migration failed: {e}")
        print("🔄 Trying direct SQLite connection...")
        run_direct_sqlite_migration(db_path)

def run_direct_sqlite_migration(db_path):
    """Run migrations using direct SQLite connection"""
    if not os.path.exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check and migrate blogs table
        cursor.execute("PRAGMA table_info(blogs)")
        blog_columns = [column[1] for column in cursor.fetchall()]
        print(f"📋 Blogs columns: {blog_columns}")
        
        if 'generate_feed' not in blog_columns:
            print("➕ Adding generate_feed to blogs...")
            cursor.execute("ALTER TABLE blogs ADD COLUMN generate_feed BOOLEAN DEFAULT 0")
        
        if 'feed_generated' not in blog_columns:
            print("➕ Adding feed_generated to blogs...")
            cursor.execute("ALTER TABLE blogs ADD COLUMN feed_generated BOOLEAN DEFAULT 0")
        
        # Check and migrate transcripts table
        cursor.execute("PRAGMA table_info(transcripts)")
        transcript_columns = [column[1] for column in cursor.fetchall()]
        print(f"📋 Transcripts columns: {transcript_columns}")
        
        if 'generate_feed' not in transcript_columns:
            print("➕ Adding generate_feed to transcripts...")
            cursor.execute("ALTER TABLE transcripts ADD COLUMN generate_feed BOOLEAN DEFAULT 0")
        
        if 'feed_generated' not in transcript_columns:
            print("➕ Adding feed_generated to transcripts...")
            cursor.execute("ALTER TABLE transcripts ADD COLUMN feed_generated BOOLEAN DEFAULT 0")
        
        conn.commit()
        print("✅ Direct SQLite migrations completed!")
        
    except sqlite3.Error as e:
        print(f"❌ SQLite error: {e}")
        conn.rollback()
    finally:
        conn.close()

def verify_migrations():
    """Verify that migrations were applied correctly"""
    inspector = inspect(engine)
    
    print("\n🔍 Verifying migrations...")
    
    required_tables = ['blogs', 'transcripts']
    all_good = True
    
    for table in required_tables:
        if table in inspector.get_table_names():
            existing_columns = [col['name'] for col in inspector.get_columns(table)]
            print(f"\n📊 {table} table:")
            
            for column in ['generate_feed', 'feed_generated']:
                if column in existing_columns:
                    print(f"   ✅ {column} - EXISTS")
                else:
                    print(f"   ❌ {column} - MISSING")
                    all_good = False
        else:
            print(f"❌ Table {table} doesn't exist")
            all_good = False
    
    if all_good:
        print("\n🎉 All migrations verified successfully!")
    else:
        print("\n💥 Some migrations failed!")
    
    return all_good

if __name__ == "__main__":
    run_migrations()
    verify_migrations()