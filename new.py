"""
SQLite-specific script to update database models with category-subcategory relationships.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text, inspect, Column, String, ForeignKey
from sqlalchemy.orm import sessionmaker
import uuid

# Database configuration - update with your actual SQLite database URL
DATABASE_URL = "sqlite:///./test.db"  # Update this to match your actual SQLite file

# Initialize database connection
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def check_table_exists(table_name):
    """Check if a table exists in SQLite database."""
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()

def get_column_info(table_name):
    """Get information about columns in a SQLite table."""
    inspector = inspect(engine)
    columns = inspector.get_columns(table_name)
    return [col['name'] for col in columns]

def generate_uuid():
    """Generate a UUID string."""
    return str(uuid.uuid4())

def update_database_sqlite():
    """Update SQLite database with new category-subcategory relationships."""
    db = SessionLocal()
    
    try:
        print("Starting SQLite database update...")
        
        # 1. Check if tables exist
        required_tables = ['subcategories', 'categories', 'feeds']
        for table in required_tables:
            if not check_table_exists(table):
                print(f"❌ '{table}' table does not exist.")
                return
        
        print("✅ All required tables exist")
        
        # 2. Get current columns in feeds table
        feed_columns = get_column_info('feeds')
        print(f"\nCurrent columns in 'feeds' table: {feed_columns}")
        
        # 3. Add category_id column if it doesn't exist (SQLite specific syntax)
        if 'category_id' not in feed_columns:
            print("\nAdding 'category_id' column to feeds table...")
            # SQLite doesn't support adding columns with constraints in ALTER TABLE
            # We need to recreate the table or use a workaround
            
            # First check what type categories.id uses
            category_columns = db.execute(text("PRAGMA table_info(categories);")).fetchall()
            category_id_type = "TEXT"  # default
            for col in category_columns:
                if col[1] == 'id':
                    category_id_type = col[2]
                    break
            
            try:
                # Try simple ALTER TABLE first (works in newer SQLite)
                db.execute(text(f"""
                    ALTER TABLE feeds 
                    ADD COLUMN category_id {category_id_type};
                """))
                print("✅ Added 'category_id' column")
            except Exception as e:
                print(f"⚠️ Simple ALTER failed: {e}")
                print("Using workaround for older SQLite...")
                # Workaround for older SQLite that doesn't support ALTER TABLE ADD COLUMN
                # This is complex - you might need to manually add the column via SQLite browser
                print("Please add 'category_id' column manually to 'feeds' table")
                print(f"Column type should be: {category_id_type}")
                return
        else:
            print("✅ 'category_id' column already exists")
        
        # 4. Add subcategory_id column if it doesn't exist
        if 'subcategory_id' not in feed_columns:
            print("\nAdding 'subcategory_id' column to feeds table...")
            try:
                db.execute(text(f"""
                    ALTER TABLE feeds 
                    ADD COLUMN subcategory_id TEXT;
                """))
                print("✅ Added 'subcategory_id' column")
            except Exception as e:
                print(f"⚠️ Could not add subcategory_id: {e}")
        else:
            print("✅ 'subcategory_id' column already exists")
        
        # 5. Check if categories table has uuid column
        category_columns = get_column_info('categories')
        if 'uuid' not in category_columns:
            print("\nAdding 'uuid' column to categories table...")
            try:
                db.execute(text("""
                    ALTER TABLE categories 
                    ADD COLUMN uuid TEXT;
                """))
                print("✅ Added 'uuid' column to categories table")
                
                # Generate UUIDs for existing records
                categories = db.execute(text("SELECT id FROM categories WHERE uuid IS NULL")).fetchall()
                for cat in categories:
                    new_uuid = generate_uuid()
                    db.execute(text("UPDATE categories SET uuid = :uuid WHERE id = :id"), 
                              {"uuid": new_uuid, "id": cat[0]})
                print(f"✅ Generated UUIDs for {len(categories)} categories")
            except Exception as e:
                print(f"⚠️ Could not add uuid to categories: {e}")
        else:
            print("✅ 'uuid' column already exists in categories table")
        
        # 6. Check if subcategories table exists and has uuid column
        if check_table_exists('subcategories'):
            subcategory_columns = get_column_info('subcategories')
            if 'uuid' not in subcategory_columns:
                print("\nAdding 'uuid' column to subcategories table...")
                try:
                    db.execute(text("""
                        ALTER TABLE subcategories 
                        ADD COLUMN uuid TEXT;
                    """))
                    print("✅ Added 'uuid' column to subcategories table")
                    
                    # Generate UUIDs for existing records
                    subcategories = db.execute(text("SELECT id FROM subcategories WHERE uuid IS NULL")).fetchall()
                    for subcat in subcategories:
                        new_uuid = generate_uuid()
                        db.execute(text("UPDATE subcategories SET uuid = :uuid WHERE id = :id"), 
                                  {"uuid": new_uuid, "id": subcat[0]})
                    print(f"✅ Generated UUIDs for {len(subcategories)} subcategories")
                except Exception as e:
                    print(f"⚠️ Could not add uuid to subcategories: {e}")
            else:
                print("✅ 'uuid' column already exists in subcategories table")
        else:
            print("⚠️ 'subcategories' table doesn't exist - skipping uuid column")
        
        # 7. SQLite foreign key support needs to be enabled
        print("\nEnabling SQLite foreign key support...")
        db.execute(text("PRAGMA foreign_keys = ON;"))
        print("✅ Foreign key support enabled")
        
        # 8. Create indexes for better performance
        print("\nCreating indexes for better performance...")
        indexes_to_create = [
            ("idx_feeds_category_id", "feeds", "category_id"),
            ("idx_feeds_subcategory_id", "feeds", "subcategory_id"),
        ]
        
        if check_table_exists('subcategories'):
            indexes_to_create.append(("idx_subcategories_category_id", "subcategories", "category_id"))
        
        for index_name, table_name, column_name in indexes_to_create:
            try:
                # Check if index already exists
                existing = db.execute(text(f"""
                    SELECT name FROM sqlite_master 
                    WHERE type='index' AND name='{index_name}';
                """)).fetchone()
                
                if not existing:
                    db.execute(text(f"""
                        CREATE INDEX {index_name} ON {table_name}({column_name});
                    """))
                    print(f"✅ Created index {index_name}")
                else:
                    print(f"✅ Index {index_name} already exists")
            except Exception as e:
                print(f"⚠️ Could not create index {index_name}: {e}")
        
        # 9. Commit all changes
        db.commit()
        
        print("\n" + "="*50)
        print("✅ SQLite database update completed successfully!")
        print("="*50)
        
        # 10. Show summary
        print("\nSummary of changes:")
        print("1. Added category_id column to feeds table")
        print("2. Added subcategory_id column to feeds table")
        print("3. Added uuid columns to categories and subcategories tables")
        print("4. Generated UUIDs for existing records")
        print("5. Enabled foreign key support")
        print("6. Created indexes for better performance")
        
        # 11. Note about foreign keys in SQLite
        print("\n📝 Note for SQLite:")
        print("SQLite has limited ALTER TABLE support. To add proper foreign key constraints,")
        print("you may need to:")
        print("1. Export data")
        print("2. Recreate the table with constraints")
        print("3. Import data back")
        print("\nOr use SQLite browser tool to add foreign keys manually")
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error updating database: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

def verify_changes_sqlite():
    """Verify that the changes were applied successfully in SQLite."""
    print("\n" + "="*50)
    print("Verifying SQLite database changes...")
    print("="*50)
    
    db = SessionLocal()
    
    try:
        # Check feeds table columns
        result = db.execute(text("PRAGMA table_info(feeds);")).fetchall()
        print("\nColumns in 'feeds' table:")
        new_columns_found = False
        for col in result:
            if col[1] in ['category_id', 'subcategory_id']:
                print(f"  ✅ {col[1]}: {col[2]} (NEW)")
                new_columns_found = True
            else:
                print(f"  - {col[1]}: {col[2]}")
        
        if not new_columns_found:
            print("  No new columns found in feeds table")
        
        # Check indexes
        print("\nIndexes on 'feeds' table:")
        indexes = db.execute(text("""
            SELECT name, sql FROM sqlite_master 
            WHERE type='index' AND tbl_name='feeds'
            AND name LIKE 'idx_feeds_%';
        """)).fetchall()
        
        if indexes:
            for idx in indexes:
                print(f"  ✅ {idx[0]}")
        else:
            print("  No new indexes found")
        
        # Count feeds
        result = db.execute(text("""
            SELECT 
                COUNT(*) as total_feeds,
                COUNT(CASE WHEN category_id IS NOT NULL THEN 1 END) as feeds_with_category,
                COUNT(CASE WHEN subcategory_id IS NOT NULL THEN 1 END) as feeds_with_subcategory
            FROM feeds;
        """)).fetchone()
        
        if result:
            print(f"\nFeed statistics:")
            print(f"  Total feeds: {result[0]}")
            if result[0] > 0:
                print(f"  Feeds with category: {result[1]} ({result[1]/result[0]*100:.1f}%)")
                print(f"  Feeds with subcategory: {result[2]} ({result[2]/result[0]*100:.1f}%)")
        
        # Check if categories have UUIDs
        result = db.execute(text("""
            SELECT 
                COUNT(*) as total_categories,
                COUNT(CASE WHEN uuid IS NOT NULL THEN 1 END) as categories_with_uuid
            FROM categories;
        """)).fetchone()
        
        if result:
            print(f"\nCategory statistics:")
            print(f"  Total categories: {result[0]}")
            print(f"  Categories with UUID: {result[1]} ({result[1]/result[0]*100:.1f}%)" if result[0] > 0 else "  No categories")
        
        print("\n✅ SQLite verification completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during verification: {e}")
    finally:
        db.close()

def create_subcategories_table_if_not_exists():
    """Create subcategories table if it doesn't exist."""
    db = SessionLocal()
    
    try:
        if not check_table_exists('subcategories'):
            print("\nCreating 'subcategories' table...")
            
            # Check categories table structure to match ID type
            category_info = db.execute(text("PRAGMA table_info(categories);")).fetchall()
            id_type = "TEXT"  # default
            for col in category_info:
                if col[1] == 'id':
                    id_type = col[2]
                    break
            
            # Create subcategories table
            db.execute(text(f"""
                CREATE TABLE subcategories (
                    id {id_type} PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    category_id {id_type} NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    uuid TEXT
                );
            """))
            
            db.commit()
            print("✅ Created 'subcategories' table")
            
            # Create index
            db.execute(text("""
                CREATE INDEX idx_subcategories_category_id ON subcategories(category_id);
            """))
            print("✅ Created index on category_id")
            
            return True
        else:
            print("✅ 'subcategories' table already exists")
            return False
            
    except Exception as e:
        db.rollback()
        print(f"❌ Error creating subcategories table: {e}")
        return False
    finally:
        db.close()

if __name__ == "__main__":
    print("SQLite Database Migration Script")
    print("="*50)
    
    # First, create subcategories table if needed
    create_subcategories_table_if_not_exists()
    
    # Then update the database
    update_database_sqlite()
    
    # Then verify the changes
    verify_changes_sqlite()
    
    print("\n" + "="*50)
    print("Migration script completed!")
    print("="*50)
    print("\n📋 Next steps:")
    print("1. Restart your FastAPI application")
    print("2. New feeds will automatically get category/subcategory assignments")
    print("3. Run your application to test the changes")