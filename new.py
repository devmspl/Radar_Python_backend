# migration_script.py
from database import engine
from sqlalchemy import text

def add_feed_columns():
    """Add new columns to feeds table for SQLite."""
    with engine.connect() as conn:
        try:
            # SQLite doesn't have information_schema, so we'll use a different approach
            # Try to query the feeds table with the new column to see if it exists
            try:
                conn.execute(text("SELECT content_type FROM feeds LIMIT 1"))
                print("✅ content_type column already exists")
            except Exception:
                # Column doesn't exist, so add it
                conn.execute(text("""
                    ALTER TABLE feeds 
                    ADD COLUMN content_type VARCHAR(50) DEFAULT 'Blog'
                """))
                print("✅ Added content_type column")
            
            # Check and add skills column
            try:
                conn.execute(text("SELECT skills FROM feeds LIMIT 1"))
                print("✅ skills column already exists")
            except Exception:
                conn.execute(text("""
                    ALTER TABLE feeds 
                    ADD COLUMN skills TEXT DEFAULT '[]'
                """))
                print("✅ Added skills column")
            
            # Check and add tools column
            try:
                conn.execute(text("SELECT tools FROM feeds LIMIT 1"))
                print("✅ tools column already exists")
            except Exception:
                conn.execute(text("""
                    ALTER TABLE feeds 
                    ADD COLUMN tools TEXT DEFAULT '[]'
                """))
                print("✅ Added tools column")
            
            # Check and add roles column
            try:
                conn.execute(text("SELECT roles FROM feeds LIMIT 1"))
                print("✅ roles column already exists")
            except Exception:
                conn.execute(text("""
                    ALTER TABLE feeds 
                    ADD COLUMN roles TEXT DEFAULT '[]'
                """))
                print("✅ Added roles column")
            
            conn.commit()
            print("🎉 All columns added successfully!")
            
        except Exception as e:
            print(f"❌ Error adding columns: {e}")
            conn.rollback()
            raise

if __name__ == "__main__":
    add_feed_columns()

# fix_enum_migration.py
# from database import engine
# from sqlalchemy import text

# def fix_enum_issue():
#     """Fix the enum value mismatch in the database."""
#     with engine.connect() as conn:
#         try:
#             # Check if we have the old content_type column with string values
#             result = conn.execute(text("SELECT content_type FROM feeds LIMIT 1"))
#             row = result.fetchone()
            
#             if row and row[0] in ['Blog', 'Webinar', 'Podcast', 'Video']:
#                 print("🔄 Converting string enum values to uppercase...")
                
#                 # Update all records to use uppercase enum values
#                 conn.execute(text("""
#                     UPDATE feeds 
#                     SET content_type = UPPER(content_type)
#                     WHERE content_type IN ('Blog', 'Webinar', 'Podcast', 'Video')
#                 """))
                
#                 print("✅ Converted enum values to uppercase")
            
#             conn.commit()
#             print("🎉 Enum issue fixed!")
            
#         except Exception as e:
#             print(f"❌ Error fixing enum: {e}")
#             conn.rollback()

# if __name__ == "__main__":
#     fix_enum_issue()