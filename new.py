import sqlite3
import sys
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker

# Update with your actual database URL
DATABASE_URL = "sqlite:///./test.db"  # Change to your actual database

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def check_table_columns():
    """Check what columns exist in the subcategories table."""
    inspector = inspect(engine)
    columns = inspector.get_columns('subcategories')
    
    print("Current columns in 'subcategories' table:")
    for col in columns:
        print(f"  - {col['name']}: {col['type']}")
    
    return [col['name'] for col in columns]

def fix_subcategories_table():
    """Add missing columns to subcategories table."""
    db = SessionLocal()
    
    try:
        print("Fixing subcategories table...")
        
        # Check current columns
        existing_columns = check_table_columns()
        
        # List of required columns
        required_columns = [
            ('is_active', 'BOOLEAN DEFAULT TRUE'),
            ('created_at', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
            ('updated_at', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
            ('uuid', 'VARCHAR(36)')
        ]
        
        # Add missing columns
        for column_name, column_type in required_columns:
            if column_name not in existing_columns:
                print(f"Adding {column_name} column...")
                try:
                    db.execute(text(f"""
                        ALTER TABLE subcategories 
                        ADD COLUMN {column_name} {column_type};
                    """))
                    print(f"✅ Added {column_name} column")
                except Exception as e:
                    print(f"⚠️ Could not add {column_name}: {e}")
        
        # Generate UUIDs for existing rows if uuid column was just added
        if 'uuid' not in existing_columns:
            print("Generating UUIDs for existing subcategories...")
            # Get all subcategories without UUID
            import uuid
            subcategories = db.execute(text("SELECT id FROM subcategories WHERE uuid IS NULL")).fetchall()
            
            for subcat in subcategories:
                new_uuid = str(uuid.uuid4())
                db.execute(text("UPDATE subcategories SET uuid = :uuid WHERE id = :id"), 
                          {"uuid": new_uuid, "id": subcat[0]})
            
            print(f"✅ Generated UUIDs for {len(subcategories)} subcategories")
        
        db.commit()
        
        # Verify fixes
        print("\n✅ Fixed subcategories table")
        check_table_columns()
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error fixing table: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

def simple_sqlite_fix():
    """Simple SQLite fix for the immediate issue."""
    conn = sqlite3.connect('test.db')  # Change to your database file
    cursor = conn.cursor()
    
    print("Running simple SQLite fix...")
    
    # Add is_active column if it doesn't exist
    try:
        cursor.execute("ALTER TABLE subcategories ADD COLUMN is_active BOOLEAN DEFAULT TRUE;")
        print("✅ Added is_active column")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("✅ is_active column already exists")
        else:
            print(f"⚠️ Error with is_active: {e}")
    
    # Add created_at column if it doesn't exist
    try:
        cursor.execute("ALTER TABLE subcategories ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;")
        print("✅ Added created_at column")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("✅ created_at column already exists")
        else:
            print(f"⚠️ Error with created_at: {e}")
    
    # Add updated_at column if it doesn't exist
    try:
        cursor.execute("ALTER TABLE subcategories ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;")
        print("✅ Added updated_at column")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("✅ updated_at column already exists")
        else:
            print(f"⚠️ Error with updated_at: {e}")
    
    # Add uuid column if it doesn't exist
    try:
        cursor.execute("ALTER TABLE subcategories ADD COLUMN uuid TEXT;")
        print("✅ Added uuid column")
        
        # Generate UUIDs for existing rows
        import uuid
        cursor.execute("SELECT id FROM subcategories WHERE uuid IS NULL OR uuid = ''")
        subcategories = cursor.fetchall()
        
        for subcat in subcategories:
            new_uuid = str(uuid.uuid4())
            cursor.execute("UPDATE subcategories SET uuid = ? WHERE id = ?", (new_uuid, subcat[0]))
        
        print(f"✅ Generated UUIDs for {len(subcategories)} subcategories")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("✅ uuid column already exists")
        else:
            print(f"⚠️ Error with uuid: {e}")
    
    conn.commit()
    
    # Verify
    print("\n✅ Fixed subcategories table")
    
    # Show table structure
    cursor.execute("PRAGMA table_info(subcategories);")
    columns = cursor.fetchall()
    print("\nCurrent table structure:")
    for col in columns:
        print(f"  {col[1]} ({col[2]})")
    
    conn.close()

if __name__ == "__main__":
    print("Fixing subcategories table structure...")
    print("="*50)
    
    # Try the simple fix first
    simple_sqlite_fix()
    
    print("\n" + "="*50)
    print("Fix completed!")
    print("="*50)
    print("\nRestart your FastAPI application to apply changes.")