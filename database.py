from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import settings

engine = create_engine(
    settings.DATABASE_URL, 
    connect_args={"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
# Add this to your database.py file
def get_db_session():
    """Create a new database session for background threads"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def run_migrations():
    """Manually add missing columns for new features if they don't exist."""
    from sqlalchemy import text
    
    # Columns to add: (table_name, column_name, type_def)
    new_columns = [
        ('feeds', 'embedding', 'TEXT'),
        ('feeds', 'click_count', 'INTEGER DEFAULT 0'),
        ('feeds', 'language', 'VARCHAR(10) DEFAULT "en"'),
        ('concepts', 'embedding', 'TEXT'),
        ('concepts', 'click_count', 'INTEGER DEFAULT 0'),
        ('content_lists', 'embedding', 'TEXT'),
        ('content_lists', 'click_count', 'INTEGER DEFAULT 0'),
        ('sources', 'click_count', 'INTEGER DEFAULT 0'),
        ('subcategories', 'click_count', 'INTEGER DEFAULT 0')
    ]
    
    with engine.connect() as conn:
        for table, column, type_def in new_columns:
            try:
                # SQLite syntax for adding columns
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {type_def}"))
                conn.commit()
            except Exception:
                # Column likely already exists or table doesn't exist yet
                pass