# migrate_profile_photo.py
from sqlalchemy import text
from database import engine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_profile_photo_column():
    """
    Add profile_photo column to users table if it doesn't exist (SQLite version)
    """
    try:
        with engine.connect() as connection:
            
            # Check table columns using SQLite PRAGMA
            result = connection.execute(text("PRAGMA table_info(users);"))
            columns = [row[1] for row in result.fetchall()]  # row[1] = column name

            if "profile_photo" not in columns:
                # Add the column
                connection.execute(
                    text("ALTER TABLE users ADD COLUMN profile_photo VARCHAR(500);")
                )
                connection.commit()
                logger.info("✅ Successfully added profile_photo column to users table")
            else:
                logger.info("ℹ️ profile_photo column already exists")

    except Exception as e:
        logger.error(f"❌ Error adding profile_photo column: {e}")
        raise


if __name__ == "__main__":
    add_profile_photo_column()
