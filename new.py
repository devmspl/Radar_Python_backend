import json
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, inspect
from models import Feed, Slide, Base  # adjust import path if needed

# Replace with your SQLite DB path
DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(DATABASE_URL, echo=True)
Session = sessionmaker(bind=engine)
session = Session()

# Inspect tables to make sure they exist
inspector = inspect(engine)
print("Tables in DB:", inspector.get_table_names())

# ----- Update feeds.categories -----
feeds = session.query(Feed).all()
for feed in feeds:
    if feed.categories:
        try:
            # If already a JSON string, this will succeed
            json.loads(feed.categories)
        except (TypeError, json.JSONDecodeError):
            # Convert list/other to JSON string
            feed.categories = json.dumps(feed.categories)
session.commit()
print("Updated feeds.categories to JSON.")

# ----- Update slides.bullets -----
slides = session.query(Slide).all()
for slide in slides:
    if slide.bullets:
        try:
            json.loads(slide.bullets)
        except (TypeError, json.JSONDecodeError):
            slide.bullets = json.dumps(slide.bullets)
session.commit()
print("Updated slides.bullets to JSON.")

session.close()
print("Migration complete!")
