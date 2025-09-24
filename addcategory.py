from sqlalchemy.orm import Session
from models import Category  # your Category model
from database import engine, SessionLocal  # your DB setup
import uuid
def generate_uuid():
    return str(uuid.uuid4())
categories_data =[
  {"name": "YouTube Channels", "description": "Content sourced from YouTube channels relevant to revenue operations and sales.", "note": "Includes RevGenius, RevOps Co-op, DealHub, Gong, etc.", "admin_note": "Track channel popularity and update links periodically."},
  {"name": "Revenue Operations", "description": "Topics related to Revenue Operations (RevOps) strategy, tools, and best practices.", "note": "Includes RevOps Co-op, Revenue Operations Alliance, ayeQ RevOps.", "admin_note": "Focus on emerging trends in RevOps."},
  {"name": "Sales Enablement", "description": "Content on sales techniques, execution platforms, and productivity.", "note": "Includes Masterclass content, Aptitude 8 HubSpot tutorials.", "admin_note": "Categorize by methodology and sales stage."},
  {"name": "CRM & HubSpot", "description": "Content related to CRM platforms and HubSpot tools.", "note": "Includes tutorials and guides.", "admin_note": "Keep updated with new HubSpot features."},
  {"name": "Incentive Compensation", "description": "Articles and resources on incentive plans and sales performance management.", "note": "Includes Xactly Corp content.", "admin_note": "Highlight trends in sales compensation."},
  {"name": "Revenue Intelligence", "description": "Insights and tools for revenue intelligence and analytics.", "note": "Includes Gong blogs, podcasts, webinars.", "admin_note": "Focus on data-backed insights."},
  {"name": "Webinars & Videos", "description": "Recorded webinars, instructional videos, and video blogs.", "note": "Includes content from company websites and YouTube.", "admin_note": "Track release dates and formats."},
  {"name": "Blogs & Articles", "description": "Written content from company blogs covering revenue, sales, and operations.", "note": "Includes Outreach.io Blog, Gong Blog, Zip Blog.", "admin_note": "Ensure accurate tagging by topic."},
  {"name": "Reports & Guides", "description": "Downloadable reports, guides, and research papers.", "note": "Includes industry reports, customer success guides, and research from Zip and Gong.", "admin_note": "Include publication dates and sources."},
  {"name": "Customer Stories & Case Studies", "description": "Examples and success stories from customers.", "note": "Includes Outreach, Gong, and Zip customer stories.", "admin_note": "Highlight key metrics and outcomes."},
  {"name": "Product News & Updates", "description": "Announcements and updates about company products.", "note": "Includes Outreach.io product news.", "admin_note": "Track version and feature updates."},
  {"name": "Events & Webinars", "description": "Information on industry events, online webinars, and live sessions.", "note": "Includes Zip industry webinars and events, Gong live events.", "admin_note": "Include registration links and dates."},
  {"name": "Podcasts", "description": "Audio content covering revenue operations and sales intelligence.", "note": "Includes Gong's Reveal: Revenue AI Podcast.", "admin_note": "Track episode numbers and topics."},
  {"name": "Procurement & Sales Tools", "description": "Content and resources related to procurement and sales tools.", "note": "Includes Zip resources for procurement, tools content.", "admin_note": "Focus on software and process optimization."},
  {"name": "Educational Resources", "description": "Guides, tutorials, and courses to improve skills in sales and revenue operations.", "note": "Includes HubSpot courses, Masterclass content.", "admin_note": "Track skill levels and relevance."}
]



def add_categories(categories_list, admin_id):
    db: Session = SessionLocal()
    try:
        category_objects = []
        for cat in categories_list:
            category = Category(
                name=cat["name"],
                description=cat.get("description"),
                note=cat.get("note"),
                admin_note=cat.get("admin_note"),
                admin_id=admin_id  # assign admin who creates
            )
            category_objects.append(category)
        
        db.add_all(category_objects)
        db.commit()
        print(f"Inserted {len(category_objects)} categories successfully!")
    except Exception as e:
        db.rollback()
        print("Error inserting categories:", e)
    finally:
        db.close()

add_categories(categories_data, admin_id=2)
