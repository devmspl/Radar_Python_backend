import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import SessionLocal
from models import SkillTool, Role
from datetime import datetime

def seed_roles():
    db = SessionLocal()
    try:
        roles_data = [
            # Engineering Roles (25)
            {"title": "Software Engineer", "category": "Engineering", "description": "Develops software applications and systems", "seniority_levels": ["Junior", "Mid", "Senior", "Lead", "Principal"]},
            {"title": "Frontend Developer", "category": "Engineering", "description": "Specializes in client-side web development", "seniority_levels": ["Junior", "Mid", "Senior", "Lead"]},
            {"title": "Backend Developer", "category": "Engineering", "description": "Focuses on server-side logic and databases", "seniority_levels": ["Junior", "Mid", "Senior", "Lead", "Principal"]},
            {"title": "Full Stack Developer", "category": "Engineering", "description": "Works on both frontend and backend development", "seniority_levels": ["Junior", "Mid", "Senior", "Lead"]},
            {"title": "DevOps Engineer", "category": "Engineering", "description": "Manages infrastructure and deployment pipelines", "seniority_levels": ["Junior", "Mid", "Senior", "Lead"]},
            {"title": "Data Scientist", "category": "Data & Analytics", "description": "Analyzes complex data to extract insights", "seniority_levels": ["Junior", "Mid", "Senior", "Lead"]},
            {"title": "Data Analyst", "category": "Data & Analytics", "description": "Interprets data to help business decisions", "seniority_levels": ["Junior", "Mid", "Senior"]},
            {"title": "Machine Learning Engineer", "category": "Engineering", "description": "Builds and deploys ML models", "seniority_levels": ["Junior", "Mid", "Senior", "Lead"]},
            {"title": "AI Researcher", "category": "Research", "description": "Conducts research in artificial intelligence", "seniority_levels": ["Junior", "Mid", "Senior", "Principal"]},
            {"title": "Quality Assurance Engineer", "category": "Engineering", "description": "Tests software for quality assurance", "seniority_levels": ["Junior", "Mid", "Senior", "Lead"]},
            {"title": "Security Engineer", "category": "Engineering", "description": "Focuses on cybersecurity", "seniority_levels": ["Junior", "Mid", "Senior", "Lead"]},
            {"title": "Cloud Architect", "category": "Engineering", "description": "Designs cloud infrastructure", "seniority_levels": ["Mid", "Senior", "Lead", "Principal"]},
            {"title": "Database Administrator", "category": "Engineering", "description": "Manages database systems", "seniority_levels": ["Junior", "Mid", "Senior"]},
            {"title": "System Administrator", "category": "IT", "description": "Manages IT infrastructure", "seniority_levels": ["Junior", "Mid", "Senior"]},
            {"title": "Network Engineer", "category": "Engineering", "description": "Designs and maintains networks", "seniority_levels": ["Junior", "Mid", "Senior"]},
            {"title": "Mobile Developer", "category": "Engineering", "description": "Develops mobile applications", "seniority_levels": ["Junior", "Mid", "Senior", "Lead"]},
            {"title": "Game Developer", "category": "Engineering", "description": "Creates video games", "seniority_levels": ["Junior", "Mid", "Senior"]},
            {"title": "Embedded Systems Engineer", "category": "Engineering", "description": "Works on hardware-software integration", "seniority_levels": ["Junior", "Mid", "Senior"]},
            {"title": "Blockchain Developer", "category": "Engineering", "description": "Develops blockchain applications", "seniority_levels": ["Junior", "Mid", "Senior"]},
            {"title": "AR/VR Developer", "category": "Engineering", "description": "Creates augmented/virtual reality experiences", "seniority_levels": ["Junior", "Mid", "Senior"]},
            {"title": "Data Engineer", "category": "Engineering", "description": "Builds data pipelines", "seniority_levels": ["Junior", "Mid", "Senior", "Lead"]},
            {"title": "Data Architect", "category": "Engineering", "description": "Designs data systems", "seniority_levels": ["Senior", "Lead", "Principal"]},
            {"title": "Technical Lead", "category": "Engineering", "description": "Leads technical direction", "seniority_levels": ["Senior", "Lead"]},
            {"title": "Software Architect", "category": "Engineering", "description": "Designs system architecture", "seniority_levels": ["Senior", "Lead", "Principal"]},
            {"title": "Site Reliability Engineer", "category": "Engineering", "description": "Ensures system reliability", "seniority_levels": ["Mid", "Senior", "Lead"]},

            # Design Roles (20)
            {"title": "UX Designer", "category": "Design", "description": "Designs user experiences and interfaces", "seniority_levels": ["Junior", "Mid", "Senior", "Lead"]},
            {"title": "UI Designer", "category": "Design", "description": "Creates visual design elements", "seniority_levels": ["Junior", "Mid", "Senior"]},
            {"title": "Graphic Designer", "category": "Design", "description": "Designs visual content for various media", "seniority_levels": ["Junior", "Mid", "Senior"]},
            {"title": "Product Designer", "category": "Design", "description": "Designs product features and interfaces", "seniority_levels": ["Junior", "Mid", "Senior", "Lead"]},
            {"title": "UX Researcher", "category": "Research", "description": "Conducts user research studies", "seniority_levels": ["Junior", "Mid", "Senior"]},
            {"title": "Content Designer", "category": "Design", "description": "Designs content structure and flow", "seniority_levels": ["Junior", "Mid", "Senior"]},
            {"title": "Motion Designer", "category": "Design", "description": "Creates animated graphics and videos", "seniority_levels": ["Junior", "Mid", "Senior"]},
            {"title": "Brand Designer", "category": "Design", "description": "Develops brand identity systems", "seniority_levels": ["Mid", "Senior", "Lead"]},
            {"title": "UX Architect", "category": "Design", "description": "Designs information architecture", "seniority_levels": ["Mid", "Senior", "Lead"]},
            {"title": "Service Designer", "category": "Design", "description": "Designs service experiences", "seniority_levels": ["Mid", "Senior"]},
            {"title": "Industrial Designer", "category": "Design", "description": "Designs physical products", "seniority_levels": ["Junior", "Mid", "Senior"]},
            {"title": "3D Artist", "category": "Design", "description": "Creates 3D models and animations", "seniority_levels": ["Junior", "Mid", "Senior"]},
            {"title": "Visual Designer", "category": "Design", "description": "Creates visual concepts", "seniority_levels": ["Junior", "Mid", "Senior"]},
            {"title": "Interaction Designer", "category": "Design", "description": "Designs interactive experiences", "seniority_levels": ["Junior", "Mid", "Senior"]},
            {"title": "Design Systems Manager", "category": "Design", "description": "Manages design systems", "seniority_levels": ["Senior", "Lead"]},
            {"title": "Creative Director", "category": "Design", "description": "Leads creative vision", "seniority_levels": ["Senior", "Director"]},
            {"title": "Art Director", "category": "Design", "description": "Directs visual style", "seniority_levels": ["Mid", "Senior", "Director"]},
            {"title": "Production Designer", "category": "Design", "description": "Designs for production", "seniority_levels": ["Junior", "Mid"]},
            {"title": "Accessibility Designer", "category": "Design", "description": "Focuses on accessible design", "seniority_levels": ["Mid", "Senior"]},
            {"title": "Design Technologist", "category": "Design", "description": "Bridges design and engineering", "seniority_levels": ["Mid", "Senior"]},

            # Product & Management Roles (25)
            {"title": "Product Manager", "category": "Product", "description": "Defines product vision and strategy", "seniority_levels": ["Associate", "Mid", "Senior", "Lead", "Director"]},
            {"title": "Project Manager", "category": "Management", "description": "Manages project timelines and resources", "seniority_levels": ["Junior", "Mid", "Senior", "Director"]},
            {"title": "Product Owner", "category": "Product", "description": "Defines product requirements", "seniority_levels": ["Junior", "Mid", "Senior"]},
            {"title": "Scrum Master", "category": "Management", "description": "Facilitates agile processes", "seniority_levels": ["Junior", "Mid", "Senior"]},
            {"title": "Business Analyst", "category": "Business", "description": "Analyzes business processes", "seniority_levels": ["Junior", "Mid", "Senior"]},
            {"title": "Engineering Manager", "category": "Management", "description": "Manages engineering team", "seniority_levels": ["Manager", "Senior Manager", "Director"]},
            {"title": "Director of Engineering", "category": "Management", "description": "Directs engineering teams", "seniority_levels": ["Director", "Senior Director"]},
            {"title": "Operations Manager", "category": "Operations", "description": "Manages business operations", "seniority_levels": ["Manager", "Senior Manager", "Director"]},
            {"title": "Program Manager", "category": "Management", "description": "Manages complex programs", "seniority_levels": ["Mid", "Senior", "Director"]},
            {"title": "Technical Program Manager", "category": "Management", "description": "Manages technical programs", "seniority_levels": ["Mid", "Senior", "Director"]},
            {"title": "Product Operations Manager", "category": "Operations", "description": "Optimizes product operations", "seniority_levels": ["Mid", "Senior"]},
            {"title": "Release Manager", "category": "Management", "description": "Manages product releases", "seniority_levels": ["Mid", "Senior"]},
            {"title": "Delivery Manager", "category": "Management", "description": "Ensures project delivery", "seniority_levels": ["Mid", "Senior"]},
            {"title": "Agile Coach", "category": "Management", "description": "Coaches agile practices", "seniority_levels": ["Mid", "Senior"]},
            {"title": "Transformation Manager", "category": "Management", "description": "Leads organizational change", "seniority_levels": ["Senior", "Director"]},
            {"title": "Portfolio Manager", "category": "Management", "description": "Manages project portfolio", "seniority_levels": ["Senior", "Director"]},
            {"title": "Resource Manager", "category": "Management", "description": "Manages team resources", "seniority_levels": ["Mid", "Senior"]},
            {"title": "Risk Manager", "category": "Management", "description": "Manages project risks", "seniority_levels": ["Mid", "Senior"]},
            {"title": "Change Manager", "category": "Management", "description": "Manages organizational change", "seniority_levels": ["Mid", "Senior"]},
            {"title": "Quality Manager", "category": "Management", "description": "Manages quality processes", "seniority_levels": ["Mid", "Senior"]},
            {"title": "Process Manager", "category": "Management", "description": "Optimizes business processes", "seniority_levels": ["Mid", "Senior"]},
            {"title": "Strategy Manager", "category": "Management", "description": "Develops business strategy", "seniority_levels": ["Senior", "Director"]},
            {"title": "Innovation Manager", "category": "Management", "description": "Drives innovation initiatives", "seniority_levels": ["Senior", "Director"]},
            {"title": "Transformation Lead", "category": "Management", "description": "Leads transformation projects", "seniority_levels": ["Senior", "Director"]},
            {"title": "Business Operations Manager", "category": "Operations", "description": "Manages business operations", "seniority_levels": ["Manager", "Senior Manager"]},

            # Marketing Roles (15)
            {"title": "Content Strategist", "category": "Marketing", "description": "Plans and manages content creation", "seniority_levels": ["Junior", "Mid", "Senior"]},
            {"title": "Digital Marketer", "category": "Marketing", "description": "Executes online marketing campaigns", "seniority_levels": ["Junior", "Mid", "Senior"]},
            {"title": "SEO Specialist", "category": "Marketing", "description": "Optimizes websites for search engines", "seniority_levels": ["Junior", "Mid", "Senior"]},
            {"title": "Social Media Manager", "category": "Marketing", "description": "Manages social media presence", "seniority_levels": ["Junior", "Mid", "Senior"]},
            {"title": "Marketing Manager", "category": "Marketing", "description": "Leads marketing strategies", "seniority_levels": ["Manager", "Senior Manager", "Director"]},
            {"title": "Growth Marketer", "category": "Marketing", "description": "Focuses on user acquisition and retention", "seniority_levels": ["Mid", "Senior", "Lead"]},
            {"title": "Product Marketing Manager", "category": "Marketing", "description": "Bridges product and marketing teams", "seniority_levels": ["Mid", "Senior", "Director"]},
            {"title": "Content Marketing Manager", "category": "Marketing", "description": "Manages content marketing strategy", "seniority_levels": ["Manager", "Senior Manager"]},
            {"title": "Email Marketing Specialist", "category": "Marketing", "description": "Creates and manages email campaigns", "seniority_levels": ["Junior", "Mid", "Senior"]},
            {"title": "PPC Specialist", "category": "Marketing", "description": "Manages paid advertising campaigns", "seniority_levels": ["Junior", "Mid", "Senior"]},
            {"title": "Marketing Analyst", "category": "Analytics", "description": "Analyzes marketing performance data", "seniority_levels": ["Junior", "Mid", "Senior"]},
            {"title": "Community Manager", "category": "Marketing", "description": "Manages online communities", "seniority_levels": ["Junior", "Mid", "Senior"]},
            {"title": "Event Manager", "category": "Marketing", "description": "Plans and executes events", "seniority_levels": ["Mid", "Senior"]},
            {"title": "Public Relations Manager", "category": "Marketing", "description": "Manages public image", "seniority_levels": ["Mid", "Senior", "Director"]},
            {"title": "Brand Manager", "category": "Marketing", "description": "Manages brand identity", "seniority_levels": ["Mid", "Senior", "Director"]},

            # Sales & Customer Service Roles (15)
            {"title": "Sales Executive", "category": "Sales", "description": "Drives sales and revenue growth", "seniority_levels": ["Junior", "Mid", "Senior", "Lead"]},
            {"title": "Business Development", "category": "Business", "description": "Identifies new business opportunities", "seniority_levels": ["Junior", "Mid", "Senior", "Manager"]},
            {"title": "Customer Success Manager", "category": "Customer Service", "description": "Ensures customer satisfaction", "seniority_levels": ["Junior", "Mid", "Senior", "Lead"]},
            {"title": "Sales Manager", "category": "Sales", "description": "Leads sales team and strategy", "seniority_levels": ["Manager", "Senior Manager", "Director"]},
            {"title": "Account Executive", "category": "Sales", "description": "Manages client accounts and relationships", "seniority_levels": ["Junior", "Mid", "Senior"]},
            {"title": "Sales Development Representative", "category": "Sales", "description": "Generates and qualifies leads", "seniority_levels": ["Junior", "Mid"]},
            {"title": "Customer Support Specialist", "category": "Customer Service", "description": "Provides customer support", "seniority_levels": ["Junior", "Mid", "Senior"]},
            {"title": "Technical Support Engineer", "category": "Customer Service", "description": "Provides technical assistance", "seniority_levels": ["Junior", "Mid", "Senior"]},
            {"title": "Implementation Specialist", "category": "Customer Service", "description": "Helps customers implement products", "seniority_levels": ["Mid", "Senior"]},
            {"title": "Success Engineer", "category": "Customer Service", "description": "Technical role in customer success", "seniority_levels": ["Mid", "Senior"]},
            {"title": "Account Manager", "category": "Sales", "description": "Manages customer relationships", "seniority_levels": ["Junior", "Mid", "Senior"]},
            {"title": "Sales Operations Manager", "category": "Sales", "description": "Optimizes sales processes", "seniority_levels": ["Mid", "Senior", "Manager"]},
            {"title": "Customer Experience Manager", "category": "Customer Service", "description": "Manages customer experience", "seniority_levels": ["Manager", "Senior Manager"]},
            {"title": "Support Team Lead", "category": "Customer Service", "description": "Leads support team", "seniority_levels": ["Senior", "Lead"]},
            {"title": "Client Services Director", "category": "Customer Service", "description": "Directs client services", "seniority_levels": ["Director", "Senior Director"]},
        ]
        
        for i, role_data in enumerate(roles_data):
            role = Role(
                title=role_data["title"],
                category=role_data["category"],
                description=role_data["description"],
                seniority_levels=role_data["seniority_levels"],
                popularity=100 - i,  # Vary popularity
                is_active=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.add(role)
        
        db.commit()
        print(f"✅ Added {len(roles_data)} roles to the database")
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error seeding roles: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

def seed_skill_tools():
    db = SessionLocal()
    try:
        skills_data = [
            # Programming Languages (20)
            {"name": "Python", "category": "Programming", "description": "High-level programming language"},
            {"name": "JavaScript", "category": "Programming", "description": "Scripting language for web development"},
            {"name": "TypeScript", "category": "Programming", "description": "Typed superset of JavaScript"},
            {"name": "Java", "category": "Programming", "description": "Object-oriented programming language"},
            {"name": "C++", "category": "Programming", "description": "General-purpose programming language"},
            {"name": "C#", "category": "Programming", "description": "Microsoft programming language"},
            {"name": "Go", "category": "Programming", "description": "Google's programming language"},
            {"name": "Rust", "category": "Programming", "description": "Systems programming language"},
            {"name": "Swift", "category": "Programming", "description": "Apple's programming language"},
            {"name": "Kotlin", "category": "Programming", "description": "Modern programming language for JVM"},
            {"name": "PHP", "category": "Programming", "description": "Server-side scripting language"},
            {"name": "Ruby", "category": "Programming", "description": "Dynamic programming language"},
            {"name": "Scala", "category": "Programming", "description": "Functional programming language"},
            {"name": "R", "category": "Programming", "description": "Language for statistical computing"},
            {"name": "MATLAB", "category": "Programming", "description": "Numerical computing environment"},
            {"name": "SQL", "category": "Programming", "description": "Database query language"},
            {"name": "HTML/CSS", "category": "Programming", "description": "Web markup and styling languages"},
            {"name": "Shell Scripting", "category": "Programming", "description": "Command line scripting"},
            {"name": "Dart", "category": "Programming", "description": "Client-optimized language for apps"},
            {"name": "Perl", "category": "Programming", "description": "General-purpose programming language"},

            # Frameworks & Libraries (20)
            {"name": "React", "category": "Frontend", "description": "JavaScript library for building UIs"},
            {"name": "Vue.js", "category": "Frontend", "description": "Progressive JavaScript framework"},
            {"name": "Angular", "category": "Frontend", "description": "TypeScript-based framework"},
            {"name": "Svelte", "category": "Frontend", "description": "Compiler-based JavaScript framework"},
            {"name": "Node.js", "category": "Backend", "description": "JavaScript runtime environment"},
            {"name": "Express.js", "category": "Backend", "description": "Web framework for Node.js"},
            {"name": "Django", "category": "Backend", "description": "Python web framework"},
            {"name": "Flask", "category": "Backend", "description": "Python micro web framework"},
            {"name": "Spring Boot", "category": "Backend", "description": "Java framework"},
            {"name": "Ruby on Rails", "category": "Backend", "description": "Ruby web framework"},
            {"name": "Laravel", "category": "Backend", "description": "PHP web framework"},
            {"name": "FastAPI", "category": "Backend", "description": "Modern Python web framework"},
            {"name": "GraphQL", "category": "Backend", "description": "Query language for APIs"},
            {"name": "REST APIs", "category": "Backend", "description": "Architectural style for web services"},
            {"name": "jQuery", "category": "Frontend", "description": "JavaScript library"},
            {"name": "Bootstrap", "category": "Frontend", "description": "CSS framework"},
            {"name": "Tailwind CSS", "category": "Frontend", "description": "Utility-first CSS framework"},
            {"name": "Sass/SCSS", "category": "Frontend", "description": "CSS preprocessor"},
            {"name": "Webpack", "category": "Frontend", "description": "Module bundler"},
            {"name": "Babel", "category": "Frontend", "description": "JavaScript compiler"},

            # Databases (15)
            {"name": "MySQL", "category": "Database", "description": "Relational database management system"},
            {"name": "PostgreSQL", "category": "Database", "description": "Advanced relational database"},
            {"name": "MongoDB", "category": "Database", "description": "NoSQL document database"},
            {"name": "Redis", "category": "Database", "description": "In-memory data structure store"},
            {"name": "SQLite", "category": "Database", "description": "Lightweight relational database"},
            {"name": "Oracle", "category": "Database", "description": "Enterprise relational database"},
            {"name": "SQL Server", "category": "Database", "description": "Microsoft database management system"},
            {"name": "Cassandra", "category": "Database", "description": "NoSQL distributed database"},
            {"name": "Elasticsearch", "category": "Database", "description": "Search and analytics engine"},
            {"name": "DynamoDB", "category": "Database", "description": "NoSQL database service"},
            {"name": "Firebase", "category": "Database", "description": "Google's mobile platform"},
            {"name": "Supabase", "category": "Database", "description": "Open source Firebase alternative"},
            {"name": "Neo4j", "category": "Database", "description": "Graph database management system"},
            {"name": "InfluxDB", "category": "Database", "description": "Time series database"},
            {"name": "Couchbase", "category": "Database", "description": "NoSQL document database"},

            # Cloud & DevOps (15)
            {"name": "AWS", "category": "Cloud", "description": "Amazon Web Services cloud platform"},
            {"name": "Azure", "category": "Cloud", "description": "Microsoft cloud platform"},
            {"name": "Google Cloud", "category": "Cloud", "description": "Google cloud platform"},
            {"name": "Docker", "category": "DevOps", "description": "Containerization platform"},
            {"name": "Kubernetes", "category": "DevOps", "description": "Container orchestration platform"},
            {"name": "Jenkins", "category": "DevOps", "description": "Automation server"},
            {"name": "Git", "category": "Version Control", "description": "Distributed version control system"},
            {"name": "GitHub", "category": "Version Control", "description": "Code hosting platform"},
            {"name": "GitLab", "category": "Version Control", "description": "DevOps platform"},
            {"name": "Bitbucket", "category": "Version Control", "description": "Git repository management"},
            {"name": "Terraform", "category": "DevOps", "description": "Infrastructure as code tool"},
            {"name": "Ansible", "category": "DevOps", "description": "Configuration management tool"},
            {"name": "Prometheus", "category": "DevOps", "description": "Monitoring and alerting toolkit"},
            {"name": "Grafana", "category": "DevOps", "description": "Metrics dashboard and graph editor"},
            {"name": "Splunk", "category": "DevOps", "description": "Platform for searching and analyzing data"},

            # Design Tools (15)
            {"name": "Figma", "category": "Design", "description": "Collaborative design tool"},
            {"name": "Sketch", "category": "Design", "description": "Vector graphics editor"},
            {"name": "Adobe XD", "category": "Design", "description": "UX/UI design tool"},
            {"name": "Photoshop", "category": "Design", "description": "Image editing software"},
            {"name": "Illustrator", "category": "Design", "description": "Vector graphics editor"},
            {"name": "InDesign", "category": "Design", "description": "Desktop publishing software"},
            {"name": "After Effects", "category": "Design", "description": "Motion graphics software"},
            {"name": "Premiere Pro", "category": "Design", "description": "Video editing software"},
            {"name": "Blender", "category": "Design", "description": "3D creation suite"},
            {"name": "Canva", "category": "Design", "description": "Graphic design platform"},
            {"name": "InVision", "category": "Design", "description": "Digital product design platform"},
            {"name": "Marvel", "category": "Design", "description": "Design and prototyping tool"},
            {"name": "Principle", "category": "Design", "description": "Interactive design tool"},
            {"name": "Framer", "category": "Design", "description": "Interactive design tool"},
            {"name": "Zeplin", "category": "Design", "description": "Design handoff tool"},

            # Data & Analytics (15)
            {"name": "Tableau", "category": "Data Visualization", "description": "Data visualization tool"},
            {"name": "Power BI", "category": "Data Visualization", "description": "Business analytics tool"},
            {"name": "Excel", "category": "Productivity", "description": "Spreadsheet software"},
            {"name": "Google Sheets", "category": "Productivity", "description": "Cloud-based spreadsheet"},
            {"name": "Google Analytics", "category": "Analytics", "description": "Web analytics service"},
            {"name": "Mixpanel", "category": "Analytics", "description": "Product analytics platform"},
            {"name": "Amplitude", "category": "Analytics", "description": "Product intelligence platform"},
            {"name": "Segment", "category": "Analytics", "description": "Customer data platform"},
            {"name": "Snowflake", "category": "Data", "description": "Cloud data platform"},
            {"name": "Databricks", "category": "Data", "description": "Data and AI platform"},
            {"name": "Apache Spark", "category": "Data", "description": "Unified analytics engine"},
            {"name": "Hadoop", "category": "Data", "description": "Big data processing framework"},
            {"name": "Apache Kafka", "category": "Data", "description": "Distributed event streaming platform"},
            {"name": "Looker", "category": "Analytics", "description": "Business intelligence platform"},
            {"name": "Mode Analytics", "category": "Analytics", "description": "Analytics platform"},
        ]
        
        for i, skill_data in enumerate(skills_data):
            skill = SkillTool(
                name=skill_data["name"],
                category=skill_data["category"],
                description=skill_data["description"],
                popularity=100 - i,  # Vary popularity
                is_active=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.add(skill)
        
        db.commit()
        print(f"✅ Added {len(skills_data)} skills/tools to the database")
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error seeding skills/tools: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    print("🌱 Seeding database with roles and skills/tools...")
    seed_roles()
    seed_skill_tools()
    print("🎉 Database seeding completed!")


# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from database import engine, Base
# from models import User, UserRole, UserSkillTool

# def refresh_user_relationships():
#     """Refresh User model relationships in database"""
#     print("🔄 Refreshing User model relationships...")
    
#     # Drop and recreate only the association tables
#     UserRole.__table__.drop(engine, checkfirst=True)
#     UserSkillTool.__table__.drop(engine, checkfirst=True)
    
#     # Recreate the tables
#     UserRole.__table__.create(engine)
#     UserSkillTool.__table__.create(engine)
    
#     print("✅ User model relationships refreshed!")

# if __name__ == "__main__":
#     refresh_user_relationships()