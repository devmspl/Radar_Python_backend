from sqlalchemy.orm import Session
from models import Category  # your Category model
from database import engine, SessionLocal  # your DB setup
import uuid
def generate_uuid():
    return str(uuid.uuid4())
categories_data = [
    [
  {"name": "Electronics", "description": "Devices and gadgets like smartphones, laptops, and cameras.", "note": "High demand globally.", "admin_note": "Monitor for rapid technological advancements."},
  {"name": "Fashion & Apparel", "description": "Clothing, shoes, and accessories for all genders and ages.", "note": "Seasonal trends influence sales.", "admin_note": "Track fashion cycles closely."},
  {"name": "Home & Kitchen Appliances", "description": "Electronics and tools for household use, including refrigerators and blenders.", "note": "Essential for daily living.", "admin_note": "Focus on energy efficiency features."},
  {"name": "Beauty & Personal Care", "description": "Cosmetics, skincare, and grooming products.", "note": "Strong growth in online sales.", "admin_note": "Highlight organic and cruelty-free options."},
  {"name": "Health & Wellness", "description": "Supplements, fitness equipment, and wellness products.", "note": "Increased interest in healthy lifestyles.", "admin_note": "Ensure compliance with health regulations."},
  {"name": "Toys & Hobbies", "description": "Games, educational toys, and hobby supplies.", "note": "Popular during holiday seasons.", "admin_note": "Promote STEM-related toys."},
  {"name": "Food & Beverages", "description": "Groceries, snacks, and drinks.", "note": "Essential category with steady demand.", "admin_note": "Offer a variety of dietary options."},
  {"name": "Books & Media", "description": "Books, movies, music, and digital media.", "note": "Digital media consumption is rising.", "admin_note": "Diversify with e-books and audiobooks."},
  {"name": "Sports & Outdoors", "description": "Equipment and apparel for sports and outdoor activities.", "note": "Seasonal spikes in demand.", "admin_note": "Feature eco-friendly materials."},
  {"name": "Automotive", "description": "Car parts, accessories, and tools.", "note": "Essential for vehicle maintenance.", "admin_note": "Ensure compatibility with various models."},
  {"name": "Furniture & Home Decor", "description": "Indoor and outdoor furniture, lighting, and decorations.", "note": "High-value items with long purchase cycles.", "admin_note": "Highlight space-saving designs."},
  {"name": "Pet Supplies", "description": "Food, toys, and accessories for pets.", "note": "Growing pet ownership worldwide.", "admin_note": "Promote sustainable pet products."},
  {"name": "Office Supplies", "description": "Stationery, furniture, and equipment for offices.", "note": "Essential for businesses and remote workers.", "admin_note": "Offer ergonomic options."},
  {"name": "Jewelry & Watches", "description": "Rings, necklaces, and timepieces.", "note": "Popular for gifts and personal use.", "admin_note": "Emphasize craftsmanship and materials."},
  {"name": "Arts & Crafts", "description": "Supplies for painting, knitting, and other crafts.", "note": "Appeals to creative individuals.", "admin_note": "Feature eco-friendly materials."},
  {"name": "Gardening & Outdoor", "description": "Tools, seeds, and outdoor furniture.", "note": "Increased interest in home gardening.", "admin_note": "Promote sustainable gardening practices."},
  {"name": "Baby & Child Products", "description": "Diapers, toys, and clothing for children.", "note": "Consistent demand from parents.", "admin_note": "Ensure safety compliance."},
  {"name": "Travel & Luggage", "description": "Bags, accessories, and travel essentials.", "note": "Seasonal demand peaks during holidays.", "admin_note": "Highlight durable and lightweight options."},
  {"name": "Music Instruments", "description": "Guitars, keyboards, and other musical tools.", "note": "Appeals to hobbyists and professionals.", "admin_note": "Offer beginner to advanced levels."},
  {"name": "Collectibles", "description": "Limited edition items, antiques, and memorabilia.", "note": "Niche market with passionate collectors.", "admin_note": "Verify authenticity and provenance."},
  {"name": "Luxury Goods", "description": "High-end fashion, accessories, and watches.", "note": "Targeted towards affluent customers.", "admin_note": "Emphasize exclusivity and quality."},
  {"name": "Eco-Friendly Products", "description": "Sustainable and environmentally friendly items.", "note": "Growing consumer awareness.", "admin_note": "Highlight certifications and sourcing."},
  {"name": "Digital Services", "description": "Software, subscriptions, and online services.", "note": "High-margin and scalable.", "admin_note": "Offer trials and flexible plans."},
  {"name": "Gaming", "description": "Video games, consoles, and accessories.", "note": "Expanding global market.", "admin_note": "Stay updated with new releases."},
  {"name": "Photography", "description": "Cameras, lenses, and accessories.", "note": "Appeals to both professionals and enthusiasts.", "admin_note": "Offer bundles and starter kits."},
  {"name": "Party Supplies", "description": "Decorations, costumes, and event essentials.", "note": "Seasonal spikes during holidays.", "admin_note": "Promote themed collections."},
  {"name": "Camping & Hiking", "description": "Tents, backpacks, and outdoor gear.", "note": "Popular among adventure enthusiasts.", "admin_note": "Highlight durability and comfort."},
  {"name": "Bags & Wallets", "description": "Handbags, backpacks, and wallets.", "note": "Fashionable and functional.", "admin_note": "Feature versatile designs."},
  {"name": "Shoes & Footwear", "description": "Boots, sneakers, and formal shoes.", "note": "Essential category with diverse styles.", "admin_note": "Offer wide size ranges."},
  {"name": "Eyewear", "description": "Glasses, sunglasses, and accessories.", "note": "Protective and stylish.", "admin_note": "Highlight UV protection features."},
  {"name": "Smart Home Devices", "description": "Thermostats, lights, and security systems.", "note": "Increasing adoption in households.", "admin_note": "Ensure compatibility with major platforms."},
  {"name": "Virtual Reality", "description": "Headsets, games, and accessories.", "note": "Emerging technology with growth potential.", "admin_note": "Offer immersive experiences."},
  {"name": "3D Printing", "description": "Printers, filaments, and design software.", "note": "Appealing to tech enthusiasts and professionals.", "admin_note": "Provide educational resources."},
  {"name": "Subscription Boxes", "description": "Curated monthly deliveries of various products.", "note": "Popular for niche markets.", "admin_note": "Offer personalized options."},
  {"name": "Digital Art & NFTs", "description": "Artwork and collectibles in digital form.", "note": "Emerging market with speculative interest.", "admin_note": "Verify authenticity and ownership."},
  {"name": "Online Courses", "description": "Educational content and certifications.", "note": "Growing demand for self-paced learning.", "admin_note": "Offer accredited programs."},
  {"name": "Food Delivery", "description": "Meal kits and ready-to-eat meals.", "note": "Convenient for busy lifestyles.", "admin_note": "Ensure dietary variety."},
  {"name": "Fitness Equipment", "description": "Machines and accessories for workouts.", "note": "Increased focus on home fitness.", "admin_note": "Highlight space-saving designs."},
  {"name": "Smartphones & Accessories", "description": "Mobile phones, cases, and chargers.", "note": "High turnover with frequent upgrades.", "admin_note": "Stay updated with the latest models."},
  {"name": "Laptops & Tablets", "description": "Portable computers and accessories.", "note": "Essential for remote work and education.", "admin_note": "Offer performance comparisons."},
  {"name": "Home Security", "description": "Cameras, alarms, and locks.", "note": "Increasing concern for safety.", "admin_note": "Highlight ease of installation."},
  {"name": "Streaming Devices", "description": "Media players and smart TVs.", "note": "Popular for entertainment consumption.", "admin_note": "Ensure compatibility with major services."},
  {"name": "Electric Vehicles", "description": "Cars, bikes, and charging stations.", "note": "Growing interest in sustainable transportation.", "admin_note": "Provide range and charging information."},
  {"name": "Solar Energy Products", "description": "Panels, batteries, and accessories.", "note": "Increasing adoption for energy savings.", "admin_note": "Highlight efficiency ratings."},
  {"name": "Wearable Technology", "description": "Smartwatches, fitness trackers, and health monitors.", "note": "Popular for health tracking.", "admin_note": "Offer compatibility with major platforms."},
  {"name": "Smart Glasses", "description": "Eyewear with integrated technology.", "note": "Emerging market with growth potential.", "admin_note": "Provide user guides and tutorials."},
  {"name": "Robotics", "description": "Robots for various applications, including home and industrial use.", "note": "Advancing technology with diverse uses.", "admin_note": "Offer customization options."},
  {"name": "Drones", "description": "Unmanned aerial vehicles for photography and recreation.", "note": "Popular among hobbyists and professionals.", "admin_note": "Ensure compliance with local regulations."},
  {"name": "Electric Scooters", "description": "Battery-powered personal transportation.", "note": "Increasing adoption in urban areas.", "admin_note": "Highlight safety features."},
  {"name": "Smart Rings", "description": "Wearable rings with technology integration.", "note": "Innovative product with niche appeal.", "admin_note": "Provide compatibility information."},
  {"name": "Smart Fabrics", "description": "Textiles with integrated technology.", "note": "Emerging market with potential applications.", "admin_note": "Highlight durability and comfort."},
  {"name": "Augmented Reality", "description": "Technology that overlays digital information on the real world.", "note": "Growing interest in various industries.", "admin_note": "Offer development tools and resources."},
  {"name": "Machine Learning", "description": "Algorithms and tools for predictive analytics.", "note": "High demand in tech industry.", "admin_note": "Include educational resources."},
  {"name": "Blockchain Technology", "description": "Decentralized ledgers and cryptocurrency tools.", "note": "Emerging financial solutions.", "admin_note": "Ensure regulatory compliance."},
  {"name": "Cybersecurity", "description": "Software and services to protect digital assets.", "note": "Growing concern for data protection.", "admin_note": "Highlight certifications and standards."},
  {"name": "Cloud Computing", "description": "Cloud services and storage solutions.", "note": "High adoption by businesses.", "admin_note": "Offer scalable plans."},
  {"name": "Big Data Analytics", "description": "Tools to analyze large datasets.", "note": "Important for business intelligence.", "admin_note": "Provide integration guides."},
  {"name": "Smart Cities", "description": "Urban infrastructure with IoT integration.", "note": "Emerging global trend.", "admin_note": "Focus on sustainability features."},
  {"name": "Electric Bicycles", "description": "Battery-powered bicycles for commuting.", "note": "Eco-friendly transportation option.", "admin_note": "Highlight battery range."},
  {"name": "3D Scanning", "description": "Devices and software for 3D modeling.", "note": "Used in manufacturing and design.", "admin_note": "Provide usage tutorials."},
  {"name": "Voice Assistants", "description": "AI-driven smart speakers and assistants.", "note": "Growing household adoption.", "admin_note": "Ensure compatibility with apps."},
  {"name": "Smart Thermostats", "description": "Temperature control devices for homes.", "note": "Popular for energy savings.", "admin_note": "Highlight remote control features."},
  {"name": "Smart Locks", "description": "Digital locks for security and convenience.", "note": "High interest in home automation.", "admin_note": "Ensure compatibility with smart home systems."},
  {"name": "3D Modeling Software", "description": "Tools for creating 3D designs.", "note": "Used in architecture, gaming, and design.", "admin_note": "Offer tutorials and templates."},
  {"name": "Streaming Platforms", "description": "Online services for music, movies, and TV.", "note": "Growing consumption of digital content.", "admin_note": "Provide subscription options."},
  {"name": "E-Learning Platforms", "description": "Websites and apps for online education.", "note": "High adoption by learners.", "admin_note": "Offer certificates and accreditation."},
  {"name": "Smart Wearables", "description": "Clothing and accessories with tech features.", "note": "Emerging fashion-tech category.", "admin_note": "Highlight functionality and durability."},
  {"name": "Electric Skateboards", "description": "Battery-powered skateboards for urban commuting.", "note": "Popular with young adults.", "admin_note": "Include safety gear recommendations."},
  {"name": "Indoor Gardening", "description": "Devices and kits for growing plants indoors.", "note": "Rising interest in urban gardening.", "admin_note": "Highlight low-maintenance solutions."},
  {"name": "Aquatic Sports Gear", "description": "Equipment for swimming, diving, and water sports.", "note": "Seasonal demand and hobbyist interest.", "admin_note": "Include safety and durability info."},
  {"name": "Home Office Furniture", "description": "Desks, chairs, and accessories for remote work.", "note": "Increased demand due to home offices.", "admin_note": "Focus on ergonomics."},
  {"name": "Smart Lighting", "description": "IoT-enabled lighting solutions.", "note": "Popular in modern homes.", "admin_note": "Highlight energy efficiency."},
  {"name": "Electric Motorcycles", "description": "Battery-powered two-wheel vehicles.", "note": "Eco-friendly transport.", "admin_note": "Provide range and charging info."},
  {"name": "AI-Powered Tools", "description": "Software applications using AI to automate tasks.", "note": "Increasing adoption in industries.", "admin_note": "Include user guides."},
  {"name": "Smart Appliances", "description": "Home appliances with internet connectivity.", "note": "Convenient and energy-efficient.", "admin_note": "Highlight remote control features."},
  {"name": "Wearable Health Devices", "description": "Devices monitoring vitals and fitness.", "note": "Rising interest in personal health tracking.", "admin_note": "Include data privacy info."},
  {"name": "EdTech Tools", "description": "Technology solutions for educational purposes.", "note": "Rapidly growing sector.", "admin_note": "Offer tutorials for educators."},
  {"name": "Mobile Payment Solutions", "description": "Apps and systems for cashless transactions.", "note": "High adoption worldwide.", "admin_note": "Ensure security and compliance."},
  {"name": "Smart Kitchens", "description": "Connected appliances for modern cooking.", "note": "Appealing for home automation enthusiasts.", "admin_note": "Highlight convenience features."},
  {"name": "Pet Tech", "description": "Technology products for pets, like feeders and trackers.", "note": "Niche but growing market.", "admin_note": "Include usability instructions."},
  {"name": "Voice-Controlled Devices", "description": "Devices operated via voice commands.", "note": "Convenience in smart homes.", "admin_note": "Highlight supported commands."},
  {"name": "Autonomous Vehicles", "description": "Self-driving cars and transport solutions.", "note": "Emerging tech in transportation.", "admin_note": "Ensure regulatory compliance."},
  {"name": "Smart Mirrors", "description": "Mirrors with integrated displays and functionality.", "note": "Rising trend in luxury homes.", "admin_note": "Provide installation guidance."},
  {"name": "Eco-Friendly Transportation", "description": "Sustainable vehicles like e-bikes and electric scooters.", "note": "Rising environmental awareness.", "admin_note": "Highlight safety and sustainability."},
  {"name": "AI Art Tools", "description": "Software that creates or enhances art using AI.", "note": "Popular in creative tech circles.", "admin_note": "Ensure copyright clarity."},
  {"name": "Telehealth Solutions", "description": "Remote healthcare platforms and devices.", "note": "Growing adoption post-pandemic.", "admin_note": "Ensure privacy and security."},
  {"name": "Renewable Energy Solutions", "description": "Solar, wind, and sustainable energy products.", "note": "High interest in energy independence.", "admin_note": "Highlight efficiency metrics."},
  {"name": "Smart Sensors", "description": "IoT-enabled sensors for homes and industries.", "note": "Used for monitoring and automation.", "admin_note": "Provide compatibility info."},
  {"name": "Electric Boats", "description": "Battery-powered watercraft.", "note": "Niche eco-friendly market.", "admin_note": "Highlight range and charging options."},
  {"name": "Virtual Fitness Platforms", "description": "Online platforms for guided workouts.", "note": "Growing trend in home fitness.", "admin_note": "Offer subscription models."},
  {"name": "AI Chatbots", "description": "Automated conversational agents for businesses.", "note": "High adoption in customer service.", "admin_note": "Provide integration guidance."},
  {"name": "Wearable Payment Devices", "description": "Rings, watches, or bracelets enabling transactions.", "note": "Innovative fintech product.", "admin_note": "Highlight security features."},
  {"name": "Smart Glassware", "description": "Glasses with tech features like AR or fitness tracking.", "note": "Emerging tech in wearables.", "admin_note": "Include usage tutorials."},
  {"name": "Remote Work Tools", "description": "Software and devices for efficient telecommuting.", "note": "Growing due to remote work trend.", "admin_note": "Focus on collaboration features."},
  {"name": "Green Building Materials", "description": "Eco-friendly construction products.", "note": "High demand in sustainable construction.", "admin_note": "Include certification info."},
  {"name": "Smart Healthcare Devices", "description": "Connected devices monitoring patient health.", "note": "Rising demand in healthcare.", "admin_note": "Ensure regulatory compliance."},
  {"name": "AI Music Tools", "description": "Software creating or assisting in music production.", "note": "Emerging trend in music tech.", "admin_note": "Include copyright guidance."},
  {"name": "Connected Vehicles", "description": "Vehicles with internet-enabled features.", "note": "Growing automotive trend.", "admin_note": "Highlight safety and connectivity features."},
  {"name": "Smart Fitness Wearables", "description": "Devices monitoring activity and performance.", "note": "Popular among fitness enthusiasts.", "admin_note": "Provide compatibility info."},
  {"name": "Electric Aircraft", "description": "Battery-powered aviation solutions.", "note": "Emerging sustainable transportation.", "admin_note": "Highlight regulatory compliance."},
  {"name": "AI Robotics", "description": "Intelligent robots for home or industrial tasks.", "note": "Advanced tech sector.", "admin_note": "Include programming resources."},
  {"name": "Smart Transportation Systems", "description": "IoT-enabled traffic and transport management.", "note": "Urban efficiency solutions.", "admin_note": "Provide integration info."},
  {"name": "Voice-Activated Appliances", "description": "Home appliances controlled via voice.", "note": "Convenience feature for smart homes.", "admin_note": "Ensure compatibility with assistants."},
  {"name": "Wearable AR Devices", "description": "Augmented reality wearables like glasses and helmets.", "note": "Emerging immersive technology.", "admin_note": "Provide tutorials and demos."},
  {"name": "AI Personal Assistants", "description": "Intelligent assistants for scheduling, reminders, and tasks.", "note": "High adoption in productivity tools.", "admin_note": "Include privacy considerations."},
  {"name": "Eco-Friendly Packaging", "description": "Sustainable packaging for products.", "note": "Increasing consumer demand.", "admin_note": "Highlight biodegradability."}
]

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
