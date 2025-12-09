#!/usr/bin/env python3
"""
Script to add subcategories to ALL categories in the database
"""

import sys
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

# Update with your database URL
DATABASE_URL = "sqlite:///database.db"

def get_subcategory_mapping():
    """Return comprehensive subcategory mapping for ALL categories"""
    return {
        # UX Domain
        "UX": [
            "User Research Methods",
            "Usability Testing",
            "Information Architecture",
            "Interaction Design",
            "Wireframing & Prototyping",
            "User Journey Mapping",
            "Accessibility Design",
            "UX Writing & Microcopy",
            "Mobile UX Design",
            "Enterprise UX"
        ],
        
        # Frontend Domain
        "Frontend": [
            "HTML5 & Semantic Markup",
            "CSS3 & Modern Layouts",
            "JavaScript ES6+",
            "React.js Framework",
            "Vue.js Framework",
            "Angular Framework",
            "TypeScript Development",
            "Responsive Web Design",
            "Frontend Performance",
            "Testing (Jest, Cypress)"
        ],
        
        # Backend Domain
        "Backend": [
            "Node.js & Express",
            "Python (Django/Flask)",
            "Java (Spring Boot)",
            "Ruby on Rails",
            "PHP (Laravel)",
            "Go (Golang)",
            "Database Design & SQL",
            "RESTful API Development",
            "Authentication (OAuth, JWT)",
            "Microservices Architecture"
        ],
        
        # Fullstack Domain
        "Fullstack": [
            "MERN Stack (MongoDB, Express, React, Node)",
            "MEAN Stack (MongoDB, Express, Angular, Node)",
            "MEVN Stack (MongoDB, Express, Vue, Node)",
            "JAMStack Architecture",
            "Progressive Web Apps (PWA)",
            "Serverless Architecture",
            "DevOps Fundamentals",
            "Cloud Deployment (AWS, Azure)",
            "Fullstack Testing",
            "System Design Patterns"
        ],
        
        # DevOps Domain
        "DevOps": [
            "Docker & Containerization",
            "Kubernetes Orchestration",
            "CI/CD Pipeline Setup",
            "Infrastructure as Code (Terraform)",
            "Cloud Platforms (AWS, GCP, Azure)",
            "Monitoring & Logging (Prometheus, ELK)",
            "Configuration Management",
            "Security & Compliance",
            "Scripting & Automation",
            "Git & Version Control Strategies"
        ],
        
        # Mobile Domain
        "Mobile": [
            "React Native Development",
            "Flutter Development",
            "iOS Development (Swift)",
            "Android Development (Kotlin)",
            "Cross-Platform Solutions",
            "Mobile UI/UX Patterns",
            "App Store Optimization (ASO)",
            "Push Notification Systems",
            "Mobile Security",
            "Performance Optimization"
        ],
        
        # Technology Domain
        "Technology": [
            "Artificial Intelligence Trends",
            "Machine Learning Applications",
            "Blockchain Technology",
            "Cybersecurity Threats",
            "Cloud Computing Innovations",
            "Internet of Things (IoT)",
            "Quantum Computing Basics",
            "Edge Computing",
            "5G & Network Technology",
            "AR/VR Development Tools"
        ],
        
        # UI Design Domain
        "UI Design": [
            "Visual Design Principles",
            "Typography & Font Pairing",
            "Color Theory & Palettes",
            "Layout & Grid Systems",
            "Design Systems & Component Libraries",
            "Icon & Illustration Design",
            "Motion Design & Animation",
            "Design Tools (Figma, Sketch)",
            "Accessibility Standards",
            "Mobile UI Patterns"
        ],
        
        # Data Science Domain
        "Data Science": [
            "Python for Data Analysis",
            "R Programming for Statistics",
            "SQL & Database Management",
            "Statistical Analysis Methods",
            "Machine Learning Algorithms",
            "Data Visualization Tools",
            "Big Data Technologies",
            "Natural Language Processing (NLP)",
            "Computer Vision",
            "Data Engineering Pipelines"
        ],
        
        # Electronics Domain
        "Electronics": [
            "Consumer Electronics Reviews",
            "Computer Hardware Components",
            "Smartphones & Tablets",
            "Audio Equipment & Headphones",
            "Photography Cameras & Lenses",
            "Wearable Technology",
            "Smart Home Devices",
            "Gaming Consoles & Accessories",
            "Networking Equipment",
            "Electronic Components & Circuits"
        ],
        
        # Fashion & Apparel Domain
        "Fashion & Apparel": [
            "Men's Fashion Trends",
            "Women's Fashion Collections",
            "Children's Clothing",
            "Footwear & Shoes",
            "Fashion Accessories",
            "Sustainable & Ethical Fashion",
            "Luxury Fashion Brands",
            "Sportswear & Activewear",
            "Traditional & Ethnic Wear",
            "Workwear & Uniforms"
        ],
        
        # Home & Kitchen Appliances Domain
        "Home & Kitchen Appliances": [
            "Kitchen Appliances (Refrigerators, Ovens)",
            "Cleaning Equipment (Vacuum, Mops)",
            "Heating & Cooling Systems",
            "Laundry Appliances",
            "Home Entertainment Systems",
            "Smart Home Integration",
            "Small Kitchen Appliances",
            "Cookware & Bakeware",
            "Kitchen Storage Solutions",
            "Home Improvement Tools"
        ],
        
        # Beauty & Personal Care Domain
        "Beauty & Personal Care": [
            "Skincare Routines & Products",
            "Haircare Treatments",
            "Makeup & Cosmetics",
            "Fragrances & Perfumes",
            "Personal Hygiene Products",
            "Men's Grooming Essentials",
            "Natural & Organic Products",
            "Beauty Tools & Accessories",
            "Spa & Wellness Treatments",
            "Dental Care Products"
        ],
        
        # Health & Wellness Domain
        "Health & Wellness": [
            "Fitness Equipment & Gear",
            "Nutrition & Dietary Supplements",
            "Mental Wellness Practices",
            "Personal Care Devices",
            "First Aid & Emergency Kits",
            "Alternative Medicine",
            "Sleep & Relaxation Aids",
            "Health Monitoring Devices",
            "Senior Care Products",
            "Maternity & Baby Care"
        ],
        
        # Toys & Hobbies Domain
        "Toys & Hobbies": [
            "Educational Toys",
            "Board Games & Puzzles",
            "Collectible Toys",
            "Model Building Kits",
            "Arts & Crafts Supplies",
            "Outdoor Play Equipment",
            "Electronic Toys",
            "Role-Playing Games",
            "Hobby Tools & Equipment",
            "Seasonal & Party Toys"
        ],
        
        # Food & Beverages Domain
        "Food & Beverages": [
            "Gourmet Foods",
            "Healthy Snacks",
            "Beverages & Drinks",
            "Cooking Ingredients",
            "Frozen Foods",
            "International Cuisine",
            "Organic & Natural Foods",
            "Desserts & Sweets",
            "Food Preservation",
            "Kitchen Gadgets"
        ],
        
        # Books & Media Domain
        "Books & Media": [
            "Fiction Books (All Genres)",
            "Non-Fiction & Educational",
            "Academic Textbooks",
            "E-books & Audiobooks",
            "Movies & TV Shows",
            "Music Albums & Streaming",
            "Magazines & Newspapers",
            "Educational Media",
            "Digital Art & Comics",
            "Streaming Platforms"
        ],
        
        # Sports & Outdoors Domain
        "Sports & Outdoors": [
            "Fitness Equipment",
            "Outdoor Adventure Gear",
            "Team Sports Equipment",
            "Individual Sports Gear",
            "Water Sports Equipment",
            "Winter Sports Gear",
            "Camping & Hiking Equipment",
            "Cycling Gear & Bikes",
            "Yoga & Pilates Equipment",
            "Adventure Sports Gear"
        ],
        
        # Automotive Domain
        "Automotive": [
            "Car Parts & Components",
            "Car Accessories",
            "Tools & Workshop Equipment",
            "Car Care Products",
            "Performance Upgrades",
            "Interior Accessories",
            "Exterior Accessories",
            "Safety Equipment",
            "Motorcycle Gear",
            "RV & Camping Vehicles"
        ],
        
        # Furniture & Home Decor Domain
        "Furniture & Home Decor": [
            "Living Room Furniture",
            "Bedroom Furniture",
            "Office & Workspace Furniture",
            "Outdoor & Garden Furniture",
            "Lighting Fixtures",
            "Home Decor Accessories",
            "Wall Art & Decorations",
            "Rugs & Carpets",
            "Window Treatments",
            "Storage & Organization"
        ],
        
        # Pet Supplies Domain
        "Pet Supplies": [
            "Dog Supplies & Accessories",
            "Cat Supplies & Toys",
            "Bird Care Products",
            "Fish & Aquarium Supplies",
            "Small Animal Supplies",
            "Pet Food & Nutrition",
            "Toys & Entertainment",
            "Health & Grooming Products",
            "Beds & Housing",
            "Training & Behavior Aids"
        ],
        
        # Office Supplies Domain
        "Office Supplies": [
            "Stationery & Writing Instruments",
            "Office Furniture",
            "Printing & Paper Products",
            "Organization & Storage",
            "Technology & Electronics",
            "Meeting & Presentation Tools",
            "Mailing & Shipping Supplies",
            "Cleaning & Maintenance",
            "Breakroom Supplies",
            "Safety & Security"
        ],
        
        # Jewelry & Watches Domain
        "Jewelry & Watches": [
            "Necklaces & Pendants",
            "Rings & Bands",
            "Earrings & Studs",
            "Bracelets & Bangles",
            "Watches & Timepieces",
            "Luxury Jewelry",
            "Costume & Fashion Jewelry",
            "Custom & Handmade Pieces",
            "Jewelry Care & Cleaning",
            "Gift & Presentation"
        ],
        
        # Arts & Crafts Domain
        "Arts & Crafts": [
            "Painting Supplies",
            "Drawing & Sketching",
            "Sewing & Fabric Crafts",
            "Paper Crafts",
            "Jewelry Making",
            "Pottery & Clay Work",
            "Woodworking",
            "DIY Home Decor",
            "Craft Tools & Equipment",
            "Seasonal Crafts"
        ],
        
        # Gardening & Outdoor Domain
        "Gardening & Outdoor": [
            "Gardening Tools",
            "Plants & Seeds",
            "Soil & Fertilizers",
            "Outdoor Furniture",
            "Lawn Care Equipment",
            "Watering Systems",
            "Garden Decor",
            "Pest Control",
            "Greenhouse Supplies",
            "Landscaping Materials"
        ],
        
        # Baby & Child Products Domain
        "Baby & Child Products": [
            "Diapers & Changing",
            "Baby Clothing",
            "Nursery Furniture",
            "Feeding Supplies",
            "Baby Toys",
            "Safety Products",
            "Bath & Grooming",
            "Travel Gear",
            "Educational Toys",
            "Health & Wellness"
        ],
        
        # Travel & Luggage Domain
        "Travel & Luggage": [
            "Luggage & Suitcases",
            "Travel Bags & Backpacks",
            "Travel Accessories",
            "Travel Clothing",
            "Travel Tech & Gadgets",
            "Travel Documents & Security",
            "Health & Wellness Travel",
            "Adventure Travel Gear",
            "Business Travel Essentials",
            "Family Travel Products"
        ],
        
        # Music Instruments Domain
        "Music Instruments": [
            "String Instruments",
            "Keyboard & Piano",
            "Wind Instruments",
            "Percussion Instruments",
            "Electronic Instruments",
            "Instrument Accessories",
            "Recording Equipment",
            "Sheet Music & Books",
            "Instrument Maintenance",
            "Music Software"
        ],
        
        # Collectibles Domain
        "Collectibles": [
            "Action Figures",
            "Trading Cards",
            "Coins & Currency",
            "Stamps",
            "Antiques",
            "Memorabilia",
            "Limited Edition Items",
            "Art Collectibles",
            "Vintage Items",
            "Display & Storage"
        ],
        
        # Luxury Goods Domain
        "Luxury Goods": [
            "Luxury Watches",
            "Designer Handbags",
            "High-End Jewelry",
            "Luxury Fashion",
            "Premium Electronics",
            "Luxury Travel",
            "Fine Dining",
            "Exclusive Experiences",
            "Limited Edition Products",
            "Personalized Luxury"
        ],
        
        # Eco-Friendly Products Domain
        "Eco-Friendly Products": [
            "Sustainable Materials",
            "Zero Waste Products",
            "Energy Efficient Appliances",
            "Organic & Natural",
            "Recycled Materials",
            "Biodegradable Products",
            "Eco-Friendly Packaging",
            "Water Conservation",
            "Carbon Neutral Products",
            "Sustainable Fashion"
        ],
        
        # Digital Services Domain
        "Digital Services": [
            "Software as a Service (SaaS)",
            "Cloud Storage Solutions",
            "Digital Marketing Services",
            "Web Development Services",
            "Graphic Design Services",
            "Content Creation Services",
            "IT Support Services",
            "Cybersecurity Services",
            "Data Analytics Services",
            "Consulting Services"
        ],
        
        # Gaming Domain
        "Gaming": [
            "PC Gaming Hardware",
            "Console Gaming",
            "Mobile Gaming",
            "Virtual Reality Gaming",
            "Gaming Accessories",
            "Game Development Tools",
            "Esports Equipment",
            "Board & Card Games",
            "Gaming Collectibles",
            "Gaming Merchandise"
        ],
        
        # Photography Domain
        "Photography": [
            "DSLR & Mirrorless Cameras",
            "Camera Lenses",
            "Lighting Equipment",
            "Camera Accessories",
            "Studio Equipment",
            "Drone Photography",
            "Photo Editing Software",
            "Printing & Display",
            "Videography Equipment",
            "Photography Techniques"
        ],
        
        # Party Supplies Domain
        "Party Supplies": [
            "Decorations & Balloons",
            "Tableware & Serving",
            "Party Favors",
            "Costumes & Dress-up",
            "Themed Party Supplies",
            "Invitations & Stationery",
            "Entertainment & Games",
            "Food & Beverage Serving",
            "Lighting & Sound",
            "Cleanup & Storage"
        ],
        
        # Camping & Hiking Domain
        "Camping & Hiking": [
            "Tents & Shelter",
            "Sleeping Bags & Mats",
            "Backpacks & Bags",
            "Cooking & Food Prep",
            "Clothing & Footwear",
            "Navigation & Safety",
            "Lighting & Power",
            "Tools & Equipment",
            "Hygiene & Sanitation",
            "First Aid & Survival"
        ],
        
        # Bags & Wallets Domain
        "Bags & Wallets": [
            "Backpacks",
            "Handbags & Purses",
            "Luggage & Travel Bags",
            "Messenger Bags",
            "Wallets & Card Holders",
            "Laptop Bags",
            "Sports & Gym Bags",
            "Specialty Bags",
            "Bag Accessories",
            "Materials & Care"
        ],
        
        # Shoes & Footwear Domain
        "Shoes & Footwear": [
            "Athletic Shoes",
            "Casual Shoes",
            "Dress Shoes",
            "Boots",
            "Sandals & Flip-flops",
            "Work & Safety Shoes",
            "Slippers & Indoor",
            "Children's Shoes",
            "Shoe Care Products",
            "Orthopedic Footwear"
        ],
        
        # Eyewear Domain
        "Eyewear": [
            "Prescription Glasses",
            "Sunglasses",
            "Contact Lenses",
            "Eyewear Accessories",
            "Sports Eyewear",
            "Computer Glasses",
            "Safety Glasses",
            "Fashion Eyewear",
            "Kids Eyewear",
            "Eyewear Care"
        ],
        
        # Smart Home Devices Domain
        "Smart Home Devices": [
            "Smart Lighting Systems",
            "Home Security Cameras",
            "Smart Thermostats",
            "Smart Entertainment",
            "Smart Kitchen Appliances",
            "Cleaning Robots",
            "Smart Locks",
            "Energy Management",
            "Voice Assistants",
            "Home Automation Hubs"
        ],
        
        # Virtual Reality Domain
        "Virtual Reality": [
            "VR Headsets",
            "VR Games & Experiences",
            "VR Development Tools",
            "Enterprise VR Solutions",
            "Educational VR Content",
            "VR Accessories",
            "Social VR Platforms",
            "VR Content Creation",
            "Medical VR Applications",
            "Architecture & Design VR"
        ],
        
        # 3D Printing Domain
        "3D Printing": [
            "3D Printers",
            "Printing Filaments",
            "3D Modeling Software",
            "3D Scanning Equipment",
            "Post-Processing Tools",
            "Industrial 3D Printing",
            "Educational 3D Printing",
            "3D Printing Materials",
            "Maintenance & Repair",
            "3D Printing Services"
        ],
        
        # Subscription Boxes Domain
        "Subscription Boxes": [
            "Beauty & Skincare Boxes",
            "Food & Snack Boxes",
            "Book & Reading Boxes",
            "Craft & DIY Boxes",
            "Fitness & Wellness Boxes",
            "Tech & Gadget Boxes",
            "Kids & Educational Boxes",
            "Pet Supplies Boxes",
            "Lifestyle Boxes",
            "Specialty Boxes"
        ],
        
        # Digital Art & NFTs Domain
        "Digital Art & NFTs": [
            "Digital Art Creation",
            "NFT Marketplaces",
            "Crypto Art",
            "Generative Art",
            "Digital Collectibles",
            "Blockchain for Artists",
            "3D & Motion Art",
            "Digital Art Tools",
            "NFT Investment",
            "Community & DAOs"
        ],
        
        # Online Courses Domain
        "Online Courses": [
            "Programming & Tech",
            "Business & Marketing",
            "Creative Arts",
            "Health & Wellness",
            "Language Learning",
            "Academic Subjects",
            "Professional Development",
            "Personal Development",
            "Certification Programs",
            "Kids & Teens Education"
        ],
        
        # Food Delivery Domain
        "Food Delivery": [
            "Meal Kit Services",
            "Restaurant Delivery",
            "Grocery Delivery",
            "Specialty Food Delivery",
            "Healthy Meal Plans",
            "International Cuisine",
            "Subscription Meals",
            "Corporate Catering",
            "Event Catering",
            "Food Delivery Apps"
        ],
        
        # Fitness Equipment Domain
        "Fitness Equipment": [
            "Cardio Machines",
            "Strength Training Equipment",
            "Home Gym Equipment",
            "Fitness Accessories",
            "Yoga & Pilates Equipment",
            "Outdoor Fitness Gear",
            "Recovery Equipment",
            "Smart Fitness Devices",
            "Gym Flooring & Mats",
            "Fitness Apparel"
        ],
        
        # Smartphones & Accessories Domain
        "Smartphones & Accessories": [
            "Smartphone Models",
            "Phone Cases & Covers",
            "Screen Protectors",
            "Chargers & Cables",
            "Wireless Accessories",
            "Audio Accessories",
            "Mobile Photography",
            "Phone Repair Tools",
            "Mobile Security",
            "Smartphone Apps"
        ],
        
        # Laptops & Tablets Domain
        "Laptops & Tablets": [
            "Laptop Models",
            "Tablet Devices",
            "Laptop Accessories",
            "Tablet Accessories",
            "Gaming Laptops",
            "Business Laptops",
            "Convertible Devices",
            "Charging Solutions",
            "Carrying Cases",
            "Performance Upgrades"
        ],
        
        # Home Security Domain
        "Home Security": [
            "Security Cameras",
            "Alarm Systems",
            "Smart Locks",
            "Motion Sensors",
            "Video Doorbells",
            "Security Lighting",
            "Safe & Lockboxes",
            "Monitoring Services",
            "Access Control",
            "Privacy Solutions"
        ],
        
        # Streaming Devices Domain
        "Streaming Devices": [
            "Streaming Sticks",
            "Smart TVs",
            "Media Players",
            "Gaming Consoles",
            "Streaming Accessories",
            "Audio Streaming",
            "Subscription Services",
            "Content Platforms",
            "Parental Controls",
            "Streaming Quality"
        ],
        
        # Electric Vehicles Domain
        "Electric Vehicles": [
            "Electric Cars",
            "Electric Bikes",
            "Charging Stations",
            "EV Accessories",
            "Maintenance & Service",
            "Battery Technology",
            "EV Infrastructure",
            "Government Incentives",
            "Performance EVs",
            "EV Safety"
        ],
        
        # Solar Energy Products Domain
        "Solar Energy Products": [
            "Solar Panels",
            "Solar Batteries",
            "Inverters & Converters",
            "Solar Water Heaters",
            "Solar Lighting",
            "Portable Solar",
            "Solar Monitoring",
            "Installation Services",
            "Maintenance & Repair",
            "Solar Financing"
        ],
        
        # Wearable Technology Domain
        "Wearable Technology": [
            "Smartwatches",
            "Fitness Trackers",
            "Smart Glasses",
            "Health Monitors",
            "Wearable Cameras",
            "Smart Jewelry",
            "Wearable Payments",
            "Medical Wearables",
            "Sports Wearables",
            "Wearable Apps"
        ],
        
        # Smart Glasses Domain
        "Smart Glasses": [
            "AR Smart Glasses",
            "Fitness Smart Glasses",
            "Hearing Enhancement",
            "Prescription Smart Glasses",
            "Enterprise Smart Glasses",
            "Gaming Smart Glasses",
            "Safety Smart Glasses",
            "Fashion Smart Glasses",
            "Navigation Glasses",
            "Smart Glass Apps"
        ],
        
        # Robotics Domain
        "Robotics": [
            "Home Robots",
            "Industrial Robots",
            "Educational Robots",
            "Medical Robots",
            "Service Robots",
            "Drone Technology",
            "Robotic Components",
            "Programming & Control",
            "AI Robotics",
            "Robotics Safety"
        ],
        
        # Drones Domain
        "Drones": [
            "Camera Drones",
            "Racing Drones",
            "Commercial Drones",
            "Toy Drones",
            "Drone Accessories",
            "Drone Photography",
            "Drone Regulations",
            "Drone Repair",
            "Drone Software",
            "Drone Safety"
        ],
        
        # Electric Scooters Domain
        "Electric Scooters": [
            "Commuter Scooters",
            "Performance Scooters",
            "Foldable Scooters",
            "Off-road Scooters",
            "Scooter Accessories",
            "Safety Gear",
            "Maintenance & Repair",
            "Battery Technology",
            "Scooter Sharing",
            "Regulations & Laws"
        ],
        
        # Smart Rings Domain
        "Smart Rings": [
            "Fitness Tracking Rings",
            "Payment Rings",
            "Health Monitoring Rings",
            "Access Control Rings",
            "Fashion Smart Rings",
            "Gaming Rings",
            "Notification Rings",
            "Sleep Tracking Rings",
            "Battery & Charging",
            "Compatibility & Apps"
        ],
        
        # Smart Fabrics Domain
        "Smart Fabrics": [
            "Temperature Regulating",
            "Moisture Wicking",
            "Health Monitoring Fabrics",
            "LED & Light-up Fabrics",
            "Self-cleaning Fabrics",
            "Stretch & Performance",
            "Smart Textile Sensors",
            "Sustainable Smart Fabrics",
            "Military & Protective",
            "Fashion Applications"
        ],
        
        # Augmented Reality Domain
        "Augmented Reality": [
            "AR Development Tools",
            "AR Hardware",
            "AR Applications",
            "Enterprise AR",
            "Educational AR",
            "AR Gaming",
            "AR Marketing",
            "Medical AR",
            "AR Navigation",
            "Social AR"
        ],
        
        # Machine Learning Domain
        "Machine Learning": [
            "Supervised Learning",
            "Unsupervised Learning",
            "Neural Networks",
            "Deep Learning",
            "TensorFlow Framework",
            "PyTorch Framework",
            "Model Deployment",
            "MLOps",
            "Feature Engineering",
            "Explainable AI"
        ],
        
        # Blockchain Technology Domain
        "Blockchain Technology": [
            "Cryptocurrencies",
            "Smart Contracts",
            "DeFi Platforms",
            "NFT Marketplaces",
            "Web3 Development",
            "Blockchain Platforms",
            "Crypto Wallets",
            "Blockchain Security",
            "Tokenomics",
            "DAOs & Governance"
        ],
        
        # Cybersecurity Domain
        "Cybersecurity": [
            "Network Security",
            "Application Security",
            "Cloud Security",
            "Endpoint Protection",
            "Threat Intelligence",
            "Incident Response",
            "Security Compliance",
            "Cryptography",
            "Penetration Testing",
            "Security Operations"
        ],
        
        # Cloud Computing Domain
        "Cloud Computing": [
            "AWS Services",
            "Azure Services",
            "Google Cloud",
            "Cloud Storage",
            "Cloud Security",
            "Serverless Computing",
            "Container Services",
            "Cloud Migration",
            "Cost Management",
            "Multi-Cloud Strategies"
        ],
        
        # Big Data Analytics Domain
        "Big Data Analytics": [
            "Data Processing",
            "Data Warehousing",
            "Business Intelligence",
            "Real-time Analytics",
            "Predictive Analytics",
            "Data Visualization",
            "Hadoop Ecosystem",
            "Spark Framework",
            "NoSQL Databases",
            "Data Governance"
        ],
        
        # Smart Cities Domain
        "Smart Cities": [
            "Smart Transportation",
            "Energy Management",
            "Waste Management",
            "Public Safety",
            "Urban Planning",
            "IoT Infrastructure",
            "Citizen Services",
            "Environmental Monitoring",
            "Digital Governance",
            "Smart Buildings"
        ],
        
        # Electric Bicycles Domain
        "Electric Bicycles": [
            "Commuter E-Bikes",
            "Mountain E-Bikes",
            "Folding E-Bikes",
            "Cargo E-Bikes",
            "E-Bike Accessories",
            "Battery Technology",
            "Maintenance & Repair",
            "Safety Gear",
            "Regulations",
            "E-Bike Sharing"
        ],
        
        # 3D Scanning Domain
        "3D Scanning": [
            "3D Scanners",
            "Scanning Software",
            "Photogrammetry",
            "Medical Scanning",
            "Industrial Scanning",
            "Archaeology & Preservation",
            "Quality Control",
            "Reverse Engineering",
            "Scanning Services",
            "Data Processing"
        ],
        
        # Voice Assistants Domain
        "Voice Assistants": [
            "Smart Speakers",
            "Voice App Development",
            "Natural Language Processing",
            "Voice Commerce",
            "Enterprise Voice",
            "Voice AI Technology",
            "Multilingual Support",
            "Privacy & Security",
            "Integration Platforms",
            "Future Developments"
        ],
        
        # Smart Thermostats Domain
        "Smart Thermostats": [
            "Wi-Fi Thermostats",
            "Learning Thermostats",
            "Zoned Heating/Cooling",
            "Energy Saving Features",
            "Mobile Control",
            "Integration with Smart Home",
            "Installation & Setup",
            "Maintenance",
            "Brand Comparisons",
            "Future Innovations"
        ],
        
        # Smart Locks Domain
        "Smart Locks": [
            "Keypad Locks",
            "Biometric Locks",
            "Bluetooth Locks",
            "Wi-Fi Enabled Locks",
            "Commercial Smart Locks",
            "Installation Services",
            "Security Features",
            "Integration",
            "Battery & Power",
            "Access Management"
        ],
        
        # 3D Modeling Software Domain
        "3D Modeling Software": [
            "CAD Software",
            "3D Animation Software",
            "Sculpting Tools",
            "Architectural Design",
            "Game Development Tools",
            "Industrial Design",
            "Rendering Software",
            "Simulation Software",
            "Free & Open Source",
            "Learning Resources"
        ],
        
        # Streaming Platforms Domain
        "Streaming Platforms": [
            "Video Streaming Services",
            "Music Streaming Services",
            "Live Streaming Platforms",
            "Gaming Streaming",
            "Educational Streaming",
            "Business Streaming",
            "Fitness Streaming",
            "Podcast Platforms",
            "Content Discovery",
            "Subscription Models"
        ],
        
        # E-Learning Platforms Domain
        "E-Learning Platforms": [
            "LMS Systems",
            "Course Creation Tools",
            "Interactive Learning",
            "Corporate Training",
            "K-12 Education",
            "Higher Education",
            "Skill Development",
            "Language Learning",
            "Test Preparation",
            "Certification Platforms"
        ],
        
        # Smart Wearables Domain
        "Smart Wearables": [
            "Smart Clothing",
            "Wearable Sensors",
            "Medical Wearables",
            "Sports Wearables",
            "Fashion Tech",
            "Military Wearables",
            "Elderly Care",
            "Children's Wearables",
            "Pet Wearables",
            "Wearable Data"
        ],
        
        # Electric Skateboards Domain
        "Electric Skateboards": [
            "Commuter Boards",
            "Off-road Boards",
            "Longboards",
            "Shortboards",
            "Board Accessories",
            "Safety Gear",
            "Battery & Charging",
            "Maintenance",
            "Riding Techniques",
            "Community & Events"
        ],
        
        # Indoor Gardening Domain
        "Indoor Gardening": [
            "Hydroponic Systems",
            "Grow Lights",
            "Indoor Planters",
            "Soil & Nutrients",
            "Plant Care Tools",
            "Automation Systems",
            "Vertical Gardening",
            "Herb Gardens",
            "Microgreens",
            "Gardening Kits"
        ],
        
        # Aquatic Sports Gear Domain
        "Aquatic Sports Gear": [
            "Swimming Equipment",
            "Diving Gear",
            "Surfing Equipment",
            "Kayaking & Canoeing",
            "Water Safety",
            "Wetsuits & Apparel",
            "Boating Accessories",
            "Fishing Gear",
            "Water Toys",
            "Maintenance & Care"
        ],
        
        # Home Office Furniture Domain
        "Home Office Furniture": [
            "Office Desks",
            "Ergonomic Chairs",
            "Storage Solutions",
            "Lighting",
            "Conference Furniture",
            "Standing Desks",
            "Cable Management",
            "Whiteboards & Organization",
            "Comfort Accessories",
            "Space Planning"
        ],
        
        # Smart Lighting Domain
        "Smart Lighting": [
            "Smart Bulbs",
            "Light Strips",
            "Smart Switches",
            "Outdoor Smart Lighting",
            "Color Changing Lights",
            "Automation Scenes",
            "Voice Control",
            "Energy Monitoring",
            "Installation",
            "Brand Ecosystems"
        ],
        
        # Electric Motorcycles Domain
        "Electric Motorcycles": [
            "Sport Bikes",
            "Cruiser Bikes",
            "Adventure Bikes",
            "Commuter Bikes",
            "Performance Models",
            "Charging Infrastructure",
            "Safety Gear",
            "Maintenance",
            "Battery Technology",
            "Riding Experience"
        ],
        
        # AI-Powered Tools Domain
        "AI-Powered Tools": [
            "Content Creation AI",
            "Design AI Tools",
            "Marketing Automation",
            "Customer Service AI",
            "Analytics & Insights",
            "Development Tools",
            "Productivity AI",
            "Research Tools",
            "Creative AI",
            "Business Intelligence"
        ],
        
        # Smart Appliances Domain
        "Smart Appliances": [
            "Smart Refrigerators",
            "Smart Ovens",
            "Smart Laundry",
            "Smart Dishwashers",
            "Smart Coffee Makers",
            "Energy Management",
            "Remote Control",
            "Maintenance Alerts",
            "Integration",
            "Future Innovations"
        ],
        
        # Wearable Health Devices Domain
        "Wearable Health Devices": [
            "Heart Rate Monitors",
            "Blood Pressure Monitors",
            "Glucose Monitors",
            "Sleep Trackers",
            "Activity Trackers",
            "Medical Alert Systems",
            "Fertility Trackers",
            "Mental Health Monitors",
            "Elderly Monitoring",
            "Data Privacy"
        ],
        
        # EdTech Tools Domain
        "EdTech Tools": [
            "Learning Management Systems",
            "Interactive Whiteboards",
            "Student Response Systems",
            "Assessment Tools",
            "Collaboration Platforms",
            "Adaptive Learning",
            "Parent Communication",
            "Administrative Tools",
            "Special Education",
            "Gamification"
        ],
        
        # Mobile Payment Solutions Domain
        "Mobile Payment Solutions": [
            "Digital Wallets",
            "Contactless Payments",
            "Peer-to-Peer Payments",
            "QR Code Payments",
            "Biometric Payments",
            "Merchant Solutions",
            "Security & Fraud",
            "Cross-border Payments",
            "Loyalty Programs",
            "Future Trends"
        ],
        
        # Smart Kitchens Domain
        "Smart Kitchens": [
            "Connected Appliances",
            "Inventory Management",
            "Recipe Assistance",
            "Meal Planning",
            "Food Preservation",
            "Cooking Automation",
            "Energy Efficiency",
            "Kitchen Safety",
            "Entertainment Integration",
            "Future Kitchen Tech"
        ],
        
        # Pet Tech Domain
        "Pet Tech": [
            "Smart Feeders",
            "GPS Trackers",
            "Pet Cameras",
            "Health Monitors",
            "Training Devices",
            "Smart Toys",
            "Pet Doors",
            "Grooming Tech",
            "Pet Apps",
            "Safety Devices"
        ],
        
        # Voice-Controlled Devices Domain
        "Voice-Controlled Devices": [
            "Smart Speakers",
            "Voice-Controlled TVs",
            "Home Automation",
            "Car Voice Systems",
            "Voice Commerce",
            "Accessibility Devices",
            "Privacy Controls",
            "Multi-language Support",
            "Integration APIs",
            "Future Developments"
        ],
        
        # Autonomous Vehicles Domain
        "Autonomous Vehicles": [
            "Self-driving Cars",
            "Autonomous Trucks",
            "Robo-taxis",
            "Delivery Robots",
            "Sensor Technology",
            "AI Algorithms",
            "Safety Systems",
            "Regulations",
            "Infrastructure",
            "Future Mobility"
        ],
        
        # Smart Mirrors Domain
        "Smart Mirrors": [
            "Fitness Smart Mirrors",
            "Beauty & Makeup Mirrors",
            "Retail Smart Mirrors",
            "Bathroom Smart Mirrors",
            "Health Monitoring Mirrors",
            "Interactive Displays",
            "Privacy Features",
            "Installation",
            "App Integration",
            "Future Applications"
        ],
        
        # Eco-Friendly Transportation Domain
        "Eco-Friendly Transportation": [
            "Electric Vehicles",
            "Bicycles & E-Bikes",
            "Public Transport",
            "Carpooling & Sharing",
            "Sustainable Fuels",
            "Urban Mobility",
            "Infrastructure",
            "Policy & Incentives",
            "Carbon Offsetting",
            "Future Innovations"
        ],
        
        # AI Art Tools Domain
        "AI Art Tools": [
            "Image Generation AI",
            "Art Style Transfer",
            "3D Art Generation",
            "Animation AI",
            "Music & Audio AI",
            "Writing AI Tools",
            "Video Generation",
            "Creative Assistants",
            "Ethical Considerations",
            "Commercial Applications"
        ],
        
        # Telehealth Solutions Domain
        "Telehealth Solutions": [
            "Virtual Consultations",
            "Remote Monitoring",
            "Mental Health Apps",
            "Prescription Services",
            "Medical Records",
            "Specialist Access",
            "Insurance Integration",
            "Medical Devices",
            "Privacy & Security",
            "Future of Healthcare"
        ],
        
        # Renewable Energy Solutions Domain
        "Renewable Energy Solutions": [
            "Solar Power Systems",
            "Wind Energy",
            "Hydroelectric Power",
            "Geothermal Energy",
            "Energy Storage",
            "Smart Grids",
            "Energy Efficiency",
            "Green Building",
            "Community Energy",
            "Policy & Finance"
        ],
        
        # Smart Sensors Domain
        "Smart Sensors": [
            "Environmental Sensors",
            "Motion Sensors",
            "Biometric Sensors",
            "Industrial Sensors",
            "IoT Sensors",
            "Health Sensors",
            "Agricultural Sensors",
            "Security Sensors",
            "Wireless Sensors",
            "Data Analytics"
        ],
        
        # Electric Boats Domain
        "Electric Boats": [
            "Electric Yachts",
            "Electric Speedboats",
            "Fishing Boats",
            "Pontoon Boats",
            "Charging Infrastructure",
            "Battery Technology",
            "Performance",
            "Maintenance",
            "Safety Equipment",
            "Regulations"
        ],
        
        # Virtual Fitness Platforms Domain
        "Virtual Fitness Platforms": [
            "Live Streaming Workouts",
            "On-demand Classes",
            "Personal Training Apps",
            "Fitness Challenges",
            "Community Features",
            "Progress Tracking",
            "Equipment Integration",
            "Nutrition Planning",
            "Mental Wellness",
            "Subscription Models"
        ],
        
        # AI Chatbots Domain
        "AI Chatbots": [
            "Customer Service Bots",
            "Sales & Marketing Bots",
            "HR & Recruitment Bots",
            "Educational Bots",
            "Healthcare Bots",
            "Financial Bots",
            "Multilingual Bots",
            "Voice Bots",
            "Development Platforms",
            "Analytics & Optimization"
        ],
        
        # Wearable Payment Devices Domain
        "Wearable Payment Devices": [
            "Smartwatch Payments",
            "Smart Ring Payments",
            "Payment Bracelets",
            "Contactless Cards",
            "Biometric Payments",
            "Security Features",
            "Bank Integration",
            "Loyalty Programs",
            "International Use",
            "Future Technology"
        ],
        
        # Smart Glassware Domain
        "Smart Glassware": [
            "Hydration Tracking",
            "Temperature Control",
            "Nutrition Monitoring",
            "Social Features",
            "Entertainment Integration",
            "Health Applications",
            "Design & Aesthetics",
            "Battery & Charging",
            "App Connectivity",
            "Future Innovations"
        ],
        
        # Remote Work Tools Domain
        "Remote Work Tools": [
            "Video Conferencing",
            "Collaboration Software",
            "Project Management",
            "Time Tracking",
            "Communication Tools",
            "Security Solutions",
            "Productivity Apps",
            "Virtual Offices",
            "Employee Engagement",
            "Future of Work"
        ],
        
        # Green Building Materials Domain
        "Green Building Materials": [
            "Sustainable Insulation",
            "Eco-friendly Flooring",
            "Low-VOC Paints",
            "Recycled Materials",
            "Energy-efficient Windows",
            "Green Roofing",
            "Bamboo Products",
            "Cork Materials",
            "Certifications",
            "Innovative Materials"
        ],
        
        # Smart Healthcare Devices Domain
        "Smart Healthcare Devices": [
            "Remote Patient Monitoring",
            "Telemedicine Devices",
            "Wearable Diagnostics",
            "Smart Pill Dispensers",
            "Medical Alert Systems",
            "Fitness for Health",
            "Mental Health Devices",
            "Chronic Disease Management",
            "Data Integration",
            "Regulatory Compliance"
        ],
        
        # AI Music Tools Domain
        "AI Music Tools": [
            "Music Composition AI",
            "Audio Mastering AI",
            "Voice Synthesis",
            "Music Recommendation",
            "Instrument Simulation",
            "Sound Design",
            "Music Education",
            "Production Assistance",
            "Copyright & Licensing",
            "Future of Music Tech"
        ],
        
        # Connected Vehicles Domain
        "Connected Vehicles": [
            "Vehicle-to-Vehicle (V2V)",
            "Vehicle-to-Infrastructure (V2I)",
            "In-car Entertainment",
            "Navigation Systems",
            "Safety Features",
            "Diagnostics & Maintenance",
            "Over-the-air Updates",
            "Data Privacy",
            "Insurance Telematics",
            "Future Connectivity"
        ],
        
        # Smart Fitness Wearables Domain
        "Smart Fitness Wearables": [
            "Activity Trackers",
            "Heart Rate Monitors",
            "GPS Watches",
            "Sleep Trackers",
            "Workout Guidance",
            "Recovery Tracking",
            "Community Features",
            "App Integration",
            "Battery Life",
            "Future Features"
        ],
        
        # Electric Aircraft Domain
        "Electric Aircraft": [
            "Electric Airplanes",
            "Urban Air Mobility",
            "Drone Technology",
            "Battery Systems",
            "Charging Infrastructure",
            "Regulations",
            "Safety Systems",
            "Commercial Applications",
            "Environmental Impact",
            "Future Aviation"
        ],
        
        # AI Robotics Domain
        "AI Robotics": [
            "Home Assistant Robots",
            "Industrial Automation",
            "Medical Robotics",
            "Agricultural Robots",
            "Service Robots",
            "AI Algorithms",
            "Sensor Technology",
            "Human-Robot Interaction",
            "Ethical Considerations",
            "Future Developments"
        ],
        
        # Smart Transportation Systems Domain
        "Smart Transportation Systems": [
            "Traffic Management",
            "Public Transit",
            "Parking Solutions",
            "Ride-sharing Platforms",
            "Freight & Logistics",
            "Infrastructure",
            "Data Analytics",
            "Sustainability",
            "Policy & Planning",
            "Future Mobility"
        ],
        
        # Voice-Activated Appliances Domain
        "Voice-Activated Appliances": [
            "Kitchen Appliances",
            "Home Entertainment",
            "Climate Control",
            "Lighting Systems",
            "Security Devices",
            "Laundry Appliances",
            "Cleaning Devices",
            "Integration Platforms",
            "Privacy Controls",
            "Future Home Tech"
        ],
        
        # Wearable AR Devices Domain
        "Wearable AR Devices": [
            "AR Glasses",
            "AR Headsets",
            "AR Contact Lenses",
            "Enterprise AR",
            "Consumer AR",
            "Medical AR",
            "Education AR",
            "Gaming AR",
            "Development Tools",
            "Future Applications"
        ],
        
        # AI Personal Assistants Domain
        "AI Personal Assistants": [
            "Virtual Assistants",
            "Scheduling & Calendar",
            "Task Management",
            "Information Retrieval",
            "Learning & Adaptation",
            "Multi-modal Interaction",
            "Privacy & Security",
            "Business Applications",
            "Future Capabilities",
            "Ethical AI"
        ],
        
        # Eco-Friendly Packaging Domain
        "Eco-Friendly Packaging": [
            "Biodegradable Materials",
            "Compostable Packaging",
            "Recycled Materials",
            "Minimalist Design",
            "Reusable Systems",
            "Edible Packaging",
            "Plant-based Materials",
            "Supply Chain Optimization",
            "Consumer Education",
            "Regulatory Compliance"
        ],
        
        # Revenue Operations Domain
        "Revenue Operations": [
            "Sales Operations",
            "Marketing Operations",
            "Customer Success",
            "Data Analytics",
            "Process Automation",
            "CRM Management",
            "Revenue Forecasting",
            "Cross-functional Alignment",
            "Technology Stack",
            "Performance Metrics"
        ],
        
        # React Native Domain
        "React Native": [
            "React Native Fundamentals",
            "Component Development",
            "Navigation Solutions",
            "State Management",
            "Native Modules",
            "Performance Optimization",
            "Testing Strategies",
            "App Deployment",
            "Cross-platform Development",
            "Best Practices"
        ],
        
        # Default for any unmatched categories
        "default": [
            "Fundamentals & Basics",
            "Advanced Techniques",
            "Tools & Technologies",
            "Best Practices",
            "Case Studies & Examples",
            "Industry Trends",
            "Career Development",
            "Community & Resources",
            "Certification & Training",
            "Future Developments"
        ]
    }

def add_all_subcategories():
    """Add subcategories to ALL categories in the database"""
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    try:
        from models import Category, SubCategory
        
        # Get all active categories
        categories = db.query(Category).filter(Category.is_active == True).all()
        print(f"Found {len(categories)} active categories")
        
        mapping = get_subcategory_mapping()
        total_added = 0
        total_skipped = 0
        categories_processed = 0
        
        for category in categories:
            print(f"\nProcessing: {category.name} (ID: {category.id})")
            
            # Get subcategories for this category
            subcategories = mapping.get(category.name, mapping.get("default", []))
            
            if not subcategories:
                print(f"  ⚠ No subcategories defined for '{category.name}'")
                continue
            
            added_count = 0
            skipped_count = 0
            
            for subcat_name in subcategories[:10]:  # Add up to 10 per category
                # Check if subcategory already exists
                existing = db.query(SubCategory).filter(
                    SubCategory.name == subcat_name,
                    SubCategory.category_id == category.id
                ).first()
                
                if existing:
                    skipped_count += 1
                    continue
                
                try:
                    # Create new subcategory
                    subcategory = SubCategory(
                        name=subcat_name,
                        description=f"Subcategory for {category.name}: {subcat_name}",
                        category_id=category.id,
                        is_active=True
                    )
                    
                    db.add(subcategory)
                    added_count += 1
                    total_added += 1
                    
                except IntegrityError:
                    db.rollback()
                    skipped_count += 1
                    total_skipped += 1
                    print(f"  ⚠ Duplicate entry for '{subcat_name}'")
                    continue
                except Exception as e:
                    db.rollback()
                    print(f"  ❌ Error adding '{subcat_name}': {str(e)}")
                    continue
            
            if added_count > 0:
                print(f"  ✓ Added {added_count} subcategories")
            if skipped_count > 0:
                print(f"  ⚠ Skipped {skipped_count} (already exist)")
            
            categories_processed += 1
            
            # Commit every 10 categories to avoid huge transactions
            if categories_processed % 10 == 0:
                db.commit()
                print(f"\n✅ Committed batch of 10 categories")
        
        # Final commit
        db.commit()
        
        print("\n" + "="*60)
        print("SUBCATEGORY POPULATION COMPLETE")
        print("="*60)
        print(f"Total categories processed: {categories_processed}")
        print(f"Total subcategories added: {total_added}")
        print(f"Total subcategories skipped (already exist): {total_skipped}")
        
        # Show new totals
        new_subcat_count = db.query(SubCategory).filter(SubCategory.is_active == True).count()
        print(f"New total subcategories in database: {new_subcat_count}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        db.rollback()
        return False
    finally:
        db.close()
        print("\nDatabase connection closed.")

def verify_subcategories():
    """Verify which categories have subcategories"""
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    try:
        from models import Category, SubCategory
        
        categories = db.query(Category).filter(Category.is_active == True).all()
        
        print("\nSUBCATEGORY VERIFICATION REPORT")
        print("="*60)
        
        categories_with_subs = 0
        categories_without_subs = 0
        
        for category in categories:
            subcount = db.query(SubCategory).filter(
                SubCategory.category_id == category.id,
                SubCategory.is_active == True
            ).count()
            
            if subcount > 0:
                categories_with_subs += 1
                print(f"✓ {category.name}: {subcount} subcategories")
            else:
                categories_without_subs += 1
                print(f"✗ {category.name}: No subcategories")
        
        print("\n" + "="*60)
        print(f"Categories with subcategories: {categories_with_subs}")
        print(f"Categories without subcategories: {categories_without_subs}")
        print(f"Total categories: {len(categories)}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        db.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage subcategories for all categories")
    parser.add_argument("--add", action="store_true", help="Add subcategories to all categories")
    parser.add_argument("--verify", action="store_true", help="Verify which categories have subcategories")
    parser.add_argument("--count", type=int, default=5, help="Number of subcategories per category (default: 5)")
    
    args = parser.parse_args()
    
    if args.add:
        print("Adding subcategories to ALL categories...")
        print("This will add appropriate subcategories to each domain.")
        confirm = input("Are you sure? (y/n): ").strip().lower()
        if confirm == 'y':
            add_all_subcategories()
        else:
            print("Operation cancelled.")
    elif args.verify:
        verify_subcategories()
    else:
        # Interactive mode
        print("="*60)
        print("SUBCATEGORY MANAGEMENT SYSTEM")
        print("="*60)
        print("\nOptions:")
        print("1. Add subcategories to ALL categories")
        print("2. Verify current subcategory status")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            print("\nThis will add appropriate subcategories to each category.")
            print("Each category will receive up to 5 relevant subcategories.")
            confirm = input("\nProceed? (y/n): ").strip().lower()
            if confirm == 'y':
                add_all_subcategories()
            else:
                print("Operation cancelled.")
        elif choice == "2":
            verify_subcategories()
        elif choice == "3":
            print("Goodbye!")
        else:
            print("Invalid choice.")


