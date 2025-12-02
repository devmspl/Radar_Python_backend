from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import Dict, Any
import json
import logging
from database import get_db
import models, schemas
from dependencies import get_current_user
from datetime import datetime
from models import Role, SkillTool, UserOnboarding, UserRole, UserSkillTool, Category
router = APIRouter(
    prefix="/onboarding",
    tags=["Onboarding"]
)
logger = logging.getLogger(__name__)
# Your questionnaire data
QUESTIONNAIRE_DATA = {
    "id": "complete_onboarding",
    "title": "Complete Your Profile",
    "totalSteps": 14,
    "questions": [
      {
        "id": "q1",
        "step": 1,
        "type": "multi_select_with_input",
        "title": "Domains of interest?",
        "subtitle": "Select Limit 3 to 4",
        "required": True,
        "minSelections": 1,
        "maxSelections": 4,
        "options": [
          { "id": "ux_design", "label": "UX Design", "value": "ux_design" },
          {
            "id": "product_mgmt",
            "label": "Product Management",
            "value": "product_management"
          },
          { "id": "science", "label": "Science", "value": "science" },
          { "id": "marketing", "label": "Marketing", "value": "marketing" },
          { "id": "writing", "label": "Writing", "value": "writing" },
          { "id": "lifestyle", "label": "Lifestyle", "value": "lifestyle" },
          { "id": "coding", "label": "Coding", "value": "coding" },
          {
            "id": "photography",
            "label": "Photography",
            "value": "photography"
          },
          {
            "id": "consulting",
            "label": "Consulting & HR",
            "value": "consulting_hr"
          }
        ],
        "customInput": {
          "enabled": True,
          "placeholder": "Enter Domain",
          "label": "Or"
        }
      },
      {
        "id": "q2",
        "step": 2,
        "type": "searchable_multi_select",
        "title": "Skill & Tools",
        "subtitle": "Select Limit 3 to 4",
        "required": True,
        "minSelections": 1,
        "maxSelections": 4,
        "searchPlaceholder": "Search skills & tools",
        "options": [
          { "id": "web_design", "label": "Web Design", "value": "web_design" },
          {
            "id": "ui_ux_design",
            "label": "UI/UX Design",
            "value": "ui_ux_design"
          },
          { "id": "ai_design", "label": "AI Design", "value": "ai_design" },
          { "id": "figma", "label": "Figma", "value": "figma" },
          { "id": "sketch", "label": "Sketch", "value": "sketch" },
          { "id": "adobe_xd", "label": "Adobe XD", "value": "adobe_xd" }
        ],
        "customInput": {
          "enabled": True,
          "placeholder": "Enter Skill or Tools",
          "label": "Or"
        }
      },
      {
        "id": "q3",
        "step": 3,
        "type": "searchable_multi_select",
        "title": "What role/roles are you interested in?",
        "subtitle": "Select Limit 3 to 4",
        "required": True,
        "minSelections": 1,
        "maxSelections": 4,
        "searchPlaceholder": "Search roles",
        "options": [
          {
            "id": "ui_designer",
            "label": "UI Designer",
            "value": "ui_designer"
          },
          {
            "id": "graphic_designer",
            "label": "Graphic Designer",
            "value": "graphic_designer"
          },
          {
            "id": "web_designer",
            "label": "Web Designer",
            "value": "web_designer"
          },
          {
            "id": "ux_designer",
            "label": "UX Designer",
            "value": "ux_designer"
          },
          {
            "id": "product_designer",
            "label": "Product Designer",
            "value": "product_designer"
          }
        ],
        "customInput": {
          "enabled": True,
          "placeholder": "Enter Role",
          "label": "Or"
        }
      },
      {
        "id": "q4",
        "step": 4,
        "type": "social_links",
        "title": "Connect with social",
        "subtitle": "Add your social media profiles",
        "required": False,
        "socialPlatforms": [
          {
            "id": "linkedin",
            "name": "LinkedIn",
            "icon": "linkedinDark",
            "placeholder": "Add Link"
          },
          {
            "id": "twitter",
            "name": "Twitter",
            "icon": "twitterDark",
            "placeholder": "Add Link"
          },
          {
            "id": "github",
            "name": "GitHub",
            "icon": "github",
            "placeholder": "Add Link"
          }
        ]
      },

      {
        "id": "q5",
        "step": 5,
        "type": "multi_text_input",
        "required": True,
        "fields": [
          {
            "id": "work_email",
            "label": "Work Email",
            "placeholder": "Work email",
            "inputType": "email",
            "validation": {
              "type": "email",
              "message": "Please enter a valid work email address"
            }
          },
          {
            "id": "personal_email",
            "label": "Personal Email",
            "placeholder": "Personal email",
            "inputType": "email",
            "validation": {
              "type": "email",
              "message": "Please enter a valid personal email address"
            }
          }
        ]
      },

      {
        "id": "q6",
        "step": 6,
        "type": "radio",
        "title": "Are you looking for a job?",
        "required": True,
        "options": [
          { "id": "yes", "label": "Yes", "value": "yes" },
          { "id": "no", "label": "No", "value": "no" }
        ]
      },
      {
        "id": "q7",
        "step": 7,
        "type": "radio",
        "title": "Career Stage?",
        "required": True,
        "options": [
          {
            "id": "exploration",
            "label": "Exploration",
            "value": "exploration"
          },
          {
            "id": "establishment",
            "label": "Establishment",
            "value": "establishment"
          },
          { "id": "mid_career", "label": "Mid-Career", "value": "mid_career" },
          {
            "id": "late-career",
            "label": "Late-Career",
            "value": "late-career"
          }
        ]
      },
      {
        "id": "q8",
        "step": 8,
        "type": "radio",
        "title": "Years of experience?",
        "required": True,
        "options": [
          { "id": "0_1", "label": "0-1 Years", "value": "0_1" },
          { "id": "1_2", "label": "1-2 Years", "value": "1_2" },
          { "id": "3_5", "label": "3-5 Years", "value": "3_5" },
          { "id": "5_10", "label": "5-10 Years", "value": "5_10" },
          { "id": "10_plus", "label": "10+ Years", "value": "10_plus" }
        ]
      },
      {
        "id": "q9",
        "step": 9,
        "type": "searchable_multi_select",
        "title": "Goal - Career Growth? Interviewing? Profile building? Skill Building? Certifications? Licenses? Industry Events?",
        "required": True,
        "minSelections": 1,
        "maxSelections": 2,
        "searchPlaceholder": "Search goal",
        "options": [
          {
            "id": "web_designer",
            "label": "Web Designer",
            "value": "web_designer"
          },
          {
            "id": "ui_ux_designer",
            "label": "UI/UX Designer",
            "value": "ui_ux_designer"
          },
          {
            "id": "ai_designer",
            "label": "AI Designer",
            "value": "ai_designer"
          },
          {
            "id": "figma_designer",
            "label": "Figma Designer",
            "value": "figma_designer"
          }
        ]
      },
      {
        "id": "q10",
        "step": 10,
        "title": "Market/Geography?",
        "required": True,
        "placeholder": "e.g. San Francisco, Remote, etc.",
        "type": "searchable_multi_select",
        "minSelections": 1,
        "maxSelections": 2,
        "searchPlaceholder": "Search Market / Geography",
        "options": [
          {
            "id": "web_designer",
            "label": "Web Designer",
            "value": "web_designer"
          },
          {
            "id": "ui_ux_designer",
            "label": "UI/UX Designer",
            "value": "ui_ux_designer"
          },
          {
            "id": "ai_designer",
            "label": "AI Designer",
            "value": "ai_designer"
          },
          {
            "id": "figma_designer",
            "label": "Figma Designer",
            "value": "figma_designer"
          }
        ],
        "customInput": {
          "enabled": True,
          "inpulLabel": "Enter Market / Geography",
          "placeholder": "Enter Role",
          "label": "Or"
        }
      },
      {
        "id": "q11",
        "step": 11,
        "title": "Qualifications / Credentials?",
        "required": True,
        "placeholder": "Previous/Current companies",
        "type": "searchable_multi_select",
        "minSelections": 1,
        "maxSelections": 2,
        "searchPlaceholder": "Search Market / Geography",
        "options": [
          {
            "id": "web_designer",
            "label": "Web Designer",
            "value": "web_designer"
          },
          {
            "id": "ui_ux_designer",
            "label": "UI/UX Designer",
            "value": "ui_ux_designer"
          },
          {
            "id": "ai_designer",
            "label": "AI Designer",
            "value": "ai_designer"
          },
          {
            "id": "figma_designer",
            "label": "Figma Designer",
            "value": "figma_designer"
          }
        ]
      },
      {
        "id": "q12",
        "step": 12,
        "type": "multi_searchable_single_select",
        "required": True,
        "inputFields": [
          {
            "id": "school",
            "label": "School",
            "placeholder": "Search school",
            "inputType": "select",
            "options": [
              { "id": "sc1", "label": "School A", "value": "school_a" },
              { "id": "sc2", "label": "School B", "value": "school_b" }
            ],
            "validation": {
              "type": "required",
              "message": "School is required"
            }
          },
          {
            "id": "university",
            "label": "University",
            "placeholder": "Search university",
            "inputType": "select",
            "options": [
              {
                "id": "uni1",
                "label": "University A",
                "value": "university_a"
              },
              { "id": "uni2", "label": "University B", "value": "university_b" }
            ],
            "validation": {
              "type": "required",
              "message": "University is required"
            }
          }
        ]
      },
      {
        "id": "q13",
        "step": 13,
        "title": "Companies",
        "required": True,
        "placeholder": "Previous/Current companies",
        "type": "searchable_multi_select",
        "minSelections": 1,
        "maxSelections": 2,
        "searchPlaceholder": "Search Market / Geography",
        "options": [
          {
            "id": "web_designer",
            "label": "Web Designer",
            "value": "web_designer"
          },
          {
            "id": "ui_ux_designer",
            "label": "UI/UX Designer",
            "value": "ui_ux_designer"
          },
          {
            "id": "ai_designer",
            "label": "AI Designer",
            "value": "ai_designer"
          },
          {
            "id": "figma_designer",
            "label": "Figma Designer",
            "value": "figma_designer"
          }
        ]
      },
      {
        "id": "q14",
        "step": 14,
        "placeholder": "Certifications, memberships, etc.",
        "label": "Certifications",
        "type": "multi_searchable_single_select",
        "required": True,
        "inputFields": [
          {
            "id": "industryAffiliations",
            "label": "Industry Affiliations",
            "placeholder": "Search Industry Affiliations",
            "inputType": "select",
            "options": [
              { "id": "cert1", "label": "Certification A", "value": "cert_a" },
              { "id": "cert2", "label": "Certification B", "value": "cert_b" }
            ],
            "validation": {
              "type": "required",
              "message": "Industry Affiliations is required"
            }
          },
          {
            "id": "certifications",
            "label": "Certifications?",
            "placeholder": "Search Certifications",
            "inputType": "select",
            "options": [
              { "id": "cert1", "label": "Certification A", "value": "cert_a" },
              { "id": "cert2", "label": "Certification B", "value": "cert_b" }
            ],
            "validation": {
              "type": "required",
              "message": "Certifications is required"
            }
          }
        ]
      }
    ]
}

# @router.get("/questionnaire", response_model=schemas.QuestionnaireResponse)
# async def get_questionnaire():
#     """Get the complete onboarding questionnaire"""
#     return QUESTIONNAIRE_DATA

@router.get("/status")
async def get_onboarding_status(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's onboarding status"""
    onboarding_data = db.query(models.UserOnboarding).filter(
        models.UserOnboarding.user_id == current_user.id
    ).first()
    
    if not onboarding_data:
        return {
            "has_onboarding_data": False,
            "is_completed": False,
            "completed_steps": 0,
            "total_steps": QUESTIONNAIRE_DATA["totalSteps"]
        }
    
    # Calculate completed steps (simplified logic)
    completed_steps = 0
    if onboarding_data.domains_of_interest: completed_steps += 1
    if onboarding_data.skills_tools: completed_steps += 1
    if onboarding_data.interested_roles: completed_steps += 1
    if onboarding_data.social_links: completed_steps += 1
    if onboarding_data.work_email: completed_steps += 1
    if onboarding_data.looking_for_job: completed_steps += 1
    if onboarding_data.career_stage: completed_steps += 1
    if onboarding_data.years_experience: completed_steps += 1
    if onboarding_data.goals: completed_steps += 1
    if onboarding_data.market_geography: completed_steps += 1
    if onboarding_data.qualifications: completed_steps += 1
    if onboarding_data.education: completed_steps += 1
    if onboarding_data.companies: completed_steps += 1
    if onboarding_data.certifications: completed_steps += 1
    
    return {
        "has_onboarding_data": True,
        "is_completed": onboarding_data.is_completed,
        "completed_steps": completed_steps,
        "total_steps": QUESTIONNAIRE_DATA["totalSteps"],
        "completed_at": onboarding_data.completed_at
    }

@router.post("/step/{step_number}")
async def save_onboarding_step(
    step_number: int,
    step_data: Dict[str, Any],
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Save data for a specific onboarding step"""
    
    # Get or create onboarding data for user
    onboarding_data = db.query(models.UserOnboarding).filter(
        models.UserOnboarding.user_id == current_user.id
    ).first()
    
    if not onboarding_data:
        onboarding_data = models.UserOnboarding(user_id=current_user.id)
        db.add(onboarding_data)
        db.commit()
        db.refresh(onboarding_data)
    
    # Map step number to field
    step_mapping = {
        1: "domains_of_interest",
        2: "skills_tools", 
        3: "interested_roles",
        4: "social_links",
        5: "emails",  # Special handling needed
        6: "looking_for_job",
        7: "career_stage",
        8: "years_experience",
        9: "goals",
        10: "market_geography",
        11: "qualifications",
        12: "education",
        13: "companies",
        14: "certifications"
    }
    
    field_name = step_mapping.get(step_number)
    
    if not field_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid step number: {step_number}"
        )
    
    # Handle different step types
    if step_number == 5:
        # Handle email fields - step_data should be dict with work_email and personal_email
        if not isinstance(step_data, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Step 5 data must be a dictionary with work_email and personal_email"
            )
        
        if "work_email" in step_data:
            # Basic email validation
            if "@" not in step_data["work_email"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid work email format"
                )
            onboarding_data.work_email = step_data["work_email"]
        
        if "personal_email" in step_data:
            if "@" not in step_data["personal_email"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid personal email format"
                )
            onboarding_data.personal_email = step_data["personal_email"]
    
    elif step_number in [1, 2, 3]:
        # Get validation rules from QUESTIONNAIRE_DATA
        question_data = None
        for question in QUESTIONNAIRE_DATA["questions"]:
            if question["step"] == step_number:
                question_data = question
                break
        
        # Handle ID arrays for steps 1-3
        selected_ids = []
        custom_inputs = []
        
        # Extract IDs/values from different possible formats
        if isinstance(step_data, dict):
            # Handle custom inputs separately
            if "custom_inputs" in step_data:
                custom_inputs = step_data["custom_inputs"]
            
            if "selected_options" in step_data:
                items = step_data["selected_options"]
            elif "value" in step_data:
                items = [step_data["value"]]
            else:
                items = list(step_data.values()) if step_data else []
        elif isinstance(step_data, list):
            items = step_data
        else:
            items = [step_data]
        
        # Process each item
        for item in items:
            if isinstance(item, dict):
                # Handle {"id": 1, "value": "ux_design", "label": "UX Design"} format
                if "id" in item and item["id"] is not None:
                    selected_ids.append(item["id"])
                elif "value" in item and item["value"] is not None:
                    selected_ids.append(item["value"])
            elif isinstance(item, (int, str)):
                selected_ids.append(item)
        
        # Add custom inputs
        if custom_inputs and isinstance(custom_inputs, list):
            selected_ids.extend(custom_inputs)
        
        # Validate min/max selections if defined in questionnaire
        if question_data:
            min_selections = question_data.get("minSelections")
            max_selections = question_data.get("maxSelections")
            
            total_selections = len(selected_ids)
            
            if min_selections and total_selections < min_selections:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Minimum {min_selections} selection(s) required"
                )
            
            if max_selections and total_selections > max_selections:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Maximum {max_selections} selection(s) allowed"
                )
        
        # Convert numeric strings to integers and map string values to IDs
        processed_ids = []
        for item in selected_ids:
            if isinstance(item, str) and item.isdigit():
                processed_ids.append(int(item))
            elif isinstance(item, str):
                # Try to find in database
                found_id = None
                
                if step_number == 1:  # Domains
                    domain = db.query(Category).filter(
                        (Category.name.ilike(f"%{item}%")) |
                        (Category.name == item)
                    ).first()
                    if domain:
                        found_id = domain.id
                
                elif step_number == 2:  # Skills
                    skill = db.query(models.SkillTool).filter(
                        (models.SkillTool.name.ilike(f"%{item}%")) |
                        (models.SkillTool.name == item)
                    ).first()
                    if skill:
                        found_id = skill.id
                
                elif step_number == 3:  # Roles
                    role = db.query(models.Role).filter(
                        (models.Role.title.ilike(f"%{item}%")) |
                        (models.Role.title == item)
                    ).first()
                    if role:
                        found_id = role.id
                
                # Store ID if found, otherwise store the string (custom input)
                if found_id:
                    processed_ids.append(found_id)
                else:
                    processed_ids.append(item)
            else:
                # Direct integer ID
                processed_ids.append(item)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_processed_ids = []
        for item in processed_ids:
            if item not in seen:
                seen.add(item)
                unique_processed_ids.append(item)
        
        # Store the processed IDs
        setattr(onboarding_data, field_name, unique_processed_ids)
        
        # 🔥 Also sync to UserRole and UserSkillTool tables for consistency
        if step_number == 3:  # Roles
            # Clear existing UserRole entries
            db.query(models.UserRole).filter(
                models.UserRole.user_id == current_user.id
            ).delete()
            
            # Add to UserRole table (only for valid role IDs)
            for item in unique_processed_ids:
                if isinstance(item, int):
                    role_exists = db.query(models.Role).filter(
                        models.Role.id == item
                    ).first()
                    if role_exists:
                        user_role = models.UserRole(
                            user_id=current_user.id,
                            role_id=item,
                            seniority_level="mid_level",
                            is_current=False,
                            is_target=True
                        )
                        db.add(user_role)
        
        elif step_number == 2:  # Skills
            # Clear existing UserSkillTool entries
            db.query(models.UserSkillTool).filter(
                models.UserSkillTool.user_id == current_user.id
            ).delete()
            
            # Add to UserSkillTool table (only for valid skill IDs)
            for item in unique_processed_ids:
                if isinstance(item, int):
                    skill_exists = db.query(models.SkillTool).filter(
                        models.SkillTool.id == item
                    ).first()
                    if skill_exists:
                        user_skill = models.UserSkillTool(
                            user_id=current_user.id,
                            skill_tool_id=item,
                            proficiency_level="intermediate",
                            years_of_experience=None
                        )
                        db.add(user_skill)
    
    elif step_number == 4:  # Social links
        # Validate social links structure
        if not isinstance(step_data, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Social links must be a dictionary"
            )
        
        # Validate each URL if provided
        valid_socials = {}
        for platform, url in step_data.items():
            if url and isinstance(url, str):
                # Basic URL validation
                if url.startswith(("http://", "https://")):
                    valid_socials[platform] = url
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"{platform} URL must start with http:// or https://"
                    )
        
        setattr(onboarding_data, field_name, valid_socials)
    
    elif step_number == 6:  # Looking for job (radio button)
        if step_data not in ["yes", "no"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid value for 'looking for job'. Must be 'yes' or 'no'."
            )
        setattr(onboarding_data, field_name, step_data)
    
    elif step_number == 7:  # Career stage
        valid_stages = ["exploration", "establishment", "mid_career", "late-career"]
        if step_data not in valid_stages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid career stage. Must be one of: {', '.join(valid_stages)}"
            )
        setattr(onboarding_data, field_name, step_data)
    
    elif step_number == 8:  # Years of experience
        valid_experience = ["0_1", "1_2", "3_5", "5_10", "10_plus"]
        if step_data not in valid_experience:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid years of experience. Must be one of: {', '.join(valid_experience)}"
            )
        setattr(onboarding_data, field_name, step_data)
    
    elif isinstance(field_name, str):
        # Handle other steps normally
        if isinstance(step_data, dict):
            if "selected_options" in step_data:
                step_data = step_data["selected_options"]
            elif "value" in step_data:
                step_data = step_data["value"]
            elif "selected_value" in step_data:
                step_data = step_data["selected_value"]
        
        setattr(onboarding_data, field_name, step_data)
    
    db.commit()
    db.refresh(onboarding_data)
    
    return {
        "message": f"Step {step_number} data saved successfully",
        "step": step_number,
        "field": field_name,
        "user_id": current_user.id
    }


@router.get("/data")
async def get_onboarding_data(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's complete onboarding data with names instead of IDs"""
    onboarding_data = db.query(models.UserOnboarding).filter(
        models.UserOnboarding.user_id == current_user.id
    ).first()
    
    if not onboarding_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Onboarding data not found"
        )
    
    # Convert to dict
    result = onboarding_data.__dict__.copy()
    
    # Handle domains_of_interest (step 1) - Replace IDs with names
    domains_data = onboarding_data.domains_of_interest
    if domains_data:
        domain_names = []
        
        # Handle different data formats
        if isinstance(domains_data, str):
            try:
                # Clean and parse JSON string
                if domains_data.startswith('"[') and domains_data.endswith(']"'):
                    domains_data = domains_data[1:-1]
                domains_data = json.loads(domains_data)
            except:
                domains_data = []
        
        if isinstance(domains_data, list):
            for item in domains_data:
                if isinstance(item, int) or (isinstance(item, str) and item.isdigit()):
                    # Numeric ID - get name from Category
                    domain = db.query(Category).filter(
                        Category.id == int(item)
                    ).first()
                    if domain:
                        domain_names.append(domain.name)
                elif isinstance(item, str):
                    # String value - could be name or custom input
                    domain_names.append(item)
        
        # Replace IDs with names
        result["domains_of_interest"] = domain_names
    
    # Handle skills_tools (step 2) - Replace IDs with names
    skills_data = onboarding_data.skills_tools
    if skills_data:
        skill_names = []
        
        # Handle string format
        if isinstance(skills_data, str):
            try:
                if skills_data.startswith('"[') and skills_data.endswith(']"'):
                    skills_data = skills_data[1:-1]
                skills_data = json.loads(skills_data)
            except:
                skills_data = []
        
        if isinstance(skills_data, list):
            for item in skills_data:
                if isinstance(item, int) or (isinstance(item, str) and item.isdigit()):
                    # Numeric ID - get name from SkillTool
                    skill = db.query(models.SkillTool).filter(
                        models.SkillTool.id == int(item)
                    ).first()
                    if skill:
                        skill_names.append(skill.name)
                    else:
                        skill_names.append(f"Unknown ID: {item}")
                elif isinstance(item, str):
                    # String value - could be name or custom input
                    skill_names.append(item)
        
        # Replace IDs with names
        result["skills_tools"] = skill_names
    
    # Handle interested_roles (step 3) - Replace IDs with names
    roles_data = onboarding_data.interested_roles
    if roles_data:
        role_names = []
        
        if isinstance(roles_data, list):
            for item in roles_data:
                if isinstance(item, int) or (isinstance(item, str) and item.isdigit()):
                    # Numeric ID - get title from Role
                    role = db.query(models.Role).filter(
                        models.Role.id == int(item)
                    ).first()
                    if role:
                        role_names.append(role.title)
                    else:
                        role_names.append(f"Unknown ID: {item}")
                elif isinstance(item, str):
                    # String value - could be title or custom input
                    role_names.append(item)
        
        # Replace IDs with names
        result["interested_roles"] = role_names
    
    # Remove the detail fields from the response
    result.pop("domains_details", None)
    result.pop("skills_tools_details", None)
    result.pop("roles_details", None)
    
    # Clean up SQLAlchemy internal fields
    result.pop("_sa_instance_state", None)
    
    return result


@router.get("/user/{user_id}/domains")
def get_user_domains(
    user_id: int, 
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Get all domains of interest for a specific user"""
    try:
        # Check permissions (user can view their own data, admin can view any)
        if current_user.id != user_id and current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to view this user's data"
            )
        
        onboarding = db.query(models.UserOnboarding).filter(
            models.UserOnboarding.user_id == user_id
        ).first()
        
        if not onboarding or not onboarding.domains_of_interest:
            return {
                "user_id": user_id,
                "domains": [],
                "count": 0
            }
        
        # Parse domains data
        domains_data = onboarding.domains_of_interest
        if isinstance(domains_data, str):
            try:
                if domains_data.startswith('"['):
                    domains_data = domains_data[1:-1]
                domains_data = json.loads(domains_data)
            except:
                domains_data = []
        
        domains = []
        if isinstance(domains_data, list):
            for item in domains_data:
                if isinstance(item, int) or (isinstance(item, str) and item.isdigit()):
                    domain = db.query(Category).filter(
                        Category.id == int(item)
                    ).first()
                    if domain:
                        domains.append({
                            "id": domain.id,
                            "uuid": domain.uuid,
                            "name": domain.name,
                            "description": domain.description,
                            "is_active": domain.is_active,
                            "type": "database"
                        })
                elif isinstance(item, str):
                    # Try to find by name
                    domain = db.query(Category).filter(
                        Category.name.ilike(f"%{item}%")
                    ).first()
                    if domain:
                        domains.append({
                            "id": domain.id,
                            "uuid": domain.uuid,
                            "name": domain.name,
                            "description": domain.description,
                            "is_active": domain.is_active,
                            "type": "database"
                        })
                    else:
                        domains.append({
                            "id": None,
                            "name": item,
                            "type": "custom"
                        })
        
        return {
            "user_id": user_id,
            "domains": domains,
            "count": len(domains)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching user domains: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch user domains"
        )


@router.put("/data")
async def update_onboarding_data(
    update_data: schemas.OnboardingUpdate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update onboarding data"""
    onboarding_data = db.query(models.UserOnboarding).filter(
        models.UserOnboarding.user_id == current_user.id
    ).first()
    
    if not onboarding_data:
        # Create new onboarding data if it doesn't exist
        onboarding_data = models.UserOnboarding(
            user_id=current_user.id,
            **update_data.dict(exclude_unset=True)
        )
        db.add(onboarding_data)
    else:
        # Update existing data
        for field, value in update_data.dict(exclude_unset=True).items():
            setattr(onboarding_data, field, value)
    
    db.commit()
    db.refresh(onboarding_data)
    
    return onboarding_data

from models import Role, SkillTool,UserRole, UserSkillTool, User
from sqlalchemy.orm import Session, joinedload  # Add joinedload here
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
# ========== PYDANTIC SCHEMAS ==========

class RoleResponse(BaseModel):
    id: int
    title: str
    category: str
    description: Optional[str] = None
    is_active: bool = True

    class Config:
        from_attributes = True

class SkillToolResponse(BaseModel):
    id: int
    name: str
    category: str
    description: Optional[str] = None
    popularity: int = 0
    is_active: bool = True

    class Config:
        from_attributes = True

class UserRoleCreate(BaseModel):
    role_id: int
    seniority_level: str = "mid_level"
    is_current: bool = False
    is_target: bool = False

class UserSkillToolCreate(BaseModel):
    skill_tool_id: int
    proficiency_level: str = "intermediate"
    years_of_experience: Optional[int] = None

class BulkRolesRequest(BaseModel):
    user_id: int
    roles: List[UserRoleCreate]
    replace_existing: bool = False

class BulkSkillsToolsRequest(BaseModel):
    user_id: int
    skills_tools: List[UserSkillToolCreate]
    replace_existing: bool = False

# ========== ROLES ENDPOINTS ==========

@router.get("/roles", response_model=List[RoleResponse])
def get_roles(
    category: Optional[str] = Query(None, description="Filter by category"),
    search: Optional[str] = Query(None, description="Search by title"),
    active_only: bool = Query(True, description="Show only active roles"),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Get all available roles with filtering."""
    try:
        query = db.query(Role)
        
        if active_only:
            query = query.filter(Role.is_active == True)
        
        if category:
            query = query.filter(Role.category == category)
        
        if search:
            query = query.filter(Role.title.ilike(f'%{search}%'))
        
        query = query.order_by(Role.popularity.desc(), Role.title)
        
        roles = query.offset((page - 1) * limit).limit(limit).all()
        
        return roles
        
    except Exception as e:
        logger.error(f"Error fetching roles: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch roles"
        )

# @router.get("/roles/categories")
# def get_role_categories(db: Session = Depends(get_db)):
#     """Get all available role categories."""
#     try:
#         categories = db.query(Role.category).filter(
#             Role.is_active == True
#         ).distinct().all()
        
#         return {
#             "categories": [cat[0] for cat in categories if cat[0]]
#         }
        
#     except Exception as e:
#         logger.error(f"Error fetching role categories: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to fetch role categories"
#         )

# ========== SKILLS & TOOLS ENDPOINTS ==========

@router.get("/skills-tools", response_model=List[SkillToolResponse])
def get_skills_tools(
    category: Optional[str] = Query(None, description="Filter by category"),
    search: Optional[str] = Query(None, description="Search by name"),
    active_only: bool = Query(True, description="Show only active skills/tools"),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Get all available skills and tools with filtering."""
    try:
        query = db.query(SkillTool)
        
        if active_only:
            query = query.filter(SkillTool.is_active == True)
        
        if category:
            query = query.filter(SkillTool.category == category)
        
        if search:
            query = query.filter(SkillTool.name.ilike(f'%{search}%'))
        
        query = query.order_by(SkillTool.popularity.desc(), SkillTool.name)
        
        skills_tools = query.offset((page - 1) * limit).limit(limit).all()
        
        return skills_tools
        
    except Exception as e:
        logger.error(f"Error fetching skills/tools: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch skills/tools"
        )

# @router.get("/skills-tools/categories")
# def get_skill_tool_categories(db: Session = Depends(get_db)):
#     """Get all available skill/tool categories."""
#     try:
#         categories = db.query(SkillTool.category).filter(
#             SkillTool.is_active == True
#         ).distinct().all()
        
#         return {
#             "categories": [cat[0] for cat in categories if cat[0]]
#         }
        
#     except Exception as e:
#         logger.error(f"Error fetching skill/tool categories: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to fetch skill/tool categories"
#         )

# ========== USER ONBOARDING MANAGEMENT ==========

@router.post("/user/roles", response_model=Dict[str, Any])
def add_user_roles(
    request: BulkRolesRequest,
    db: Session = Depends(get_db)
):
    """Add roles for a user during onboarding."""
    try:
        user = db.query(User).filter(User.id == request.user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        if request.replace_existing:
            # Remove existing roles
            db.query(UserRole).filter(UserRole.user_id == request.user_id).delete()
        
        added_roles = []
        for role_data in request.roles:
            # Verify role exists
            role = db.query(Role).filter(
                Role.id == role_data.role_id,
                Role.is_active == True
            ).first()
            
            if not role:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Role with ID {role_data.role_id} not found"
                )
            
            user_role = UserRole(
                user_id=request.user_id,
                role_id=role_data.role_id,
                seniority_level=role_data.seniority_level,
                is_current=role_data.is_current,
                is_target=role_data.is_target
            )
            db.add(user_role)
            added_roles.append({
                "role_id": role.id,
                "title": role.title,
                "seniority_level": role_data.seniority_level
            })
        
        db.commit()
        
        return {
            "message": "Roles added successfully",
            "user_id": request.user_id,
            "added_roles": added_roles,
            "total_roles": len(added_roles)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error adding user roles: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add user roles"
        )

@router.post("/user/skills-tools", response_model=Dict[str, Any])
def add_user_skills_tools(
    request: BulkSkillsToolsRequest,
    db: Session = Depends(get_db)
):
    """Add skills and tools for a user during onboarding."""
    try:
        user = db.query(User).filter(User.id == request.user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        if request.replace_existing:
            # Remove existing skills/tools
            db.query(UserSkillTool).filter(UserSkillTool.user_id == request.user_id).delete()
        
        added_skills = []
        for skill_data in request.skills_tools:
            # Verify skill/tool exists
            skill_tool = db.query(SkillTool).filter(
                SkillTool.id == skill_data.skill_tool_id,
                SkillTool.is_active == True
            ).first()
            
            if not skill_tool:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Skill/Tool with ID {skill_data.skill_tool_id} not found"
                )
            
            user_skill_tool = UserSkillTool(
                user_id=request.user_id,
                skill_tool_id=skill_data.skill_tool_id,
                proficiency_level=skill_data.proficiency_level,
                years_of_experience=skill_data.years_of_experience
            )
            db.add(user_skill_tool)
            added_skills.append({
                "skill_tool_id": skill_tool.id,
                "name": skill_tool.name,
                "proficiency_level": skill_data.proficiency_level
            })
        
        db.commit()
        
        return {
            "message": "Skills/Tools added successfully",
            "user_id": request.user_id,
            "added_skills": added_skills,
            "total_skills": len(added_skills)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error adding user skills/tools: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add user skills/tools"
        )

@router.get("/user/{user_id}/roles")
def get_user_roles(user_id: int, db: Session = Depends(get_db)):
    """Get all roles for a user from onboarding data."""
    try:
        onboarding = db.query(models.UserOnboarding).filter(
            models.UserOnboarding.user_id == user_id
        ).first()
        
        if not onboarding or not onboarding.interested_roles:
            return {
                "user_id": user_id,
                "roles": []
            }
        
        # Parse roles data
        roles_data = onboarding.interested_roles
        
        # Handle string format if any
        if isinstance(roles_data, str):
            try:
                if roles_data.startswith('"['):
                    roles_data = roles_data[1:-1]
                roles_data = json.loads(roles_data)
            except:
                roles_data = []
        
        roles = []
        if isinstance(roles_data, list):
            for item in roles_data:
                if isinstance(item, int) or (isinstance(item, str) and item.isdigit()):
                    role = db.query(models.Role).filter(
                        models.Role.id == int(item)
                    ).first()
                    if role:
                        roles.append({
                            "id": role.id,
                            "role_id": role.id,
                            "title": role.title,
                            "category": role.category,
                            "seniority_level": "not_specified",  # Default
                            "is_current": False,
                            "is_target": True  # Assuming interested roles are target roles
                        })
                elif isinstance(item, str):
                    # String value - try to find in database
                    role = db.query(models.Role).filter(
                        models.Role.title.ilike(f"%{item}%")
                    ).first()
                    if role:
                        roles.append({
                            "id": role.id,
                            "role_id": role.id,
                            "title": role.title,
                            "category": role.category,
                            "seniority_level": "not_specified",
                            "is_current": False,
                            "is_target": True
                        })
        
        return {
            "user_id": user_id,
            "roles": roles
        }
        
    except Exception as e:
        logger.error(f"Error fetching user roles: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch user roles"
        )

@router.get("/user/{user_id}/skills-tools")
def get_user_skills_tools(user_id: int, db: Session = Depends(get_db)):
    """Get all skills and tools for a user from onboarding data."""
    try:
        onboarding = db.query(models.UserOnboarding).filter(
            models.UserOnboarding.user_id == user_id
        ).first()
        
        if not onboarding or not onboarding.skills_tools:
            return {
                "user_id": user_id,
                "skills_tools": []
            }
        
        # Parse skills data
        skills_data = onboarding.skills_tools
        if isinstance(skills_data, str):
            try:
                if skills_data.startswith('"['):
                    skills_data = skills_data[1:-1]
                skills_data = json.loads(skills_data)
            except:
                skills_data = []
        
        skill_ids = []
        for item in skills_data if isinstance(skills_data, list) else []:
            if isinstance(item, int):
                skill_ids.append(item)
            elif isinstance(item, str) and item.isdigit():
                skill_ids.append(int(item))
        
        # Fetch skills from database
        skills = []
        if skill_ids:
            skill_objects = db.query(models.SkillTool).filter(
                models.SkillTool.id.in_(skill_ids)
            ).all()
            
            for skill in skill_objects:
                skills.append({
                    "id": skill.id,
                    "name": skill.name,
                    "category": skill.category,
                    "skill_tool_id": skill.id,
                    "proficiency_level": "intermediate",  # Default or from another table
                    "years_of_experience": None
                })
        
        return {
            "user_id": user_id,
            "skills_tools": skills
        }
        
    except Exception as e:
        logger.error(f"Error fetching user skills/tools: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch user skills/tools"
        )

@router.get("/user/{user_id}/onboarding-status")
def get_user_onboarding_status(user_id: int, db: Session = Depends(get_db)):
    """Get user's onboarding completion status."""
    try:
        onboarding = db.query(UserOnboarding).filter(
            UserOnboarding.user_id == user_id
        ).first()
        
        user_roles_count = db.query(UserRole).filter(
            UserRole.user_id == user_id
        ).count()
        
        user_skills_count = db.query(UserSkillTool).filter(
            UserSkillTool.user_id == user_id
        ).count()
        
        return {
            "user_id": user_id,
            "onboarding_completed": onboarding.is_completed if onboarding else False,
            "completed_at": onboarding.completed_at if onboarding else None,
            "roles_count": user_roles_count,
            "skills_count": user_skills_count,
            "has_domains": bool(onboarding.domains_of_interest) if onboarding else False
        }
        
    except Exception as e:
        logger.error(f"Error fetching onboarding status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch onboarding status"
        )