from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Any
import json

from database import get_db
import models, schemas
from dependencies import get_current_user
from datetime import datetime
router = APIRouter(
    prefix="/onboarding",
    tags=["Onboarding"]
)

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
        5: {"work_email", "personal_email"},
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
    
    # Handle step 5 separately as it has multiple fields
    if step_number == 5:
        if "work_email" in step_data:
            onboarding_data.work_email = step_data["work_email"]
        if "personal_email" in step_data:
            onboarding_data.personal_email = step_data["personal_email"]
    elif isinstance(field_name, str):
    # 🩹 Handle {"selected_options": [...]} or {"value": "..."} wrappers
      if isinstance(step_data, dict):
          if "selected_options" in step_data:
            step_data = step_data["selected_options"]
          elif "value" in step_data:
            step_data = step_data["value"]
          elif "selected_value" in step_data:  # just in case
            step_data = step_data["selected_value"]
      setattr(onboarding_data, field_name, step_data)


    
    db.commit()
    db.refresh(onboarding_data)
    
    return {"message": f"Step {step_number} data saved successfully"}

@router.post("/complete")
async def complete_onboarding(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mark onboarding as completed"""
    onboarding_data = db.query(models.UserOnboarding).filter(
        models.UserOnboarding.user_id == current_user.id
    ).first()
    
    if not onboarding_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Onboarding data not found"
        )
    
    # Check if all required fields are filled (simplified validation)
    required_fields = [
        onboarding_data.domains_of_interest,
        onboarding_data.skills_tools,
        onboarding_data.interested_roles,
        onboarding_data.work_email,
        onboarding_data.looking_for_job,
        onboarding_data.career_stage,
        onboarding_data.years_experience,
        onboarding_data.goals,
        onboarding_data.market_geography
    ]
    
    if not all(required_fields):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please complete all required fields before finishing onboarding"
        )
    
    onboarding_data.is_completed = True
    onboarding_data.completed_at = datetime.utcnow()
    
    db.commit()
    
    return {"message": "Onboarding completed successfully"}

@router.get("/data", response_model=schemas.OnboardingResponse)
async def get_onboarding_data(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's complete onboarding data"""
    onboarding_data = db.query(models.UserOnboarding).filter(
        models.UserOnboarding.user_id == current_user.id
    ).first()
    
    if not onboarding_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Onboarding data not found"
        )
    
    return onboarding_data

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