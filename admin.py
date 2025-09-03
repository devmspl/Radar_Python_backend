from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from database import get_db
from dependencies import get_current_admin
import models
import schemas
from dependencies import get_current_admin 
router = APIRouter(prefix="/admin", tags=["admin"])

@router.post("/categories", response_model=schemas.CategoryResponse)
def create_category(
    category: schemas.CategoryCreate,
    db: Session = Depends(get_db),
    admin_user: models.User = Depends(get_current_admin)
):
    # Check if category name already exists
    db_category = db.query(models.Category).filter(
        models.Category.name == category.name
    ).first()
    
    if db_category:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Category name already exists"
        )
    
    # Create new category
    db_category = models.Category(
        name=category.name,
        description=category.description,
        note=category.note,
        admin_note=category.admin_note,
        admin_id=admin_user.id
    )
    
    db.add(db_category)
    db.commit()
    db.refresh(db_category)
    
    return db_category

@router.get("/categories", response_model=List[schemas.CategoryResponse])
def get_categories(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    admin_user: models.User = Depends(get_current_admin)
):
    categories = db.query(models.Category).offset(skip).limit(limit).all()
    return categories

@router.get("/categories/{category_id}", response_model=schemas.CategoryResponse)
def get_category(
    category_id: int,
    db: Session = Depends(get_db),
    admin_user: models.User = Depends(get_current_admin)
):
    category = db.query(models.Category).filter(models.Category.id == category_id).first()
    
    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Category not found"
        )
    
    return category

@router.get("/categories/uuid/{category_uuid}", response_model=schemas.CategoryResponse)
def get_category_by_uuid(
    category_uuid: str,
    db: Session = Depends(get_db),
    admin_user: models.User = Depends(get_current_admin)
):
    category = db.query(models.Category).filter(models.Category.uuid == category_uuid).first()
    
    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Category not found"
        )
    
    return category

@router.put("/categories/{category_id}", response_model=schemas.CategoryResponse)
def update_category(
    category_id: int,
    category_update: schemas.CategoryUpdate,
    db: Session = Depends(get_db),
    admin_user: models.User = Depends(get_current_admin)
):
    db_category = db.query(models.Category).filter(models.Category.id == category_id).first()
    
    if not db_category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Category not found"
        )
    
    # Check if new name already exists (if name is being updated)
    if category_update.name and category_update.name != db_category.name:
        existing_category = db.query(models.Category).filter(
            models.Category.name == category_update.name
        ).first()
        
        if existing_category:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Category name already exists"
            )
    
    # Update category fields
    for field, value in category_update.dict(exclude_unset=True).items():
        setattr(db_category, field, value)
    
    db.commit()
    db.refresh(db_category)
    
    return db_category

@router.delete("/categories/{category_id}")
def delete_category(
    category_id: int,
    db: Session = Depends(get_db),
    admin_user: models.User = Depends(get_current_admin)
):
    db_category = db.query(models.Category).filter(models.Category.id == category_id).first()
    
    if not db_category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Category not found"
        )
    
    db.delete(db_category)
    db.commit()
    
    return {"message": "Category deleted successfully"}