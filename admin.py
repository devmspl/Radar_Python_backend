from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional


from database import get_db
from dependencies import get_current_admin
import models
import schemas
from schemas import (
    CategoryResponse, CategoryWithSubcategories, CategoryCreate,
    CategoryUpdate, CategoryListResponse, CategoryUpdateWithSubcategories,
    CategoryCreateResponse, SubCategoryResponse
)
from Subcategory_router import CRUDSubCategory
from dependencies import get_current_admin 
router = APIRouter(prefix="/admin", tags=["category"])

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
    # admin_user: models.User = Depends(get_current_admin)
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


@router.get("/{category_id}/with-subcategories", response_model=CategoryWithSubcategories)
def get_category_with_subcategories(
    category_id: str,
    db: Session = Depends(get_db)
):
    """Get a category with all its subcategories"""
    category = category_crud.get_with_subcategories(db, category_id)
    if not category:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Category not found")
    
    # Convert to response model
    response_data = {
        **{k: v for k, v in category.__dict__.items() if not k.startswith('_')},
        "subcategories": category.subcategories,
        "subcategory_count": len(category.subcategories)
    }
    return response_data

@router.post("/with-subcategories", response_model=CategoryCreateResponse, status_code=status.HTTP_201_CREATED)
def create_category_with_subcategories(
    category_in: CategoryCreate,
    subcategories: Optional[List[dict]] = None,
    db: Session = Depends(get_db)
):
    """Create a category with initial subcategories"""
    category_data = category_in.model_dump()
    category = category_crud.create_with_subcategories(db, category_data, subcategories)
    
    return {
        "category": category,
        "subcategories_created": len(subcategories) if subcategories else 0,
        "message": "Category created successfully"
    }

@router.put("/{category_id}/with-subcategories", response_model=CategoryWithSubcategories)
def update_category_with_subcategories(
    category_id: str,
    update_data: CategoryUpdateWithSubcategories,
    db: Session = Depends(get_db)
):
    """Update a category and its subcategories in one operation"""
    category_update = {k: v for k, v in update_data.category_data.model_dump().items() if v is not None}
    
    category = category_crud.update_with_subcategories(
        db,
        category_id,
        category_update,
        update_data.subcategories_to_add,
        update_data.subcategories_to_remove,
        update_data.subcategories_to_update
    )
    
    if not category:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Category not found")
    
    return {
        **{k: v for k, v in category.__dict__.items() if not k.startswith('_')},
        "subcategories": category.subcategories,
        "subcategory_count": len(category.subcategories)
    }

@router.get("/{category_id}/subcategories", response_model=List[SubCategoryResponse])
def get_category_subcategories(
    category_id: str,
    active_only: bool = True,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get all subcategories for a specific category"""
    # Verify category exists
    category = category_crud.get_by_id(db, category_id)
    if not category:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Category not found")
    
    return subcategory_crud.get_by_category(db, category_id, skip, limit, active_only)