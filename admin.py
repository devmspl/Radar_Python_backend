from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_,case
from typing import List, Optional


from database import get_db
from dependencies import get_current_admin
import models
import schemas
from schemas import (
    CategoryResponse, CategoryWithSubcategories, CategoryCreate,
    CategoryUpdate, CategoryListResponse, CategoryUpdateWithSubcategories,
    CategoryCreateResponse, SubCategoryResponse, CategoriesListResponse
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

@router.get("/categories", response_model=schemas.CategoryListResponse)
def get_categories(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    search: Optional[str] = None,
    include_inactive: bool = False,
    sort_by: Optional[str] = Query(None, description="Sort by field: name, created_at, etc."),
    sort_order: str = Query("asc", description="Sort order: asc or desc"),
    db: Session = Depends(get_db),
    admin_user: models.User = Depends(get_current_admin)
):
    """Get all categories with filtering, searching, and pagination options"""
    try:
        from sqlalchemy import or_, func
        
        # Base query for categories
        query = db.query(models.Category)
        
        # Filter by active status if needed
        if not include_inactive:
            query = query.filter(models.Category.is_active == True)
        
        # Search functionality
        if search:
            search_term = f"%{search}%"
            query = query.filter(
                or_(
                    models.Category.name.ilike(search_term),
                    models.Category.description.ilike(search_term),
                    models.Category.note.ilike(search_term),
                    models.Category.admin_note.ilike(search_term)
                )
            )
        
        # Sorting
        if sort_by:
            if sort_by == "name":
                sort_field = models.Category.name
            elif sort_by == "created_at":
                sort_field = models.Category.created_at
            elif sort_by == "updated_at":
                sort_field = models.Category.updated_at
            else:
                sort_field = models.Category.created_at
            
            if sort_order.lower() == "desc":
                query = query.order_by(sort_field.desc())
            else:
                query = query.order_by(sort_field.asc())
        else:
            # Default sorting by created_at descending
            query = query.order_by(models.Category.created_at.desc())
        
        # Get total count before pagination
        total = query.count()
        
        # Apply pagination
        categories = query.offset(skip).limit(limit).all()
        
        # Calculate if there are more results
        has_more = skip + limit < total
        
        # Calculate page number
        page = (skip // limit) + 1 if limit > 0 else 1
        
        # Calculate total subcategories across all returned categories
        total_subcategories = 0
        if categories:
            # Get all category IDs
            category_ids = [category.id for category in categories]
            
            # Query total subcategories for these categories
            # Assuming you have a SubCategory model with category_id field
            # and is_active field for filtering
            from models import SubCategory
            
            subcategory_query = db.query(func.count(SubCategory.id)).filter(
                SubCategory.category_id.in_(category_ids)
            )
            
            if not include_inactive:
                subcategory_query = subcategory_query.filter(SubCategory.is_active == True)
            
            total_subcategories = subcategory_query.scalar() or 0
        
        return {
            "categories": categories,
            "total": total,
            "page": page,
            "limit": limit,
            "has_more": has_more,
            "total_subcategories": total_subcategories  # This is now included
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching categories: {str(e)}" 
        )

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


@router.get("/categories/search", response_model=CategoryListResponse)
def search_categories(
    q: str = Query(..., min_length=1, description="Search query"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    include_inactive: bool = False,
    sort_by: Optional[str] = Query(None, description="Sort by field: name, created_at, etc."),
    sort_order: str = Query("asc", description="Sort order: asc or desc"),
    db: Session = Depends(get_db),
    admin_user: models.User = Depends(get_current_admin)
):
    """Search categories by name, description, note, or admin_note"""
    try:
        from sqlalchemy import func
        
        # Base query for categories
        query = db.query(models.Category)
        
        # Filter by active status if needed
        if not include_inactive:
            query = query.filter(models.Category.is_active == True)
        
        # Search functionality
        search_term = f"%{q}%"
        query = query.filter(
            or_(
                models.Category.name.ilike(search_term),
                models.Category.description.ilike(search_term),
                models.Category.note.ilike(search_term),
                models.Category.admin_note.ilike(search_term)
            )
        )
        
        # Sorting
        if sort_by:
            if sort_by == "name":
                sort_field = models.Category.name
            elif sort_by == "created_at":
                sort_field = models.Category.created_at
            elif sort_by == "updated_at":
                sort_field = models.Category.updated_at
            else:
                sort_field = models.Category.created_at
            
            if sort_order.lower() == "desc":
                query = query.order_by(sort_field.desc())
            else:
                query = query.order_by(sort_field.asc())
        else:
            # SQLite compatible sorting (removed similarity function)
            query = query.order_by(
                case(
                    (models.Category.name.like(f"{q}%"), 0),  # Exact start matches
                    (models.Category.name.like(f"%{q}%"), 1),  # Contains
                    else_=2
                ),
                models.Category.name.asc(),
                models.Category.created_at.desc()
            )
        
        # Get total count before pagination
        total = query.count()
        
        # Apply pagination
        categories = query.offset(skip).limit(limit).all()
        
        # Calculate if there are more results
        has_more = skip + limit < total
        
        # Calculate page number
        page = (skip // limit) + 1 if limit > 0 else 1
        
        # Calculate total subcategories across all returned categories
        total_subcategories = 0
        if categories:
            # Get all category IDs
            category_ids = [category.id for category in categories]
            
            # Query total subcategories for these categories
            from models import SubCategory
            
            subcategory_query = db.query(func.count(SubCategory.id)).filter(
                SubCategory.category_id.in_(category_ids)
            )
            
            if not include_inactive:
                subcategory_query = subcategory_query.filter(SubCategory.is_active == True)
            
            total_subcategories = subcategory_query.scalar() or 0
        
        return {
            "categories": categories,
            "total": total,
            "page": page,
            "limit": limit,
            "has_more": has_more,
            "total_subcategories": total_subcategories,
            "search_query": q
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching categories: {str(e)}"
        )