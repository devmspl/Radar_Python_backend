from datetime import datetime
import uuid
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_

from database import get_db
from models import SubCategory, Category
from schemas import (
    SubCategoryCreate, SubCategoryUpdate, SubCategoryResponse,
    SubCategoryListResponse, BulkSubCategoryCreate,CategoriesListResponse
)
from dependencies import get_current_admin
import models

class CRUDSubCategory:
    def get_by_id(self, db: Session, subcategory_id: int) -> Optional[SubCategory]:
        """Get subcategory by integer ID"""
        return db.query(SubCategory).filter(SubCategory.id == subcategory_id).first()
    
    def get_by_uuid(self, db: Session, subcategory_uuid: str) -> Optional[SubCategory]:
        """Get subcategory by UUID"""
        return db.query(SubCategory).filter(SubCategory.uuid == subcategory_uuid).first()
    
    def get_by_name(self, db: Session, name: str) -> Optional[SubCategory]:
        """Get subcategory by name"""
        return db.query(SubCategory).filter(SubCategory.name == name).first()
    
    def get_by_category(self, db: Session, category_id: int, skip: int = 0, limit: int = 100, active_only: bool = True) -> List[SubCategory]:
        """Get all subcategories for a specific category"""
        query = db.query(SubCategory).filter(SubCategory.category_id == category_id)
        if active_only:
            query = query.filter(SubCategory.is_active == True)
        return query.offset(skip).limit(limit).all()
    
    def get_all(self, db: Session, skip: int = 0, limit: int = 100, active_only: bool = True) -> List[SubCategory]:
        """Get all subcategories"""
        query = db.query(SubCategory)
        if active_only:
            query = query.filter(SubCategory.is_active == True)
        return query.offset(skip).limit(limit).all()
    
    def search(self, db: Session, search_term: str, skip: int = 0, limit: int = 100) -> List[SubCategory]:
        """Search subcategories by name or description"""
        return db.query(SubCategory).filter(
            or_(
                SubCategory.name.ilike(f"%{search_term}%"),
                SubCategory.description.ilike(f"%{search_term}%")
            )
        ).offset(skip).limit(limit).all()
    
    def create(self, db: Session, subcategory_data: Dict[str, Any]) -> SubCategory:
        """Create a new subcategory"""
        # Ensure category_id is integer
        try:
            category_id = int(subcategory_data["category_id"])
        except (ValueError, TypeError):
            raise ValueError("category_id must be a valid integer")
        
        # Check if category exists
        category = db.query(Category).filter(Category.id == category_id).first()
        if not category:
            raise ValueError("Category not found")
        
        # Check if subcategory with same name exists in this category
        existing = db.query(SubCategory).filter(
            SubCategory.name == subcategory_data["name"],
            SubCategory.category_id == category_id
        ).first()
        
        if existing:
            raise ValueError(f"Subcategory '{subcategory_data['name']}' already exists in this category")
        
        # Create subcategory
        subcategory = SubCategory(
            name=subcategory_data["name"],
            description=subcategory_data.get("description", ""),
            category_id=category_id,
            is_active=subcategory_data.get("is_active", True)
        )
        
        try:
            db.add(subcategory)
            db.commit()
            db.refresh(subcategory)
            return subcategory
        except Exception as e:
            db.rollback()
            raise ValueError(f"Database error: {str(e)}")
    
    def update(self, db: Session, subcategory_id: int, update_data: Dict[str, Any]) -> Optional[SubCategory]:
        """Update an existing subcategory"""
        subcategory = self.get_by_id(db, subcategory_id)
        if not subcategory:
            return None
        
        # If changing category_id, verify new category exists
        if "category_id" in update_data and update_data["category_id"] != subcategory.category_id:
            try:
                new_category_id = int(update_data["category_id"])
            except (ValueError, TypeError):
                raise ValueError("category_id must be a valid integer")
            
            new_category = db.query(Category).filter(Category.id == new_category_id).first()
            if not new_category:
                raise ValueError("New category not found")
            
            # Check if subcategory with same name exists in new category
            existing = db.query(SubCategory).filter(
                SubCategory.name == update_data.get("name", subcategory.name),
                SubCategory.category_id == new_category_id,
                SubCategory.id != subcategory_id
            ).first()
            
            if existing:
                raise ValueError(f"Subcategory with name '{update_data.get('name', subcategory.name)}' already exists in the new category")
        
        # If changing name, check for duplicates in same category
        if "name" in update_data and update_data["name"] != subcategory.name:
            category_id = update_data.get("category_id", subcategory.category_id)
            existing = db.query(SubCategory).filter(
                SubCategory.name == update_data["name"],
                SubCategory.category_id == category_id,
                SubCategory.id != subcategory_id
            ).first()
            
            if existing:
                raise ValueError(f"Subcategory with name '{update_data['name']}' already exists in this category")
        
        # Update fields
        for key, value in update_data.items():
            if hasattr(subcategory, key) and value is not None:
                setattr(subcategory, key, value)
        
        subcategory.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(subcategory)
        return subcategory
    
    def delete(self, db: Session, subcategory_id: int) -> bool:
        """Delete a subcategory"""
        subcategory = self.get_by_id(db, subcategory_id)
        if not subcategory:
            return False
        
        db.delete(subcategory)
        db.commit()
        return True
    
    def deactivate(self, db: Session, subcategory_id: int) -> Optional[SubCategory]:
        """Deactivate a subcategory"""
        subcategory = self.get_by_id(db, subcategory_id)
        if not subcategory:
            return None
        
        subcategory.is_active = False
        subcategory.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(subcategory)
        return subcategory
    
    def activate(self, db: Session, subcategory_id: int) -> Optional[SubCategory]:
        """Activate a subcategory"""
        subcategory = self.get_by_id(db, subcategory_id)
        if not subcategory:
            return None
        
        subcategory.is_active = True
        subcategory.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(subcategory)
        return subcategory
    
    def get_count_by_category(self, db: Session, category_id: int, active_only: bool = True) -> int:
        """Get count of subcategories for a category"""
        query = db.query(SubCategory).filter(SubCategory.category_id == category_id)
        if active_only:
            query = query.filter(SubCategory.is_active == True)
        return query.count()
    
    def get_multi(self, db: Session, skip: int = 0, limit: int = 100) -> List[SubCategory]:
        """Get multiple subcategories (alias for get_all)"""
        return self.get_all(db, skip, limit)


# Create instance
crud_subcategory = CRUDSubCategory()

router = APIRouter(
    prefix="/subcategories",
    tags=["subcategories"]
)


@router.post("/", response_model=SubCategoryResponse, status_code=status.HTTP_201_CREATED)
def create_subcategory(
    subcategory_in: SubCategoryCreate,
    db: Session = Depends(get_db)
):
    """Create a new subcategory"""
    try:
        # Convert Pydantic model to dict
        subcategory_data = subcategory_in.model_dump()
        
        # Ensure category_id is integer (handles string inputs from JSON)
        if 'category_id' in subcategory_data:
            try:
                subcategory_data['category_id'] = int(subcategory_data['category_id'])
            except (ValueError, TypeError):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="category_id must be a valid integer"
                )
        
        subcategory = crud_subcategory.create(db, subcategory_data)
        return subcategory
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/bulk", response_model=dict, status_code=status.HTTP_201_CREATED)
def create_bulk_subcategories(
    bulk_data: BulkSubCategoryCreate,
    db: Session = Depends(get_db)
):
    """Create multiple subcategories at once"""
    created = 0
    failed = []
    
    for subcat_data in bulk_data.subcategories:
        try:
            # Convert Pydantic model to dict and ensure category_id is integer
            subcat_dict = subcat_data.model_dump()
            if 'category_id' in subcat_dict:
                try:
                    subcat_dict['category_id'] = int(subcat_dict['category_id'])
                except (ValueError, TypeError):
                    raise ValueError("category_id must be a valid integer")
            
            crud_subcategory.create(db, subcat_dict)
            created += 1
        except Exception as e:
            failed.append({
                "name": subcat_data.name,
                "category_id": subcat_data.category_id,
                "error": str(e)
            })
    
    return {
        "message": f"Created {created} subcategories",
        "created": created,
        "failed": failed,
        "total": len(bulk_data.subcategories)
    }
# Place the search endpoint BEFORE parameterized routes
@router.get("/search", response_model=SubCategoryListResponse)
def search_subcategories(
    q: str = Query(..., min_length=1, description="Search query"),
    category_id: Optional[int] = Query(None, description="Filter by category ID"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    include_inactive: bool = False,
    sort_by: Optional[str] = Query(None, description="Sort by field: name, created_at, category_name, etc."),
    sort_order: str = Query("asc", description="Sort order: asc or desc"),
    db: Session = Depends(get_db)
):
    """Search subcategories by name or description with advanced filtering"""
    try:
        from sqlalchemy import case
        
        # Base query
        query = db.query(SubCategory)
        
        # Filter by active status if needed
        if not include_inactive:
            query = query.filter(SubCategory.is_active == True)
        
        # Filter by category if specified
        if category_id:
            query = query.filter(SubCategory.category_id == category_id)
        
        # Search functionality
        search_term = f"%{q}%"
        query = query.filter(
            or_(
                SubCategory.name.ilike(search_term),
                SubCategory.description.ilike(search_term)
            )
        )
        
        # Sorting
        if sort_by:
            if sort_by == "name":
                sort_field = SubCategory.name
            elif sort_by == "created_at":
                sort_field = SubCategory.created_at
            elif sort_by == "updated_at":
                sort_field = SubCategory.updated_at
            elif sort_by == "category_name":
                # Sort by category name - need to join
                query = query.join(Category)
                sort_field = Category.name
            else:
                sort_field = SubCategory.created_at
            
            if sort_order.lower() == "desc":
                query = query.order_by(sort_field.desc())
            else:
                query = query.order_by(sort_field.asc())
        else:
            # SQLite compatible default sorting
            query = query.order_by(
                case(
                    (SubCategory.name.like(f"{q}%"), 0),  # Exact start matches
                    (SubCategory.name.like(f"%{q}%"), 1),  # Contains
                    else_=2
                ),
                SubCategory.name.asc(),
                SubCategory.created_at.desc()
            )
        
        # Get total count before pagination
        total = query.count()
        
        # Apply pagination
        subcategories = query.offset(skip).limit(limit).all()
        
        # Calculate if there are more results
        has_more = skip + limit < total
        
        # Calculate page number
        page = (skip // limit) + 1 if limit > 0 else 1
        
        return {
            "subcategories": subcategories,
            "total": total,
            "page": page,
            "limit": limit,
            "has_more": has_more,
            "search_query": q,
            "category_id": category_id
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching subcategories: {str(e)}"
        )

@router.get("/", response_model=SubCategoryListResponse)
def get_subcategories(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    category_id: Optional[int] = None,
    active_only: bool = True,
    search: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get all subcategories with filtering options"""
    try:
        if search:
            subcategories = crud_subcategory.search(db, search, skip, limit)
            total = len(subcategories)
        elif category_id is not None:
            subcategories = crud_subcategory.get_by_category(db, category_id, skip, limit, active_only)
            total = crud_subcategory.get_count_by_category(db, category_id, active_only)
        else:
            subcategories = crud_subcategory.get_all(db, skip, limit, active_only)
            total = len(crud_subcategory.get_all(db, 0, 1000000, active_only))  # Get total count
        
        return {
            "subcategories": subcategories,
            "total": total,
            "page": (skip // limit) + 1 if limit > 0 else 1,
            "limit": limit,
            "has_more": skip + limit < total
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching subcategories: {str(e)}"
        )


@router.get("/{subcategory_id}", response_model=SubCategoryResponse)
def get_subcategory(
    subcategory_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific subcategory by integer ID"""
    subcategory = crud_subcategory.get_by_id(db, subcategory_id)
    if not subcategory:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Subcategory not found")
    return subcategory


@router.get("/uuid/{subcategory_uuid}", response_model=SubCategoryResponse)
def get_subcategory_by_uuid(
    subcategory_uuid: str,
    db: Session = Depends(get_db)
):
    """Get a specific subcategory by UUID"""
    subcategory = crud_subcategory.get_by_uuid(db, subcategory_uuid)
    if not subcategory:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Subcategory not found")
    return subcategory


@router.put("/{subcategory_id}", response_model=SubCategoryResponse)
def update_subcategory(
    subcategory_id: int,
    subcategory_in: SubCategoryUpdate,
    db: Session = Depends(get_db)
):
    """Update a subcategory"""
    # Filter out None values
    update_data = {k: v for k, v in subcategory_in.model_dump().items() if v is not None}
    
    # Handle category_id conversion if provided
    if 'category_id' in update_data:
        try:
            update_data['category_id'] = int(update_data['category_id'])
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="category_id must be a valid integer"
            )
    
    try:
        subcategory = crud_subcategory.update(db, subcategory_id, update_data)
        if not subcategory:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Subcategory not found")
        return subcategory
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating subcategory: {str(e)}"
        )


@router.delete("/{subcategory_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_subcategory(
    subcategory_id: int,
    db: Session = Depends(get_db)
):
    """Delete a subcategory"""
    success = crud_subcategory.delete(db, subcategory_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Subcategory not found")
    return None  # 204 No Content


@router.patch("/{subcategory_id}/activate", response_model=SubCategoryResponse)
def activate_subcategory(
    subcategory_id: int,
    db: Session = Depends(get_db)
):
    """Activate a subcategory"""
    subcategory = crud_subcategory.activate(db, subcategory_id)
    if not subcategory:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Subcategory not found")
    return subcategory


@router.patch("/{subcategory_id}/deactivate", response_model=SubCategoryResponse)
def deactivate_subcategory(
    subcategory_id: int,
    db: Session = Depends(get_db)
):
    """Deactivate a subcategory"""
    subcategory = crud_subcategory.deactivate(db, subcategory_id)
    if not subcategory:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Subcategory not found")
    return subcategory


@router.get("/category/{category_id}/count")
def get_subcategory_count_by_category(
    category_id: int,
    active_only: bool = True,
    db: Session = Depends(get_db)
):
    """Get count of subcategories for a specific category"""
    try:
        count = crud_subcategory.get_count_by_category(db, category_id, active_only)
        return {
            "category_id": category_id,
            "count": count,
            "active_only": active_only
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error counting subcategories: {str(e)}"
        )


@router.get("/check-name/{name}")
def check_subcategory_name(
    name: str,
    category_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Check if a subcategory name exists (optionally within a specific category)"""
    try:
        query = db.query(SubCategory).filter(SubCategory.name == name)
        
        if category_id:
            query = query.filter(SubCategory.category_id == category_id)
            existing = query.first()
            return {
                "name": name,
                "category_id": category_id,
                "exists": existing is not None,
                "subcategory": existing if existing else None
            }
        else:
            existing = query.all()
            return {
                "name": name,
                "exists": len(existing) > 0,
                "count": len(existing),
                "subcategories": existing
            }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking subcategory name: {str(e)}"
        )