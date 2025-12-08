from sqlalchemy.orm import Session
from datetime import datetime
from models import SubCategory, Category
from schemas import SubCategoryCreate, SubCategoryUpdate
import uuid
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from database import get_db
from schemas import (
    SubCategoryCreate, SubCategoryUpdate, SubCategoryResponse,
    SubCategoryListResponse, BulkSubCategoryCreate
)
from sqlalchemy import or_

class CRUDSubCategory:
    def get_by_id(self, db: Session, subcategory_id: str) -> Optional[SubCategory]:
        return db.query(SubCategory).filter(SubCategory.id == subcategory_id).first()
    
    def get_by_uuid(self, db: Session, subcategory_uuid: str) -> Optional[SubCategory]:
        return db.query(SubCategory).filter(SubCategory.uuid == subcategory_uuid).first()
    
    def get_by_name(self, db: Session, name: str) -> Optional[SubCategory]:
        return db.query(SubCategory).filter(SubCategory.name == name).first()
    
    def get_by_category(self, db: Session, category_id: str, skip: int = 0, limit: int = 100, active_only: bool = True) -> List[SubCategory]:
        query = db.query(SubCategory).filter(SubCategory.category_id == category_id)
        if active_only:
            query = query.filter(SubCategory.is_active == True)
        return query.offset(skip).limit(limit).all()
    
    def get_all(self, db: Session, skip: int = 0, limit: int = 100, active_only: bool = True) -> List[SubCategory]:
        query = db.query(SubCategory)
        if active_only:
            query = query.filter(SubCategory.is_active == True)
        return query.offset(skip).limit(limit).all()
    
    def search(self, db: Session, search_term: str, skip: int = 0, limit: int = 100) -> List[SubCategory]:
        return db.query(SubCategory).filter(
            or_(
                SubCategory.name.ilike(f"%{search_term}%"),
                SubCategory.description.ilike(f"%{search_term}%")
            )
        ).offset(skip).limit(limit).all()
    
    def create(self, db: Session, subcategory_data: Dict[str, Any]) -> SubCategory:
        # Check if category exists
        category = db.query(Category).filter(Category.id == subcategory_data["category_id"]).first()
        if not category:
            raise ValueError("Category not found")
        
        # Check if subcategory with same name exists in this category
        existing = db.query(SubCategory).filter(
            SubCategory.name == subcategory_data["name"],
            SubCategory.category_id == subcategory_data["category_id"]
        ).first()
        
        if existing:
            raise ValueError(f"Subcategory '{subcategory_data['name']}' already exists in this category")
        
        # Create subcategory with explicit field assignment
        # Make sure we're using the correct types
        subcategory = SubCategory(
            id=str(uuid.uuid4()),  # This should be String in your model
            uuid=str(uuid.uuid4()),
            name=subcategory_data["name"],
            description=subcategory_data.get("description", ""),  # Provide default
            category_id=str(subcategory_data["category_id"]),  # Ensure it's string
            is_active=subcategory_data.get("is_active", True)
        )
        
        # Debug: Print what we're trying to insert
        print(f"Inserting subcategory: {subcategory.__dict__}")
        
        try:
            db.add(subcategory)
            db.commit()
            db.refresh(subcategory)
            return subcategory
        except Exception as e:
            db.rollback()
            # Get more details about the error
            print(f"Database error: {e}")
            raise ValueError(f"Database error: {str(e)}")
    

    
    def update(self, db: Session, subcategory_id: str, update_data: Dict[str, Any]) -> Optional[SubCategory]:
        subcategory = self.get_by_id(db, subcategory_id)
        if not subcategory:
            return None
        
        # If changing category_id, verify new category exists
        if "category_id" in update_data and update_data["category_id"] != subcategory.category_id:
            new_category = db.query(Category).filter(Category.id == update_data["category_id"]).first()
            if not new_category:
                raise ValueError("New category not found")
            
            # Check if subcategory with same name exists in new category
            existing = db.query(SubCategory).filter(
                SubCategory.name == update_data.get("name", subcategory.name),
                SubCategory.category_id == update_data["category_id"],
                SubCategory.id != subcategory_id
            ).first()
            
            if existing:
                raise ValueError(f"Subcategory with name '{update_data.get('name', subcategory.name)}' already exists in the new category")
        
        # If changing name, check for duplicates in same category
        if "name" in update_data and update_data["name"] != subcategory.name:
            existing = db.query(SubCategory).filter(
                SubCategory.name == update_data["name"],
                SubCategory.category_id == update_data.get("category_id", subcategory.category_id),
                SubCategory.id != subcategory_id
            ).first()
            
            if existing:
                raise ValueError(f"Subcategory with name '{update_data['name']}' already exists in this category")
        
        for key, value in update_data.items():
            if hasattr(subcategory, key):
                setattr(subcategory, key, value)
        
        subcategory.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(subcategory)
        return subcategory
    
    def delete(self, db: Session, subcategory_id: str) -> bool:
        subcategory = self.get_by_id(db, subcategory_id)
        if not subcategory:
            return False
        
        db.delete(subcategory)
        db.commit()
        return True
    
    def deactivate(self, db: Session, subcategory_id: str) -> Optional[SubCategory]:
        subcategory = self.get_by_id(db, subcategory_id)
        if not subcategory:
            return None
        
        subcategory.is_active = False
        subcategory.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(subcategory)
        return subcategory
    
    def activate(self, db: Session, subcategory_id: str) -> Optional[SubCategory]:
        subcategory = self.get_by_id(db, subcategory_id)
        if not subcategory:
            return None
        
        subcategory.is_active = True
        subcategory.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(subcategory)
        return subcategory
    
    def get_count_by_category(self, db: Session, category_id: str, active_only: bool = True) -> int:
        query = db.query(SubCategory).filter(SubCategory.category_id == category_id)
        if active_only:
            query = query.filter(SubCategory.is_active == True)
        return query.count()
    
    # The following methods are from your old CRUD class but don't seem to be used in the router
    def get_multi(self, db: Session, skip: int = 0, limit: int = 100) -> List[SubCategory]:
        return self.get_all(db, skip, limit)

# Create instance - IMPORTANT: variable name must match what the router uses
crud_subcategory = CRUDSubCategory()  # Changed from 'subcategory' to 'crud_subcategory'

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
        subcategory_data = subcategory_in.model_dump()
        
        # Convert category_id to string if it's not already
        if 'category_id' in subcategory_data:
            subcategory_data['category_id'] = str(subcategory_data['category_id'])
        
        subcategory = crud_subcategory.create(db, subcategory_data)
        return subcategory
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        # Add more detailed logging
        print(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Internal server error: {str(e)}"
        )
@router.get("/debug/schema")
def debug_schema(db: Session = Depends(get_db)):
    """Debug endpoint to check table schema"""
    from sqlalchemy import inspect
    
    inspector = inspect(db.get_bind())
    columns = inspector.get_columns('subcategories')
    
    schema_info = []
    for column in columns:
        schema_info.append({
            'name': column['name'],
            'type': str(column['type']),
            'nullable': column['nullable'],
            'default': column.get('default'),
            'autoincrement': column.get('autoincrement', False)
        })
    
    return {"schema": schema_info}
    
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
            crud_subcategory.create(db, subcat_data.model_dump())  # Now matches variable name
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

@router.get("/", response_model=SubCategoryListResponse)
def get_subcategories(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    category_id: Optional[str] = None,
    active_only: bool = True,
    search: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get all subcategories with filtering options"""
    if search:
        subcategories = crud_subcategory.search(db, search, skip, limit)  # Now matches variable name
        total = len(subcategories)  # Simplified count
    elif category_id:
        subcategories = crud_subcategory.get_by_category(db, category_id, skip, limit, active_only)  # Added active_only
        total = crud_subcategory.get_count_by_category(db, category_id, active_only)
    else:
        subcategories = crud_subcategory.get_all(db, skip, limit, active_only)  # Now matches variable name
        total = len(subcategories)  # Simplified count
    
    return {
        "subcategories": subcategories,
        "total": total,
        "page": (skip // limit) + 1,
        "limit": limit,
        "has_more": skip + limit < total
    }

@router.get("/{subcategory_id}", response_model=SubCategoryResponse)
def get_subcategory(
    subcategory_id: str,
    db: Session = Depends(get_db)
):
    """Get a specific subcategory by ID"""
    subcategory = crud_subcategory.get_by_id(db, subcategory_id)  # Now matches variable name
    if not subcategory:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Subcategory not found")
    return subcategory

@router.get("/uuid/{subcategory_uuid}", response_model=SubCategoryResponse)
def get_subcategory_by_uuid(
    subcategory_uuid: str,
    db: Session = Depends(get_db)
):
    """Get a specific subcategory by UUID"""
    subcategory = crud_subcategory.get_by_uuid(db, subcategory_uuid)  # Now matches variable name
    if not subcategory:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Subcategory not found")
    return subcategory

@router.put("/{subcategory_id}", response_model=SubCategoryResponse)
def update_subcategory(
    subcategory_id: str,
    subcategory_in: SubCategoryUpdate,
    db: Session = Depends(get_db)
):
    """Update a subcategory"""
    update_data = {k: v for k, v in subcategory_in.model_dump().items() if v is not None}
    
    try:
        subcategory = crud_subcategory.update(db, subcategory_id, update_data)  # Now matches variable name
        if not subcategory:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Subcategory not found")
        return subcategory
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@router.delete("/{subcategory_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_subcategory(
    subcategory_id: str,
    db: Session = Depends(get_db)
):
    """Delete a subcategory"""
    success = crud_subcategory.delete(db, subcategory_id)  # Now matches variable name
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Subcategory not found")
    return None  # 204 No Content

@router.patch("/{subcategory_id}/activate", response_model=SubCategoryResponse)
def activate_subcategory(
    subcategory_id: str,
    db: Session = Depends(get_db)
):
    """Activate a subcategory"""
    subcategory = crud_subcategory.activate(db, subcategory_id)  # Now matches variable name
    if not subcategory:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Subcategory not found")
    return subcategory

@router.patch("/{subcategory_id}/deactivate", response_model=SubCategoryResponse)
def deactivate_subcategory(
    subcategory_id: str,
    db: Session = Depends(get_db)
):
    """Deactivate a subcategory"""
    subcategory = crud_subcategory.deactivate(db, subcategory_id)  # Now matches variable name
    if not subcategory:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Subcategory not found")
    return subcategory