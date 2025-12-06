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

class CRUDSubCategory:
    def get(self, db: Session, subcategory_id: str) -> Optional[SubCategory]:
        return db.query(SubCategory).filter(SubCategory.id == subcategory_id).first()
    
    def get_by_category(self, db: Session, category_id: str) -> List[SubCategory]:
        return db.query(SubCategory).filter(SubCategory.category_id == category_id).all()
    
    def get_by_name(self, db: Session, name: str) -> Optional[SubCategory]:
        return db.query(SubCategory).filter(SubCategory.name == name).first()
    
    def get_multi(self, db: Session, skip: int = 0, limit: int = 100) -> List[SubCategory]:
        return db.query(SubCategory).offset(skip).limit(limit).all()
    
    def create(self, db: Session, obj_in: SubCategoryCreate) -> SubCategory:
        db_obj = SubCategory(
            id=str(uuid.uuid4()),
            name=obj_in.name,
            description=obj_in.description,
            category_id=obj_in.category_id
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj
    
    def update(self, db: Session, subcategory_id: str, obj_in: SubCategoryUpdate) -> Optional[SubCategory]:
        db_obj = self.get(db, subcategory_id)
        if not db_obj:
            return None
        
        update_data = obj_in.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_obj, field, value)
        
        db.commit()
        db.refresh(db_obj)
        return db_obj
    
    def delete(self, db: Session, subcategory_id: str) -> Optional[SubCategory]:
        db_obj = self.get(db, subcategory_id)
        if not db_obj:
            return None
        
        db.delete(db_obj)
        db.commit()
        return db_obj
    def get_with_subcategories(self, db: Session, category_id: str) -> Optional[Category]:
        return db.query(Category).filter(Category.id == category_id).first()
    
    def create_with_subcategories(self, db: Session, category_data: Dict[str, Any], subcategories: List[Dict[str, Any]] = None) -> Category:
        # Create category
        category = Category(
            id=str(uuid.uuid4()),
            uuid=str(uuid.uuid4()),
            name=category_data["name"],
            description=category_data.get("description"),
            note=category_data.get("note"),
            admin_note=category_data.get("admin_note"),
            admin_id=category_data.get("admin_id"),
            is_active=category_data.get("is_active", True)
        )
        
        db.add(category)
        db.flush()  # Flush to get category ID
        
        # Create subcategories if provided
        if subcategories:
            for subcat_data in subcategories:
                subcategory = SubCategory(
                    id=str(uuid.uuid4()),
                    uuid=str(uuid.uuid4()),
                    name=subcat_data["name"],
                    description=subcat_data.get("description"),
                    category_id=category.id,
                    is_active=subcat_data.get("is_active", True)
                )
                db.add(subcategory)
        
        db.commit()
        db.refresh(category)
        return category
    
    def update_with_subcategories(self, db: Session, category_id: str, 
                                  category_update: Dict[str, Any],
                                  subcategories_to_add: List[Dict[str, Any]] = None,
                                  subcategories_to_remove: List[str] = None,
                                  subcategories_to_update: List[Dict[str, Any]] = None) -> Optional[Category]:
        
        category = self.get_by_id(db, category_id)
        if not category:
            return None
        
        # Update category fields
        for key, value in category_update.items():
            if hasattr(category, key) and value is not None:
                setattr(category, key, value)
        
        # Remove subcategories
        if subcategories_to_remove:
            for subcat_id in subcategories_to_remove:
                subcategory = db.query(SubCategory).filter(SubCategory.id == subcat_id).first()
                if subcategory and subcategory.category_id == category_id:
                    db.delete(subcategory)
        
        # Update subcategories
        if subcategories_to_update:
            for subcat_update in subcategories_to_update:
                subcat_id = subcat_update.get("id")
                if subcat_id:
                    subcategory = db.query(SubCategory).filter(SubCategory.id == subcat_id).first()
                    if subcategory and subcategory.category_id == category_id:
                        for key, value in subcat_update.items():
                            if hasattr(subcategory, key) and key != "id":
                                setattr(subcategory, key, value)
        
        # Add new subcategories
        if subcategories_to_add:
            for subcat_data in subcategories_to_add:
                # Check if subcategory already exists
                existing = db.query(SubCategory).filter(
                    SubCategory.name == subcat_data["name"],
                    SubCategory.category_id == category_id
                ).first()
                
                if not existing:
                    subcategory = SubCategory(
                        id=str(uuid.uuid4()),
                        uuid=str(uuid.uuid4()),
                        name=subcat_data["name"],
                        description=subcat_data.get("description"),
                        category_id=category_id,
                        is_active=subcat_data.get("is_active", True)
                    )
                    db.add(subcategory)
        
        category.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(category)
        return category
    
    def get_category_stats(self, db: Session) -> Dict[str, Any]:
        total_categories = db.query(Category).count()
        active_categories = db.query(Category).filter(Category.is_active == True).count()
        total_subcategories = db.query(SubCategory).count()
        active_subcategories = db.query(SubCategory).filter(SubCategory.is_active == True).count()
        
        return {
            "total_categories": total_categories,
            "active_categories": active_categories,
            "total_subcategories": total_subcategories,
            "active_subcategories": active_subcategories
        }

subcategory = CRUDSubCategory()

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
        subcategory = crud_subcategory.create(db, subcategory_data)
        return subcategory
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

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
            crud_subcategory.create(db, subcat_data.model_dump())
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
        subcategories = crud_subcategory.search(db, search, skip, limit)
        total = len(subcategories)  # Simplified count
    elif category_id:
        subcategories = crud_subcategory.get_by_category(db, category_id, skip, limit)
        total = crud_subcategory.get_count_by_category(db, category_id, active_only)
    else:
        subcategories = crud_subcategory.get_all(db, skip, limit, active_only)
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
    subcategory_id: str,
    subcategory_in: SubCategoryUpdate,
    db: Session = Depends(get_db)
):
    """Update a subcategory"""
    update_data = {k: v for k, v in subcategory_in.model_dump().items() if v is not None}
    
    try:
        subcategory = crud_subcategory.update(db, subcategory_id, update_data)
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
    success = crud_subcategory.delete(db, subcategory_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Subcategory not found")

@router.patch("/{subcategory_id}/activate", response_model=SubCategoryResponse)
def activate_subcategory(
    subcategory_id: str,
    db: Session = Depends(get_db)
):
    """Activate a subcategory"""
    subcategory = crud_subcategory.activate(db, subcategory_id)
    if not subcategory:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Subcategory not found")
    return subcategory

@router.patch("/{subcategory_id}/deactivate", response_model=SubCategoryResponse)
def deactivate_subcategory(
    subcategory_id: str,
    db: Session = Depends(get_db)
):
    """Deactivate a subcategory"""
    subcategory = crud_subcategory.deactivate(db, subcategory_id)
    if not subcategory:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Subcategory not found")
    return subcategory