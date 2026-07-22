from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import text
import json
from app import models
from app.api import get_db

router = APIRouter(prefix="/v1/datasets", tags=["datasets"])

@router.get("/products")
def get_all_products(skip: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=100), db: Session = Depends(get_db)):
    query = text("""
        SELECT 
            p.parent_asin,
            p.title,
            p.price,
            (
                SELECT JSON_ARRAYAGG(category_name) 
                FROM product_categories 
                WHERE parent_asin = p.parent_asin
            ) as categories,
            (
                SELECT JSON_ARRAYAGG(JSON_OBJECT('image_url', image_url, 'variant', variant)) 
                FROM product_images 
                WHERE parent_asin = p.parent_asin
            ) as images,
            (
                SELECT JSON_ARRAYAGG(description_text) 
                FROM product_descriptions 
                WHERE parent_asin = p.parent_asin
            ) as descriptions
        FROM products p
        LIMIT :limit OFFSET :skip
    """)
    
    resultados_crudos = db.execute(query, {"limit": limit, "skip": skip}).mappings().all()
    
    productos_procesados = []
    
    for fila in resultados_crudos:
        producto = dict(fila)
        
        producto['categories'] = json.loads(producto['categories']) if producto['categories'] else []
        producto['images'] = json.loads(producto['images']) if producto['images'] else []
        producto['descriptions'] = json.loads(producto['descriptions']) if producto['descriptions'] else []
        
        productos_procesados.append(producto)

    return {"items": productos_procesados, "skip": skip, "limit": limit}

@router.get("/products/{parent_asin}")
def get_single_product(parent_asin: str, db: Session = Depends(get_db)):
    product = db.query(models.Product).filter(models.Product.parent_asin == parent_asin).first()
    if not product:
        raise HTTPException(status_code=404, detail="Producto no encontrado")
    return product

@router.get("/products/{parent_asin}/reviews")
def get_product_reviews(parent_asin: str, skip: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200), db: Session = Depends(get_db)):
    product = db.query(models.Product).filter(models.Product.parent_asin == parent_asin).first()
    if not product:
        raise HTTPException(status_code=404, detail="Producto no encontrado")
    
    reviews = db.query(models.Review).filter(models.Review.parent_asin == parent_asin).offset(skip).limit(limit).all()
    return {"parent_asin": parent_asin, "items": reviews, "skip": skip, "limit": limit}