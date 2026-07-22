from sqlalchemy import Column, Integer, String, Text, Numeric, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base

class ProductFeature(Base):
    __tablename__ = "product_features"
    id = Column(Integer, primary_key=True, index=True)
    parent_asin = Column(String(20), ForeignKey("products.parent_asin", ondelete="CASCADE"))
    feature_text = Column(Text)
    
    product = relationship("Product", back_populates="features")

class ProductDescription(Base):
    __tablename__ = "product_descriptions"
    id = Column(Integer, primary_key=True, index=True)
    parent_asin = Column(String(20), ForeignKey("products.parent_asin", ondelete="CASCADE"))
    description_text = Column(Text)
    
    product = relationship("Product", back_populates="descriptions")

class ProductImage(Base):
    __tablename__ = "product_images"
    id = Column(Integer, primary_key=True, index=True)
    parent_asin = Column(String(20), ForeignKey("products.parent_asin", ondelete="CASCADE"))
    image_url = Column(Text)
    variant = Column(String(100))
    
    product = relationship("Product", back_populates="images")

class ProductVideo(Base):
    __tablename__ = "product_videos"
    id = Column(Integer, primary_key=True, index=True)
    parent_asin = Column(String(20), ForeignKey("products.parent_asin", ondelete="CASCADE"))
    title = Column(Text)
    video_url = Column(Text)
    user_id = Column(String(255))
    
    product = relationship("Product", back_populates="videos")

class ProductCategory(Base):
    __tablename__ = "product_categories"
    id = Column(Integer, primary_key=True, index=True)
    parent_asin = Column(String(20), ForeignKey("products.parent_asin", ondelete="CASCADE"))
    category_name = Column(String(255))
    
    product = relationship("Product", back_populates="categories")

class ProductDetail(Base):
    __tablename__ = "product_details"
    id = Column(Integer, primary_key=True, index=True)
    parent_asin = Column(String(20), ForeignKey("products.parent_asin", ondelete="CASCADE"))
    detail_key = Column(String(255))
    detail_value = Column(Text)
    
    product = relationship("Product", back_populates="details")

class ReviewImage(Base):
    __tablename__ = "review_images"
    id = Column(Integer, primary_key=True, index=True)
    review_id = Column(Integer, ForeignKey("reviews.id", ondelete="CASCADE"))
    small_image_url = Column(Text)
    medium_image_url = Column(Text)
    large_image_url = Column(Text)
    attachment_type = Column(String(100))
    
    review = relationship("Review", back_populates="images")

class Product(Base):
    __tablename__ = "products"

    parent_asin = Column(String(20), primary_key=True, index=True)
    main_category = Column(String(150))
    title = Column(Text)
    average_rating = Column(Numeric(3, 2))
    rating_number = Column(Integer)
    price = Column(Numeric(10, 2))
    store = Column(String(255))
    numeric_price = Column(Numeric(10, 2))
    review_count = Column(Integer)

    reviews = relationship("Review", back_populates="product", cascade="all, delete-orphan")
    features = relationship("ProductFeature", back_populates="product", cascade="all, delete-orphan")
    descriptions = relationship("ProductDescription", back_populates="product", cascade="all, delete-orphan")
    images = relationship("ProductImage", back_populates="product", cascade="all, delete-orphan")
    videos = relationship("ProductVideo", back_populates="product", cascade="all, delete-orphan")
    categories = relationship("ProductCategory", back_populates="product", cascade="all, delete-orphan")
    details = relationship("ProductDetail", back_populates="product", cascade="all, delete-orphan")

class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True)
    rating = Column(Integer)
    title = Column(Text)
    text = Column(Text)
    asin = Column(String(20))
    parent_asin = Column(String(20), ForeignKey("products.parent_asin"))
    user_id = Column(String(100))
    timestamp = Column(String(50))
    helpful_vote = Column(Integer)
    verified_purchase = Column(Boolean)

    product = relationship("Product", back_populates="reviews")
    images = relationship("ReviewImage", back_populates="review", cascade="all, delete-orphan")