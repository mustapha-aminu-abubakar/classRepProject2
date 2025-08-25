from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()
DB_URL = "sqlite:///detections.db"
engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)

class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True)
    filename = Column(String)
    image_url = Column(String)
    num_detections = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)

def log_detection(filename, image_url, num_detections):
    session = Session()
    record = Detection(
        filename=filename,
        image_url=image_url,
        num_detections=num_detections
    )
    session.add(record)
    session.commit()
    session.close()
