from flask_sqlalchemy import SQLAlchemy
from main.database import db


class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(150), nullable=False)
    last_name = db.Column(db.String(150), nullable=False)
    melanoma = db.Column(db.Boolean, nullable=True)
    image_path = db.Column(db.String(300), nullable=True)