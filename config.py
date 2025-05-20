from connect_for_db import connect

class Config:
    SQLALCHEMY_DATABASE_URI = connect
    SECRET_KEY = 'secretsecretkeyslaptasslaptasraktasraktas'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = 'static/uploads'