from flask import Flask, render_template, redirect, url_for, request, flash
from flask_sqlalchemy import SQLAlchemy
from config import Config
import os
# from main import database
# from main.models import User_models
# from flask_sqlalchemy import SQLAlchemy as sa


import os
from flask import Flask, render_template
from config import Config

def create_app():
    app = Flask(__name__, template_folder='main/templates', static_folder='static')
    app.config.from_object(Config)

    from main.database import db, migrate
    from main.models.User_models import User
    db.init_app(app)
    migrate.init_app(app, db)

    from main.routes import user_routes
    app.register_blueprint(user_routes.main)

    return app

app = create_app()


if __name__ == '__main__':
    app.run(debug=True)
