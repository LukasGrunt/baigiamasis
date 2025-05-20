from flask import Flask, render_template, redirect, url_for, request, flash, Blueprint, current_app
from flask_sqlalchemy import SQLAlchemy

from main.models.User_models import User
from main.database import db
from werkzeug.utils import secure_filename
import os
from main.models.User_models import User

# login_manager = LoginManager()
app = Flask(__name__)
main = Blueprint("main", __name__)

@main.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    filename = None
    message = None
    full_name = ""

    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        image_file = request.files['image']

        if image_file:
            filename = secure_filename(image_file.filename)

            
            upload_path = os.path.join(current_app.root_path, current_app.config['UPLOAD_FOLDER'])
            os.makedirs(upload_path, exist_ok=True)

            filepath = os.path.join(upload_path, filename)
            image_file.save(filepath)

            
            from data.preprocessing import Website_foto
            from tensorflow.keras.models import load_model

            model = load_model('best.h5')
            image = Website_foto(filepath)
            prediction = model.predict(image)[0][0]
            probability = round(prediction * 100, 2)
            full_name = f"{first_name} {last_name}"


            from main.models.User_models import User
            result_value = 1 if prediction > 0.5 else 0
            image_path = os.path.join(current_app.root_path, current_app.config['UPLOAD_FOLDER'], filename)
            user = User(
                first_name=first_name,
                last_name=last_name,
                melanoma=result_value,
                image_path=image_path
            )
            db.session.add(user)
            db.session.commit()

            db.session.add(user)
            db.session.commit()

            message = "Greičiausiai tai yra melanoma." if prediction > 0.5 else "Greičiausiai tai nėra melanoma."

    return render_template(
        'main/index.html',
        probability=probability,
        message=message,
        image_filename=filename,
        name=full_name
    )