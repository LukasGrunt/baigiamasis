from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

db = SQLAlchemy()
migrate = Migrate()
#---------------------------------------------------------------
#install migrate
#---------------------------------------------------------------

# flask db init
# flask db migrate
# flask db upgrade
# pip install mysqlclient