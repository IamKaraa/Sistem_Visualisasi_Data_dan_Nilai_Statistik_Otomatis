from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default='user')

class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nama_dataset = db.Column(db.String(150))
    date_column = db.Column(db.String(100))
    filename = db.Column(db.String(200))
    path_file = db.Column(db.Text)
    sheet_name = db.Column(db.String(100), nullable=True)
    total_rows = db.Column(db.Integer)
    total_columns = db.Column(db.Integer)
    size_mb = db.Column(db.Float)
    owner_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

class AnalysisResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'))
    column_name = db.Column(db.String(100))
    analysis_mode = db.Column(db.String(20), nullable=False)
    context = db.Column(db.String(20))
    mean = db.Column(db.Float)
    median = db.Column(db.Float)
    std = db.Column(db.Float)
    min = db.Column(db.Float)
    max = db.Column(db.Float)
    iqr = db.Column(db.Float)
    count = db.Column(db.Integer)
    anomaly_count = db.Column(db.Integer)
    trend = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=db.func.now())
