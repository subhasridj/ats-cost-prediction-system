# db.py
from sqlalchemy import create_engine, text
import os

DB_URL = os.getenv('DATABASE_URL', 'sqlite:///./ats.db')
engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if 'sqlite' in DB_URL else {})

def execute_sql(sql):
    with engine.connect() as conn:
        conn.execute(text(sql))
        conn.commit()
