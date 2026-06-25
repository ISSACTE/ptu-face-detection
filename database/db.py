import sqlite3
import os
from flask import g, current_app

DATABASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database', 'ptu_students.db')

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE, detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
    return g.db

def init_db():
    os.makedirs(os.path.dirname(DATABASE), exist_ok=True)
    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
    db.executescript('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            student_id TEXT UNIQUE NOT NULL,
            major TEXT NOT NULL,
            semester TEXT NOT NULL,
            roll_number TEXT NOT NULL,
            face_encodings BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS face_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            image_data TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students(id)
        );
    ''')
    db.commit()
    db.close()
