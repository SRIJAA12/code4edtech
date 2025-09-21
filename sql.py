import sqlite3
import hashlib

DB_NAME = "demo_users.db"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    # Create users table if not exists
    cur.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def add_user(username, password, role):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    try:
        cur.execute('INSERT INTO users (username, password, role) VALUES (?, ?, ?)',
                    (username, hash_password(password), role))
        conn.commit()
        print(f"User '{username}' added with role '{role}'.")
    except sqlite3.IntegrityError:
        print(f"Error: User '{username}' already exists.")
    conn.close()

def list_users():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute('SELECT id, username, role FROM users')
    users = cur.fetchall()
    conn.close()
    print("Users in database:")
    for user in users:
        print(f"ID: {user[0]}, Username: {user[1]}, Role: {user[2]}")

if __name__ == "__main__":
    init_db()
    # Add demo users
    add_user('student', 'student123', 'student')
    add_user('recruiter', 'recruit123', 'recruiter')
    add_user('admin', 'admin123', 'admin')
    # List all users
    list_users()
