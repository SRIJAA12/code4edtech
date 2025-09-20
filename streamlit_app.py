import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from datetime import datetime, timedelta
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
import hashlib
import sqlite3
from pathlib import Path

# Basic PDF processing (fallback)
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# Enhanced page config
st.set_page_config(
    page_title="Resume Analytics Platform",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# COMPREHENSIVE CSS WITH AUTH STYLING
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        --primary-color: #2563eb;
        --primary-light: #eff6ff;
        --primary-dark: #1e40af;
        --secondary-color: #64748b;
        --success-color: #059669;
        --success-light: #ecfdf5;
        --warning-color: #d97706;
        --warning-light: #fffbeb;
        --error-color: #dc2626;
        --error-light: #fef2f2;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --text-muted: #94a3b8;
        --border-color: #e2e8f0;
        --background-primary: #ffffff;
        --background-secondary: #f8fafc;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        --border-radius: 8px;
        --border-radius-lg: 12px;
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
        background-color: var(--background-secondary);
    }
    
    /* LOGIN/SIGNUP STYLES */
    .auth-container {
        max-width: 450px;
        margin: 2rem auto;
        background: var(--background-primary);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius-lg);
        padding: 2.5rem;
        box-shadow: var(--shadow-lg);
    }
    
    .auth-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .auth-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0 0 0.5rem 0;
    }
    
    .auth-subtitle {
        font-size: 1rem;
        color: var(--text-secondary);
        margin: 0;
    }
    
    .demo-credentials {
        background: linear-gradient(135deg, var(--primary-light), #f0f9ff);
        border: 1px solid var(--primary-color);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: var(--shadow-sm);
    }
    
    .demo-title {
        font-weight: 700;
        color: var(--primary-color);
        margin: 0 0 1rem 0;
        font-size: 1.1rem;
    }
    
    .demo-item {
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin: 0.5rem 0;
        padding: 0.25rem 0;
    }
    
    /* DASHBOARD STYLES */
    .dashboard-header {
        background: linear-gradient(135deg, var(--background-primary) 0%, var(--primary-light) 100%);
        border: 1px solid var(--border-color);
        padding: 2.5rem;
        border-radius: var(--border-radius-lg);
        margin-bottom: 2rem;
        box-shadow: var(--shadow-md);
        position: relative;
    }
    
    .dashboard-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-color), #0ea5e9);
    }
    
    .user-welcome {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
        flex-wrap: wrap;
        gap: 1rem;
    }
    
    .welcome-text {
        font-size: 1.8rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .user-info {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .user-avatar {
        width: 45px;
        height: 45px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.3rem;
        box-shadow: var(--shadow-sm);
    }
    
    .user-details {
        display: flex;
        flex-direction: column;
    }
    
    .user-name {
        font-weight: 600;
        color: var(--text-primary);
        font-size: 1rem;
    }
    
    .user-role {
        font-size: 0.875rem;
        color: var(--text-secondary);
    }
    
    /* EXISTING STYLES */
    .professional-header {
        background: linear-gradient(135deg, var(--background-primary) 0%, var(--primary-light) 100%);
        border: 1px solid var(--border-color);
        padding: 2.5rem 2rem;
        border-radius: var(--border-radius-lg);
        margin-bottom: 2rem;
        box-shadow: var(--shadow-md);
        position: relative;
    }
    
    .professional-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-color), #0ea5e9);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0 0 0.5rem 0;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        color: var(--text-secondary);
        margin: 0;
        font-weight: 400;
    }
    
    .card {
        background: var(--background-primary);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius-lg);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-sm);
    }
    
    .card-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border-color);
    }
    
    .metric-card {
        background: var(--background-primary);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius-lg);
        padding: 1.5rem;
        text-align: center;
        box-shadow: var(--shadow-sm);
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-color);
        margin: 0 0 0.25rem 0;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 0.875rem;
        font-weight: 500;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    .status-active {
        background: var(--success-light);
        color: var(--success-color);
        border: 1px solid rgba(5, 150, 105, 0.2);
    }
    
    .status-inactive {
        background: var(--error-light);
        color: var(--error-color);
        border: 1px solid rgba(220, 38, 38, 0.2);
    }
    
    .score-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .score-excellent { background: var(--success-light); color: var(--success-color); }
    .score-good { background: #f0f9ff; color: var(--primary-color); }
    .score-fair { background: var(--warning-light); color: var(--warning-color); }
    .score-poor { background: var(--error-light); color: var(--error-color); }
    
    .ml-prediction {
        background: linear-gradient(135deg, var(--primary-light), #f0f9ff);
        border: 1px solid var(--primary-color);
        border-radius: var(--border-radius);
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .ml-category {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
    }
    
    .skill-tag {
        background: var(--primary-light);
        color: var(--primary-color);
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
        margin: 0.125rem;
        display: inline-block;
        border: 1px solid rgba(37, 99, 235, 0.2);
    }
    
    .skill-matched {
        background: var(--success-light);
        color: var(--success-color);
        border-color: rgba(5, 150, 105, 0.2);
    }
    
    .chart-container {
        background: var(--background-primary);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius-lg);
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-sm);
    }
    
    .stButton > button {
        background: var(--primary-color);
        color: white;
        border: 1px solid var(--primary-color);
        border-radius: var(--border-radius);
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.875rem;
        box-shadow: var(--shadow-sm);
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: var(--primary-dark);
        border-color: var(--primary-dark);
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
    }
    
    /* RESPONSIVE DESIGN */
    @media (max-width: 768px) {
        .auth-container {
            margin: 1rem;
            padding: 2rem;
        }
        
        .user-welcome {
            flex-direction: column;
            text-align: center;
        }
        
        .welcome-text {
            font-size: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# DATABASE SETUP AND USER MANAGEMENT
class UserManager:
    def __init__(self, db_path="users.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the user database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                session_data TEXT,
                analysis_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Create demo users if they don't exist
        demo_users = [
            ("admin", "admin@resumeanalytics.com", "admin123", "Administrator", "admin"),
            ("hr_manager", "hr@company.com", "hr2024", "HR Manager", "hr"),
            ("recruiter", "recruiter@company.com", "recruit123", "Senior Recruiter", "recruiter"),
            ("demo_user", "demo@company.com", "demo123", "Demo User", "user")
        ]
        
        for username, email, password, full_name, role in demo_users:
            try:
                password_hash = self.hash_password(password)
                cursor.execute('''
                    INSERT OR IGNORE INTO users (username, email, password_hash, full_name, role)
                    VALUES (?, ?, ?, ?, ?)
                ''', (username, email, password_hash, full_name, role))
            except:
                pass  # User already exists
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password, password_hash):
        """Verify password against hash"""
        return self.hash_password(password) == password_hash
    
    def create_user(self, username, email, password, full_name, role="user"):
        """Create a new user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            password_hash = self.hash_password(password)
            
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, full_name, role)
                VALUES (?, ?, ?, ?, ?)
            ''', (username, email, password_hash, full_name, role))
            
            conn.commit()
            conn.close()
            return True, "User created successfully"
        
        except sqlite3.IntegrityError as e:
            if "username" in str(e):
                return False, "Username already exists"
            elif "email" in str(e):
                return False, "Email already exists"
            else:
                return False, "User creation failed"
        except Exception as e:
            return False, f"Database error: {str(e)}"
    
    def authenticate_user(self, username, password):
        """Authenticate user login"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, username, email, password_hash, full_name, role, is_active
                FROM users WHERE username = ? OR email = ?
            ''', (username, username))
            
            user = cursor.fetchone()
            
            if user and user[6] and self.verify_password(password, user[3]):  # is_active and password match
                # Update last login
                cursor.execute('''
                    UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
                ''', (user[0],))
                conn.commit()
                
                user_data = {
                    'id': user[0],
                    'username': user[1],
                    'email': user[2],
                    'full_name': user[4],
                    'role': user[5]
                }
                
                conn.close()
                return True, user_data
            
            conn.close()
            return False, "Invalid credentials"
        
        except Exception as e:
            return False, f"Authentication error: {str(e)}"
    
    def get_user_stats(self, user_id):
        """Get user statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT COUNT(*) as session_count, 
                       COALESCE(SUM(analysis_count), 0) as total_analysis
                FROM user_sessions WHERE user_id = ?
            ''', (user_id,))
            
            stats = cursor.fetchone()
            conn.close()
            
            return {
                'session_count': stats[0] if stats else 0,
                'total_analysis': stats[1] if stats else 0
            }
        except:
            return {'session_count': 0, 'total_analysis': 0}

# Initialize User Manager
user_manager = UserManager()

# SESSION STATE MANAGEMENT
def init_session_state():
    """Initialize session state for authentication"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'show_signup' not in st.session_state:
        st.session_state.show_signup = False
    if 'jd_text' not in st.session_state:
        st.session_state.jd_text = ""
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'processing_stats' not in st.session_state:
        st.session_state.processing_stats = {
            'total_processed': 0,
            'successful_analyses': 0,
            'session_start_time': datetime.now()
        }

init_session_state()

# ML MODEL LOADING
@st.cache_resource
def load_ml_model():
    """Load ML model with comprehensive validation"""
    try:
        if os.path.exists('resume_model.joblib'):
            with st.spinner("Loading ML model..."):
                model_data = joblib.load('resume_model.joblib')
            required_keys = ['model', 'vectorizer']
            if all(key in model_data for key in required_keys):
                if 'categories' not in model_data:
                    if hasattr(model_data['model'], 'classes_'):
                        model_data['categories'] = model_data['model'].classes_.tolist()
                    else:
                        model_data['categories'] = ['Unknown Category']
                return model_data
            else:
                return None
        else:
            return None
    except Exception as e:
        return None

ml_model = load_ml_model()

# AUTHENTICATION FUNCTIONS
def show_login_page():
    """Display login page"""
    st.markdown("""
    <div class="auth-container">
        <div class="auth-header">
            <h1 class="auth-title">🔐 Welcome Back</h1>
            <p class="auth-subtitle">Sign in to Resume Analytics Platform</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Demo Credentials
    st.markdown("""
    <div class="demo-credentials">
        <div class="demo-title">🎯 Demo Credentials - Ready to Use</div>
        <div class="demo-item"><strong>👑 Admin:</strong> admin / admin123</div>
        <div class="demo-item"><strong>👥 HR Manager:</strong> hr_manager / hr2024</div>
        <div class="demo-item"><strong>🎯 Recruiter:</strong> recruiter / recruit123</div>
        <div class="demo-item"><strong>👤 Demo User:</strong> demo_user / demo123</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Login Form
    with st.form("login_form", clear_on_submit=False):
        st.markdown("### 📝 Login Credentials")
        
        username = st.text_input(
            "Username or Email",
            placeholder="Enter your username or email",
            help="Use one of the demo credentials above",
            key="login_username"
        )
        
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password",
            help="Use the corresponding demo password",
            key="login_password"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            login_btn = st.form_submit_button("🚀 Sign In", use_container_width=True)
        
        with col2:
            signup_btn = st.form_submit_button("📝 Create Account", use_container_width=True)
    
    if login_btn:
        if username and password:
            with st.spinner("🔐 Authenticating..."):
                success, result = user_manager.authenticate_user(username, password)
                
                if success:
                    st.session_state.authenticated = True
                    st.session_state.user = result
                    st.success(f"✅ Welcome back, {result['full_name']}!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"❌ {result}")
        else:
            st.error("❌ Please enter both username and password")
    
    if signup_btn:
        st.session_state.show_signup = True
        st.rerun()

def show_signup_page():
    """Display signup page"""
    st.markdown("""
    <div class="auth-container">
        <div class="auth-header">
            <h1 class="auth-title">📝 Create Account</h1>
            <p class="auth-subtitle">Join Resume Analytics Platform</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Signup Form
    with st.form("signup_form", clear_on_submit=False):
        st.markdown("### 👤 Account Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            full_name = st.text_input(
                "Full Name *",
                placeholder="Enter your full name",
                key="signup_name"
            )
            
            username = st.text_input(
                "Username *",
                placeholder="Choose a unique username",
                key="signup_username"
            )
        
        with col2:
            email = st.text_input(
                "Email Address *",
                placeholder="Enter your email",
                key="signup_email"
            )
            
            role = st.selectbox(
                "Role *",
                ["user", "recruiter", "hr"],
                format_func=lambda x: {
                    "user": "👤 User",
                    "recruiter": "🎯 Recruiter", 
                    "hr": "👥 HR Manager"
                }[x],
                key="signup_role"
            )
        
        password = st.text_input(
            "Password *",
            type="password",
            placeholder="Create a strong password",
            help="Minimum 6 characters",
            key="signup_password"
        )
        
        confirm_password = st.text_input(
            "Confirm Password *",
            type="password",
            placeholder="Confirm your password",
            key="signup_confirm"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            signup_btn = st.form_submit_button("🚀 Create Account", use_container_width=True)
        
        with col2:
            back_btn = st.form_submit_button("◀️ Back to Login", use_container_width=True)
    
    if signup_btn:
        # Validation
        if not all([full_name, username, email, password, confirm_password]):
            st.error("❌ Please fill in all required fields")
        elif len(password) < 6:
            st.error("❌ Password must be at least 6 characters long")
        elif password != confirm_password:
            st.error("❌ Passwords do not match")
        elif "@" not in email:
            st.error("❌ Please enter a valid email address")
        else:
            with st.spinner("📝 Creating account..."):
                success, message = user_manager.create_user(username, email, password, full_name, role)
                
                if success:
                    st.success(f"✅ {message}")
                    st.info("🔐 You can now login with your credentials")
                    time.sleep(2)
                    st.session_state.show_signup = False
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
    
    if back_btn:
        st.session_state.show_signup = False
        st.rerun()

# CORE PROCESSING FUNCTIONS
def extract_text_simple(uploaded_file):
    """Enhanced text extraction"""
    try:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > 10:
            st.error(f"File too large: {file_size_mb:.1f}MB (max 10MB)")
            return ""
        
        if uploaded_file.type == "text/plain":
            text = str(uploaded_file.read(), "utf-8")
        elif uploaded_file.type == "application/pdf" and PDF_AVAILABLE:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text_parts = []
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text.strip():
                    text_parts.append(page_text)
            text = '\n'.join(text_parts)
        else:
            text = str(uploaded_file.read(), "utf-8", errors="ignore")
        
        if not text or len(text.strip()) < 10:
            st.error("Extracted text is too short or empty")
            return ""
        
        text = re.sub(r'\s+', ' ', text).strip()
        st.success(f"✅ Extracted {len(text)} characters from {uploaded_file.name}")
        return text
        
    except Exception as e:
        st.error(f"❌ Error processing {uploaded_file.name}: {e}")
        return ""

def calculate_similarity(text1, text2):
    """Calculate similarity score"""
    try:
        if not text1 or not text2:
            return 0.0
        
        text1_clean = re.sub(r'[^\w\s]', ' ', text1.lower())
        text2_clean = re.sub(r'[^\w\s]', ' ', text2.lower())
        
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=2000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        tfidf_matrix = vectorizer.fit_transform([text1_clean, text2_clean])
        tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        words1 = set(text1_clean.split())
        words2 = set(text2_clean.split())
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        jaccard_similarity = intersection / union if union > 0 else 0
        
        composite_score = (tfidf_similarity * 0.7) + (jaccard_similarity * 0.3)
        return float(composite_score) * 100
        
    except Exception as e:
        return 0.0

def get_ml_prediction(resume_text):
    """Get ML prediction"""
    if not ml_model:
        return {
            'category': 'ML Model Not Available',
            'confidence': 0.0,
            'confidence_percentage': '0.0%',
            'hiring_recommendation': 'Manual Review Required',
            'top_predictions': [],
            'insights': ['Machine learning model is not loaded'],
            'prediction_successful': False
        }
    
    try:
        resume_clean = resume_text.lower()
        vectorizer = ml_model['vectorizer']
        model = ml_model['model']
        categories = ml_model.get('categories', ['Unknown'])
        
        resume_vector = vectorizer.transform([resume_clean])
        prediction = model.predict(resume_vector)[0]
        probabilities = model.predict_proba(resume_vector)[0]
        confidence = float(max(probabilities))
        
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_predictions = []
        
        for idx in top_indices:
            if idx < len(categories):
                category = categories[idx]
                prob = float(probabilities[idx])
                top_predictions.append({
                    'category': category,
                    'probability': prob,
                    'percentage': f"{prob*100:.1f}%"
                })
        
        if confidence >= 0.9:
            hiring_rec = "Excellent Fit"
            rec_color = "excellent"
        elif confidence >= 0.75:
            hiring_rec = "Strong Candidate"
            rec_color = "good"
        elif confidence >= 0.6:
            hiring_rec = "Good Candidate"
            rec_color = "good"
        elif confidence >= 0.4:
            hiring_rec = "Potential Candidate"
            rec_color = "fair"
        else:
            hiring_rec = "Weak Match"
            rec_color = "poor"
        
        insights = []
        if confidence > 0.85:
            insights.append("🎯 High confidence ML prediction - Very reliable result")
        elif confidence > 0.65:
            insights.append("✅ Good confidence ML prediction - Reliable categorization")
        else:
            insights.append("⚠️ Moderate confidence - Consider manual review")
        
        return {
            'category': prediction,
            'confidence': confidence,
            'confidence_percentage': f"{confidence*100:.1f}%",
            'hiring_recommendation': hiring_rec,
            'hiring_recommendation_color': rec_color,
            'top_predictions': top_predictions,
            'insights': insights,
            'prediction_successful': True
        }
        
    except Exception as e:
        return {
            'category': 'Prediction Error',
            'confidence': 0.0,
            'confidence_percentage': '0.0%',
            'hiring_recommendation': 'Manual Review Required',
            'top_predictions': [],
            'insights': [f'ML prediction failed: {str(e)}'],
            'prediction_successful': False
        }

def extract_skills_advanced(text):
    """Extract skills"""
    skills_database = {
        'Programming Languages': {
            'python': ['python', 'django', 'flask', 'fastapi'],
            'javascript': ['javascript', 'react', 'node', 'vue', 'angular'],
            'java': ['java', 'spring', 'hibernate'],
            'cpp': ['c++', 'cpp'],
            'csharp': ['c#', '.net', 'asp.net'],
            'php': ['php', 'laravel', 'wordpress']
        },
        'Web Technologies': {
            'frontend': ['html', 'css', 'react', 'angular', 'vue', 'bootstrap'],
            'backend': ['api', 'rest', 'graphql', 'microservices'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'redis']
        },
        'Cloud & DevOps': {
            'cloud': ['aws', 'azure', 'gcp', 'google cloud'],
            'containers': ['docker', 'kubernetes', 'k8s'],
            'cicd': ['jenkins', 'gitlab', 'github actions']
        }
    }
    
    found_skills = {}
    text_lower = text.lower()
    
    for category, skill_groups in skills_database.items():
        category_skills = {}
        for skill_name, skill_variants in skill_groups.items():
            confidence_score = sum(text_lower.count(variant) for variant in skill_variants)
            if confidence_score > 0:
                category_skills[skill_name] = {
                    'confidence': min(100, confidence_score * 15),
                    'mentions': confidence_score
                }
        if category_skills:
            found_skills[category] = category_skills
    
    return found_skills

# VISUALIZATION FUNCTIONS
def create_score_distribution_chart(results):
    """Score distribution chart"""
    scores = [r.get('composite_score', 0) for r in results if 'composite_score' in r]
    if not scores:
        return None
    
    fig = px.histogram(
        x=scores,
        nbins=min(15, len(scores)),
        title="Score Distribution Analysis",
        labels={'x': 'Composite Score (%)', 'y': 'Number of Candidates'},
        color_discrete_sequence=['#2563eb']
    )
    
    mean_score = np.mean(scores)
    fig.add_vline(x=mean_score, line_dash="dash", line_color="#059669", 
                  annotation_text=f"Average: {mean_score:.1f}%")
    
    fig.update_layout(height=400, showlegend=False)
    return fig

def create_hiring_recommendation_chart(results):
    """Hiring recommendation chart"""
    recommendations = [r.get('hiring_recommendation', 'Unknown') for r in results if 'hiring_recommendation' in r]
    if not recommendations:
        return None
    
    rec_counts = pd.Series(recommendations).value_counts()
    color_map = {
        'Excellent Fit': '#059669',
        'Strong Candidate': '#2563eb',
        'Good Candidate': '#0ea5e9',
        'Potential Candidate': '#d97706',
        'Weak Match': '#dc2626'
    }
    
    colors = [color_map.get(rec, '#64748b') for rec in rec_counts.index]
    
    fig = go.Figure(data=[
        go.Bar(
            x=rec_counts.index,
            y=rec_counts.values,
            marker_color=colors,
            text=rec_counts.values,
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Hiring Recommendations Distribution",
        xaxis_title="Recommendation Type",
        yaxis_title="Number of Candidates",
        height=400
    )
    
    return fig

# DASHBOARD FUNCTION
def show_dashboard():
    """Display main dashboard"""
    user = st.session_state.user
    user_stats = user_manager.get_user_stats(user['id'])
    
    # Dashboard Header with User Info
    st.markdown(f"""
    <div class="dashboard-header">
        <div class="user-welcome">
            <div class="welcome-text">Welcome back, {user['full_name']}! 👋</div>
            <div class="user-info">
                <div class="user-avatar">{user['full_name'][0].upper()}</div>
                <div class="user-details">
                    <div class="user-name">{user['full_name']}</div>
                    <div class="user-role">{user['role'].title()} • {user['email']}</div>
                </div>
            </div>
        </div>
        <div style="display: flex; gap: 2rem; margin-top: 1.5rem; flex-wrap: wrap;">
            <div>
                <div style="font-size: 1.5rem; font-weight: 700; color: var(--primary-color);">{user_stats['total_analysis']}</div>
                <div style="font-size: 0.875rem; color: var(--text-secondary); font-weight: 500;">Total Analyses</div>
            </div>
            <div>
                <div style="font-size: 1.5rem; font-weight: 700; color: var(--primary-color);">{user_stats['session_count']}</div>
                <div style="font-size: 0.875rem; color: var(--text-secondary); font-weight: 500;">Sessions</div>
            </div>
            <div>
                <div style="font-size: 1.5rem; font-weight: 700; color: {'var(--success-color)' if ml_model else 'var(--error-color)'};">{'✅ Active' if ml_model else '❌ Basic'}</div>
                <div style="font-size: 0.875rem; color: var(--text-secondary); font-weight: 500;">ML Status</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main Dashboard Content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Job Description Section
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="card-title">📋 Job Description</h3>', unsafe_allow_html=True)
        
        jd_method = st.radio("Input method:", ["Paste Text", "Upload File"], label_visibility="collapsed")
        
        if jd_method == "Paste Text":
            jd_text = st.text_area(
                "Job Description Content:",
                height=200,
                placeholder="Enter the complete job description here...",
                help="Provide detailed job description for accurate matching"
            )
            
            if jd_text:
                word_count = len(jd_text.split())
                char_count = len(jd_text)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Words", word_count)
                with col_b:
                    st.metric("Characters", char_count)
                
                if word_count < 50:
                    st.warning("⚠️ Job description seems brief. Add more details for better analysis.")
                else:
                    st.success("✅ Good job description provided")
                
                st.session_state.jd_text = jd_text
        
        else:
            jd_file = st.file_uploader("Upload job description file:", type=['txt', 'pdf'])
            if jd_file:
                jd_text = extract_text_simple(jd_file)
                if jd_text:
                    st.session_state.jd_text = jd_text
                    
                    with st.expander("📄 Preview extracted content"):
                        st.text_area("Content", jd_text[:500] + "..." if len(jd_text) > 500 else jd_text, height=150, disabled=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Resume Upload Section
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="card-title">📄 Resume Files</h3>', unsafe_allow_html=True)
        
        resume_files = st.file_uploader(
            "Select resume files:",
            type=['txt', 'pdf'],
            accept_multiple_files=True,
            help="Upload multiple resume files for batch processing"
        )
        
        if resume_files:
            total_size = sum(file.size for file in resume_files)
            total_size_mb = total_size / (1024 * 1024)
            
            file_col1, file_col2 = st.columns(2)
            with file_col1:
                st.metric("Files", len(resume_files))
            with file_col2:
                st.metric("Total Size", f"{total_size_mb:.1f} MB")
            
            with st.expander("📁 File Details"):
                for i, file in enumerate(resume_files, 1):
                    file_size_kb = file.size / 1024
                    st.write(f"{i}. **{file.name}** ({file_size_kb:.1f} KB)")
        
        # Analysis Button
        if resume_files and st.session_state.jd_text:
            if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                
                start_time = time.time()
                results = []
                progress_container = st.container()
                
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    successful_count = 0
                    
                    for i, resume_file in enumerate(resume_files):
                        progress_bar.progress((i + 1) / len(resume_files))
                        status_text.text(f"Processing: {resume_file.name} ({i+1}/{len(resume_files)})")
                        
                        resume_text = extract_text_simple(resume_file)
                        
                        if resume_text:
                            similarity_score = calculate_similarity(resume_text, st.session_state.jd_text)
                            ml_prediction = get_ml_prediction(resume_text)
                            resume_skills = extract_skills_advanced(resume_text)
                            
                            # Extract candidate name
                            lines = resume_text.split('\n')[:15]
                            candidate_name = "Unknown Candidate"
                            
                            for line in lines:
                                line = line.strip()
                                if (5 < len(line) < 50 and 
                                    line.count(' ') <= 3 and
                                    not any(char.isdigit() for char in line) and
                                    '@' not in line and
                                    'http' not in line.lower()):
                                    candidate_name = line.title()
                                    break
                            
                            # Calculate composite score
                            composite_score = similarity_score
                            if ml_model and ml_prediction['prediction_successful']:
                                composite_score = (similarity_score * 0.6) + (ml_prediction['confidence'] * 100 * 0.4)
                            
                            result = {
                                'filename': resume_file.name,
                                'candidate_name': candidate_name,
                                'similarity_score': round(similarity_score, 1),
                                'composite_score': round(composite_score, 1),
                                'ml_category': ml_prediction['category'],
                                'ml_confidence': ml_prediction['confidence'],
                                'ml_confidence_percentage': ml_prediction['confidence_percentage'],
                                'hiring_recommendation': ml_prediction['hiring_recommendation'],
                                'hiring_recommendation_color': ml_prediction.get('hiring_recommendation_color', 'poor'),
                                'resume_skills': resume_skills,
                                'ml_insights': ml_prediction.get('insights', []),
                                'word_count': len(resume_text.split()),
                                'char_count': len(resume_text),
                                'file_size_kb': resume_file.size / 1024,
                                'ml_prediction_successful': ml_prediction['prediction_successful'],
                                'resume_preview': resume_text[:800] + "..." if len(resume_text) > 800 else resume_text
                            }
                            
                            results.append(result)
                            successful_count += 1
                    
                    # Sort and rank
                    results.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
                    for i, result in enumerate(results):
                        result['ranking'] = i + 1
                    
                    st.session_state.results = results
                    total_time = time.time() - start_time
                    
                    # Update statistics
                    st.session_state.processing_stats.update({
                        'total_processed': len(resume_files),
                        'successful_analyses': successful_count,
                        'total_processing_time': total_time
                    })
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Analysis completed successfully!")
                    
                    st.balloons()
                    st.success(f"""
                    **🎉 Analysis Summary**
                    - **Total files:** {len(resume_files)}
                    - **Successful:** {successful_count}
                    - **Processing time:** {total_time:.1f}s
                    - **Average per file:** {total_time/len(resume_files):.1f}s
                    """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Results Section
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="card-title">📊 Analysis Results & Analytics</h3>', unsafe_allow_html=True)
        
        if st.session_state.results:
            scores = [r.get('composite_score', 0) for r in st.session_state.results if 'composite_score' in r]
            
            # Summary Metrics
            met_col1, met_col2, met_col3, met_col4 = st.columns(4)
            
            with met_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(st.session_state.results)}</div>
                    <div class="metric-label">Total Processed</div>
                </div>
                """, unsafe_allow_html=True)
            
            with met_col2:
                avg_score = np.mean(scores) if scores else 0
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{avg_score:.1f}%</div>
                    <div class="metric-label">Average Score</div>
                </div>
                """, unsafe_allow_html=True)
            
            with met_col3:
                top_score = max(scores) if scores else 0
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{top_score:.1f}%</div>
                    <div class="metric-label">Highest Score</div>
                </div>
                """, unsafe_allow_html=True)
            
            with met_col4:
                high_performers = len([s for s in scores if s >= 80])
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{high_performers}</div>
                    <div class="metric-label">High Performers</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Essential Charts
            if len(scores) > 1:
                st.markdown("### 📈 Analytics Dashboard")
                
                charts_data = {
                    'Score Distribution': create_score_distribution_chart(st.session_state.results),
                    'Hiring Recommendations': create_hiring_recommendation_chart(st.session_state.results)
                }
                
                available_charts = {name: chart for name, chart in charts_data.items() if chart is not None}
                
                if available_charts:
                    chart_tabs = st.tabs(list(available_charts.keys()))
                    
                    for i, (chart_name, chart) in enumerate(available_charts.items()):
                        with chart_tabs[i]:
                            st.markdown(f'<div class="chart-container">', unsafe_allow_html=True)
                            st.plotly_chart(chart, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
            
            # Filtering Options
            st.markdown("### 🔍 Filter Results")
            
            filter_col1, filter_col2 = st.columns(2)
            
            with filter_col1:
                score_filter = st.selectbox(
                    "Filter by Score:",
                    ["All Scores", "Excellent (≥80%)", "Good (60-79%)", "Fair (40-59%)", "Poor (<40%)"]
                )
            
            with filter_col2:
                if ml_model:
                    rec_filter = st.selectbox(
                        "Filter by Recommendation:",
                        ["All", "Excellent Fit", "Strong Candidate", "Good Candidate", "Potential Candidate", "Weak Match"]
                    )
                else:
                    rec_filter = "All"
            
            # Apply filters
            filtered_results = st.session_state.results.copy()
            
            if score_filter != "All Scores":
                if "Excellent" in score_filter:
                    filtered_results = [r for r in filtered_results if r.get('composite_score', 0) >= 80]
                elif "Good" in score_filter:
                    filtered_results = [r for r in filtered_results if 60 <= r.get('composite_score', 0) < 80]
                elif "Fair" in score_filter:
                    filtered_results = [r for r in filtered_results if 40 <= r.get('composite_score', 0) < 60]
                elif "Poor" in score_filter:
                    filtered_results = [r for r in filtered_results if r.get('composite_score', 0) < 40]
            
            if rec_filter != "All":
                filtered_results = [r for r in filtered_results if r.get('hiring_recommendation') == rec_filter]
            
            st.info(f"📊 Displaying {len(filtered_results)} of {len(st.session_state.results)} results")
            
            # Top Results Display
            st.markdown("### 🏆 Top Candidates")
            
            for result in filtered_results[:8]:  # Show top 8
                score = result.get('composite_score', 0)
                recommendation = result.get('hiring_recommendation', 'Unknown')
                rec_color = result.get('hiring_recommendation_color', 'poor')
                
                # Determine score icon
                if score >= 80:
                    score_icon = "🏆"
                elif score >= 60:
                    score_icon = "✅"
                elif score >= 40:
                    score_icon = "⚠️"
                else:
                    score_icon = "❌"
                
                with st.expander(
                    f"{score_icon} #{result.get('ranking', '?')} - {result.get('candidate_name', 'Unknown')} - {score:.1f}%",
                    expanded=(result.get('ranking', 99) <= 3)
                ):
                    # Header information
                    info_col1, info_col2, info_col3 = st.columns(3)
                    
                    with info_col1:
                        st.markdown("**📋 Basic Information**")
                        st.write(f"**File:** {result.get('filename', 'Unknown')}")
                        st.write(f"**Candidate:** {result.get('candidate_name', 'Unknown')}")
                        st.write(f"**File Size:** {result.get('file_size_kb', 0):.1f} KB")
                    
                    with info_col2:
                        st.markdown("**📊 Performance Scores**")
                        st.write(f"**Composite Score:** {score:.1f}%")
                        st.write(f"**Similarity Score:** {result.get('similarity_score', 0):.1f}%")
                        st.write(f"**Words:** {result.get('word_count', 0):,}")
                        
                        # Visual score indicator
                        score_class = "score-excellent" if score >= 80 else "score-good" if score >= 60 else "score-fair" if score >= 40 else "score-poor"
                        st.markdown(f'<div class="score-badge {score_class}">{score:.1f}%</div>', unsafe_allow_html=True)
                    
                    with info_col3:
                        st.markdown("**🤖 ML Prediction**")
                        
                        if result.get('ml_prediction_successful', False):
                            st.markdown(f"""
                            <div class="ml-prediction">
                                <div class="ml-category">{result.get('ml_category', 'Unknown')}</div>
                                <div style="font-size: 0.9rem; color: var(--text-secondary);">Confidence: {result.get('ml_confidence_percentage', '0%')}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f'<div class="score-badge score-{rec_color}">{recommendation}</div>', unsafe_allow_html=True)
                        else:
                            st.error("ML prediction failed")
                    
                    # Detailed Analysis Tabs
                    detail_tabs = st.tabs(["Skills Analysis", "ML Insights", "Resume Content"])
                    
                    with detail_tabs[0]:
                        st.markdown("#### 🔧 Skills Breakdown")
                        
                        skills_data = result.get('resume_skills', {})
                        
                        if skills_data:
                            for category, skills in skills_data.items():
                                st.markdown(f"**{category}:**")
                                
                                skills_html = ""
                                for skill_name, skill_info in skills.items():
                                    confidence = skill_info.get('confidence', 0)
                                    mentions = skill_info.get('mentions', 0)
                                    
                                    skills_html += f'<span class="skill-tag">{skill_name.title()} ({confidence:.0f}%)</span> '
                                
                                st.markdown(skills_html, unsafe_allow_html=True)
                                st.markdown("")  # Spacing
                        else:
                            st.info("No categorized skills detected")
                    
                    with detail_tabs[1]:
                        st.markdown("#### 💡 ML Analysis Insights")
                        
                        if result.get('ml_prediction_successful', False):
                            insights = result.get('ml_insights', [])
                            if insights:
                                for i, insight in enumerate(insights, 1):
                                    st.markdown(f"**{i}.** {insight}")
                            
                            # Top ML Predictions
                            top_predictions = result.get('top_ml_predictions', [])
                            if top_predictions:
                                st.markdown("#### 🎯 Top Category Predictions")
                                
                                for i, pred in enumerate(top_predictions, 1):
                                    category = pred['category']
                                    percentage = pred['percentage']
                                    probability = pred['probability']
                                    
                                    st.write(f"**{i}. {category}** - {percentage}")
                                    st.progress(probability)
                        else:
                            st.info("ML insights not available - prediction failed")
                    
                    with detail_tabs[2]:
                        st.markdown("#### 📄 Resume Content Analysis")
                        
                        # File Statistics
                        word_count = result.get('word_count', 0)
                        char_count = result.get('char_count', 0)
                        
                        stat_col1, stat_col2 = st.columns(2)
                        
                        with stat_col1:
                            st.metric("Words", f"{word_count:,}")
                        with stat_col2:
                            st.metric("Characters", f"{char_count:,}")
                        
                        # Resume Preview
                        preview_text = result.get('resume_preview', '')
                        if preview_text:
                            st.markdown("**Resume Content Preview:**")
                            st.text_area(
                                "Content Preview:",
                                preview_text,
                                height=200,
                                disabled=True,
                                key=f"preview_{result.get('ranking', 'unknown')}"
                            )
            
            # Export Section
            st.markdown("### 📤 Export Options")
            
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                if st.button("📊 Export Detailed Report", use_container_width=True):
                    csv_data = []
                    for result in st.session_state.results:
                        csv_data.append({
                            'Rank': result.get('ranking', 0),
                            'Candidate_Name': result.get('candidate_name', ''),
                            'Filename': result.get('filename', ''),
                            'Composite_Score': result.get('composite_score', 0),
                            'Similarity_Score': result.get('similarity_score', 0),
                            'ML_Category': result.get('ml_category', ''),
                            'ML_Confidence': result.get('ml_confidence_percentage', ''),
                            'Hiring_Recommendation': result.get('hiring_recommendation', ''),
                            'Word_Count': result.get('word_count', 0)
                        })
                    
                    df = pd.DataFrame(csv_data)
                    csv_string = df.to_csv(index=False)
                    
                    st.download_button(
                        label="💾 Download CSV Report",
                        data=csv_string,
                        file_name=f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with export_col2:
                if st.button("📋 Export Executive Summary", use_container_width=True):
                    summary = f"""# Executive Summary - Resume Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analyzed by:** {user['full_name']} ({user['role'].title()})

## Analysis Overview
- **Total Candidates:** {len(st.session_state.results)}
- **Average Score:** {np.mean(scores):.1f}%
- **ML Model Status:** {'Active' if ml_model else 'Not Available'}

## Top 5 Candidates

"""
                    
                    for i, result in enumerate(st.session_state.results[:5], 1):
                        summary += f"""### {i}. {result.get('candidate_name', 'Unknown')} - {result.get('composite_score', 0):.1f}%
- **ML Category:** {result.get('ml_category', 'Unknown')}
- **ML Confidence:** {result.get('ml_confidence_percentage', '0%')}
- **Recommendation:** {result.get('hiring_recommendation', 'Unknown')}

"""
                    
                    st.download_button(
                        label="📋 Download Summary",
                        data=summary,
                        file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
        
        else:
            # Empty State
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: var(--background-secondary); border-radius: var(--border-radius-lg); margin: 2rem 0;">
                <h3 style="color: var(--text-secondary); margin-bottom: 1rem;">Ready for Analysis</h3>
                <p style="color: var(--text-muted);">Upload job description and resume files to begin comprehensive analysis</p>
                <div style="margin-top: 1rem;">
                    <span style="color: var(--primary-color);">✓ Smart Matching</span> • 
                    <span style="color: var(--primary-color);">✓ ML Predictions</span> • 
                    <span style="color: var(--primary-color);">✓ Advanced Analytics</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Enhanced Sidebar with user controls
with st.sidebar:
    if st.session_state.authenticated:
        user = st.session_state.user
        
        st.markdown("### 👤 User Account")
        st.markdown(f"""
        **Logged in as:** {user['full_name']}  
        **Role:** {user['role'].title()}  
        **Email:** {user['email']}
        """)
        
        if st.button("🔓 Logout", use_container_width=True, type="secondary"):
            # Clear all session state
            st.session_state.authenticated = False
            st.session_state.user = None
            st.session_state.jd_text = ""
            st.session_state.results = []
            st.session_state.show_signup = False
            
            st.success("✅ Logged out successfully!")
            time.sleep(1)
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("### 🤖 ML System Status")
        
        if ml_model:
            st.markdown('<div class="status-badge status-active">✅ ML System Active</div>', unsafe_allow_html=True)
            categories = ml_model.get('categories', [])
            st.info(f"**Categories Available:** {len(categories)}")
            st.info(f"**Predictions:** Enabled")
            
            with st.expander("📋 Available Categories"):
                for i, cat in enumerate(categories[:12], 1):
                    st.write(f"{i}. {cat}")
                if len(categories) > 12:
                    st.write(f"... and {len(categories)-12} more")
        
        else:
            st.markdown('<div class="status-badge status-inactive">❌ Basic Mode</div>', unsafe_allow_html=True)
            st.warning("**ML predictions unavailable**")
            st.info("Place 'resume_model.joblib' in project folder")
            
            if st.button("🔄 Retry Model Loading", use_container_width=True):
                st.cache_resource.clear()
                st.rerun()
        
        st.markdown("---")
        
        st.markdown("### 📊 Session Analytics")
        
        stats = st.session_state.processing_stats
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Processed", stats['total_processed'])
        with col2:
            st.metric("Successful", stats['successful_analyses'])
        
        # Session duration
        session_duration = datetime.now() - stats['session_start_time']
        hours, remainder = divmod(session_duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        st.metric("Session Time", f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        
        st.markdown("### 🛠️ Quick Actions")
        
        if st.button("🔄 Refresh System", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
        
        if st.button("🗑️ Clear Analysis Data", use_container_width=True):
            st.session_state.jd_text = ""
            st.session_state.results = []
            st.session_state.processing_stats = {
                'total_processed': 0,
                'successful_analyses': 0,
                'session_start_time': datetime.now()
            }
            st.success("✅ Analysis data cleared successfully")
            st.rerun()

# MAIN APPLICATION LOGIC
def main():
    """Main application logic"""
    
    # Check authentication status
    if not st.session_state.authenticated:
        # Show login or signup page
        if st.session_state.get('show_signup', False):
            show_signup_page()
        else:
            show_login_page()
    else:
        # Show main dashboard
        show_dashboard()

if __name__ == "__main__":
    main()

# Footer
if st.session_state.authenticated:
    st.markdown("""
    <div style="background: var(--background-primary); border: 1px solid var(--border-color); border-radius: var(--border-radius-lg); padding: 2rem; margin-top: 2rem; text-align: center;">
        <h3 style="margin-bottom: 0.5rem; color: var(--text-primary);">Resume Analytics Platform</h3>
        <p style="color: var(--text-secondary); margin: 0;">Secure recruitment analytics with user authentication</p>
    </div>
    """, unsafe_allow_html=True)
