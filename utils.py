import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from datetime import datetime
import sqlite3
import hashlib
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Tuple
import random

# PDF processing
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Google Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

class UserManager:
    """Handles user authentication and database operations"""
    
    def __init__(self, db_path: str = "users.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
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
        
        # User sessions table
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
        
        # Chat history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                message TEXT,
                response TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Create demo users
        demo_users = [
            ("admin", "admin@resumeanalytics.com", "admin123", "System Administrator", "admin"),
            ("hr_manager", "hr@company.com", "hr2024", "HR Manager", "recruiter"),
            ("recruiter", "recruiter@company.com", "recruit123", "Senior Recruiter", "recruiter"),
            ("demo_user", "demo@company.com", "demo123", "Demo User", "recruiter"),
            ("student", "student@university.edu", "student123", "John Student", "student"),
            ("jane_student", "jane@university.edu", "jane123", "Jane Doe", "student")
        ]
        
        for username, email, password, full_name, role in demo_users:
            try:
                password_hash = self.hash_password(password)
                cursor.execute('''
                    INSERT OR IGNORE INTO users (username, email, password_hash, full_name, role)
                    VALUES (?, ?, ?, ?, ?)
                ''', (username, email, password_hash, full_name, role))
            except Exception:
                pass
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        return self.hash_password(password) == password_hash
    
    def create_user(self, username: str, email: str, password: str, full_name: str, role: str = "user") -> Tuple[bool, str]:
        """Create new user account"""
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
            return True, "Account created successfully"
        
        except sqlite3.IntegrityError as e:
            if "username" in str(e):
                return False, "Username already exists"
            elif "email" in str(e):
                return False, "Email already exists"
            else:
                return False, "Account creation failed"
        except Exception as e:
            return False, f"Database error: {str(e)}"
    
    def authenticate_user(self, username: str, password: str) -> Tuple[bool, Any]:
        """Authenticate user login"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, username, email, password_hash, full_name, role, is_active
                FROM users WHERE username = ? OR email = ?
            ''', (username, username))
            
            user = cursor.fetchone()
            
            if user and user[6] and self.verify_password(password, user[3]):
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
            return False, "Invalid username or password"
        
        except Exception as e:
            return False, f"Authentication error: {str(e)}"
    
    def save_chat_message(self, user_id: int, message: str, response: str) -> bool:
        """Save chat message to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO chat_history (user_id, message, response)
                VALUES (?, ?, ?)
            ''', (user_id, message, response))
            
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False
    
    def get_user_stats(self, user_id: int) -> Dict[str, int]:
        """Get user statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Session stats
            cursor.execute('''
                SELECT COUNT(*) as session_count, 
                       COALESCE(SUM(analysis_count), 0) as total_analysis
                FROM user_sessions WHERE user_id = ?
            ''', (user_id,))
            
            stats = cursor.fetchone()
            
            # Chat stats
            cursor.execute('''
                SELECT COUNT(*) as chat_count
                FROM chat_history WHERE user_id = ?
            ''', (user_id,))
            
            chat_stats = cursor.fetchone()
            conn.close()
            
            return {
                'session_count': stats[0] if stats else 0,
                'total_analysis': stats[1] if stats else 0,
                'chat_count': chat_stats[0] if chat_stats else 0
            }
        except Exception:
            return {'session_count': 0, 'total_analysis': 0, 'chat_count': 0}

class ResumeAnalyzer:
    """Advanced resume analysis with scoring"""
    
    def __init__(self):
        self.skill_categories = {
            'Programming Languages': [
                'Python', 'Java', 'JavaScript', 'C++', 'C#', 'PHP', 'Ruby', 'Go', 'Swift', 'Kotlin',
                'TypeScript', 'Scala', 'R', 'MATLAB', 'Rust', 'Dart', 'Perl'
            ],
            'Web Technologies': [
                'HTML', 'CSS', 'React', 'Angular', 'Vue.js', 'Node.js', 'Django', 'Flask', 'Spring Boot',
                'Express.js', 'Bootstrap', 'jQuery', 'Sass', 'WordPress', 'Laravel', 'Rails'
            ],
            'Databases': [
                'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Oracle', 'SQLite', 'Cassandra',
                'SQL Server', 'DynamoDB', 'Neo4j', 'Firebase', 'MariaDB'
            ],
            'Cloud & DevOps': [
                'AWS', 'Azure', 'Google Cloud', 'Docker', 'Kubernetes', 'Terraform',
                'Jenkins', 'GitLab CI', 'GitHub Actions', 'Heroku', 'Vercel', 'Ansible'
            ],
            'Data Science': [
                'Machine Learning', 'Deep Learning', 'TensorFlow', 'PyTorch', 'Pandas', 'NumPy',
                'Scikit-learn', 'Matplotlib', 'Seaborn', 'Jupyter', 'Tableau', 'Power BI'
            ],
            'Soft Skills': [
                'Leadership', 'Communication', 'Problem Solving', 'Teamwork', 'Project Management',
                'Critical Thinking', 'Adaptability', 'Time Management', 'Creativity', 'Public Speaking'
            ]
        }
        
        self.resume_tips = [
            "Use strong action verbs like 'achieved', 'improved', 'led', 'developed'",
            "Quantify your achievements with specific numbers and percentages",
            "Tailor your resume to match job description keywords",
            "Keep formatting clean and consistent throughout",
            "Use a professional email address and contact information",
            "Include a compelling summary or objective statement",
            "List experience in reverse chronological order",
            "Use bullet points for easy readability",
            "Keep resume to 1-2 pages maximum",
            "Proofread carefully for grammar and spelling errors"
        ]
    
    def analyze_resume(self, resume_text: str, job_description: str = "") -> Dict[str, Any]:
        """Comprehensive resume analysis with scoring"""
        if not resume_text:
            return {
                'score': 0,
                'strengths': [],
                'improvements': [],
                'missing_skills': [],
                'recommendations': []
            }
        
        analysis = {
            'score': 0,
            'strengths': [],
            'improvements': [],
            'missing_skills': [],
            'recommendations': []
        }
        
        resume_lower = resume_text.lower()
        word_count = len(resume_text.split())
        
        # Scoring criteria (100 points total)
        score = 0
        
        # 1. Length check (15 points)
        if 200 <= word_count <= 1000:
            score += 15
            analysis['strengths'].append(f"Appropriate resume length ({word_count} words)")
        elif word_count < 200:
            analysis['improvements'].append("Resume is too brief - add more details about experience")
        else:
            analysis['improvements'].append("Resume is too long - condense to 1-2 pages")
        
        # 2. Essential sections (25 points)
        essential_sections = {
            'contact': ('@' in resume_text or 'phone' in resume_lower or 'email' in resume_lower),
            'experience': ('experience' in resume_lower or 'work' in resume_lower),
            'education': ('education' in resume_lower or 'degree' in resume_lower),
            'skills': ('skills' in resume_lower or 'technical' in resume_lower),
            'summary': ('summary' in resume_lower or 'objective' in resume_lower)
        }
        
        sections_found = sum(essential_sections.values())
        score += (sections_found / 5) * 25
        
        if sections_found >= 4:
            analysis['strengths'].append("Contains most essential resume sections")
        else:
            missing = [s for s, found in essential_sections.items() if not found]
            analysis['improvements'].append(f"Missing sections: {', '.join(missing[:2])}")
        
        # 3. Action verbs (20 points)
        action_verbs = [
            'achieved', 'improved', 'increased', 'reduced', 'developed', 'led', 'managed', 'created',
            'implemented', 'designed', 'built', 'organized', 'coordinated', 'delivered', 'optimized'
        ]
        found_verbs = [verb for verb in action_verbs if verb in resume_lower]
        verb_score = min(20, len(found_verbs) * 2.5)
        score += verb_score
        
        if len(found_verbs) >= 5:
            analysis['strengths'].append("Uses strong action verbs effectively")
        else:
            analysis['improvements'].append("Add more action verbs like 'achieved', 'improved', 'led'")
        
        # 4. Quantification (20 points)
        quantifiers = re.findall(r'\d+%|\$\d+|\d+\+|increase|decrease|\d+ years?', resume_text)
        if len(quantifiers) >= 3:
            score += 20
            analysis['strengths'].append("Includes quantified achievements")
        else:
            analysis['improvements'].append("Add quantified results (numbers, percentages)")
        
        # 5. Skills presence (20 points)
        skills_found = 0
        for category, skills in self.skill_categories.items():
            for skill in skills:
                if skill.lower() in resume_lower:
                    skills_found += 1
        
        skill_score = min(20, skills_found)
        score += skill_score
        
        if skills_found >= 8:
            analysis['strengths'].append("Contains relevant technical skills")
        else:
            analysis['improvements'].append("Include more relevant skills")
        
        # Job description matching
        if job_description:
            jd_lower = job_description.lower()
            jd_words = set(jd_lower.split())
            resume_words = set(resume_lower.split())
            overlap = len(jd_words & resume_words)
            
            if overlap > 30:
                analysis['strengths'].append("Excellent keyword alignment with job")
            elif overlap > 15:
                analysis['strengths'].append("Good keyword overlap with job")
            else:
                analysis['improvements'].append("Better align content with job requirements")
            
            # Find missing skills
            for category, skills in self.skill_categories.items():
                for skill in skills:
                    if skill.lower() in jd_lower and skill.lower() not in resume_lower:
                        analysis['missing_skills'].append(skill)
        
        analysis['score'] = min(score, 100)
        
        # Generate recommendations
        if analysis['score'] >= 80:
            analysis['recommendations'] = [
                "Excellent resume! Minor tweaks for specific applications",
                "Continue tailoring keywords for each position"
            ]
        elif analysis['score'] >= 60:
            analysis['recommendations'] = [
                "Good foundation - add more quantified achievements",
                "Consider adding more relevant skills and certifications"
            ]
        else:
            analysis['recommendations'] = [
                "Focus on essential sections and structure",
                "Add quantified achievements and action verbs",
                "Improve professional formatting"
            ]
        
        return analysis

class SmartChatbot:
    """AI-powered career chatbot"""
    
    def __init__(self):
        self.setup_ai()
        self.knowledge_base = {
            'career_advice': {
                'interview_tips': [
                    "Research the company thoroughly before the interview",
                    "Prepare specific examples using the STAR method",
                    "Practice common interview questions out loud",
                    "Prepare thoughtful questions about the role and company",
                    "Dress professionally and arrive 10-15 minutes early",
                    "Follow up with a thank-you email within 24 hours"
                ],
                'job_search': [
                    "Use multiple job search channels (LinkedIn, company websites, referrals)",
                    "Customize your resume and cover letter for each application",
                    "Network actively within your industry",
                    "Follow up on applications professionally",
                    "Keep track of applications in a spreadsheet",
                    "Consider working with recruiters in your field"
                ],
                'skill_development': [
                    "Identify in-demand skills in your field",
                    "Take online courses from reputable platforms",
                    "Build a portfolio of projects to showcase skills",
                    "Contribute to open-source projects",
                    "Attend industry conferences and meetups",
                    "Find a mentor in your desired career path"
                ]
            }
        }
    
    def setup_ai(self):
        """Setup AI capabilities"""
        self.ai_status = "Basic"
        
        # Try Google Gemini
        if GEMINI_AVAILABLE:
            api_key = os.getenv('GOOGLE_API_KEY')
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    self.gemini_model = genai.GenerativeModel('gemini-pro')
                    self.ai_status = "Enhanced"
                except Exception:
                    pass
    
    def get_response(self, message: str, resume_text: str = "", job_description: str = "") -> str:
        """Generate chatbot response"""
        if self.ai_status == "Enhanced":
            return self.gemini_response(message, resume_text, job_description)
        else:
            return self.rule_based_response(message, resume_text, job_description)
    
    def gemini_response(self, message: str, resume_text: str, job_description: str) -> str:
        """Google Gemini AI response"""
        try:
            context = ""
            if resume_text:
                context += f"\nUser's Resume (first 1000 chars): {resume_text[:1000]}"
            if job_description:
                context += f"\nTarget Job Description: {job_description[:500]}"
            
            prompt = f"""You are an expert career counselor. A student/job seeker asks: "{message}"

{context}

Provide helpful, specific, and actionable career advice. Be encouraging but realistic. 
Keep response under 400 words and use clear formatting with bullet points when appropriate."""
            
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception:
            return self.rule_based_response(message, resume_text, job_description)
    
    def rule_based_response(self, message: str, resume_text: str, job_description: str) -> str:
        """Rule-based intelligent responses"""
        message_lower = message.lower()
        
        # Resume analysis
        if any(keyword in message_lower for keyword in ['analyze', 'review', 'resume', 'feedback']):
            if not resume_text:
                return """📄 **Please upload your resume first!**

I'll provide comprehensive analysis including:
• Content quality assessment
• Structure and formatting review
• Skills alignment check
• Specific improvement recommendations
• Professional scoring (0-100)

Upload your resume above to get started! 🚀"""
            
            analyzer = ResumeAnalyzer()
            analysis = analyzer.analyze_resume(resume_text, job_description)
            
            response = f"## 📊 **Resume Analysis Results**\n\n"
            response += f"**Overall Score: {analysis['score']}/100**\n\n"
            
            if analysis['score'] >= 80:
                response += "🏆 **Excellent Resume!** You're well-prepared for applications.\n\n"
            elif analysis['score'] >= 60:
                response += "✅ **Good Resume** with solid foundation. Some improvements will enhance it.\n\n"
            else:
                response += "🔧 **Room for Improvement** - Focus on key areas below.\n\n"
            
            if analysis['strengths']:
                response += "### ✅ **Strengths:**\n"
                for strength in analysis['strengths'][:4]:
                    response += f"• {strength}\n"
                response += "\n"
            
            if analysis['improvements']:
                response += "### 🔧 **Improvements:**\n"
                for improvement in analysis['improvements'][:4]:
                    response += f"• {improvement}\n"
                response += "\n"
            
            return response
        
        # Job suitability
        elif any(keyword in message_lower for keyword in ['suitable', 'fit', 'match', 'job']):
            if not resume_text or not job_description:
                return """🎯 **Job Suitability Analysis**

I need both your resume and the job description to assess compatibility.

**What I'll analyze:**
• Keyword matching between resume and job requirements
• Skills alignment and gaps
• Experience relevance
• Overall competitiveness
• Specific recommendations to improve your candidacy

Please upload your resume and paste the job description! 📈"""
            
            # Calculate basic compatibility
            resume_words = set(resume_text.lower().split())
            jd_words = set(job_description.lower().split())
            overlap = len(resume_words & jd_words)
            similarity = (overlap / len(jd_words)) * 100 if jd_words else 0
            
            if similarity > 35:
                match_level = "🏆 **Excellent Match!**"
                advice = "You're very well-suited. Apply with confidence and highlight your relevant experience!"
            elif similarity > 25:
                match_level = "✅ **Good Match**"
                advice = "You have strong qualifications. Tailor your application to emphasize relevant skills!"
            elif similarity > 15:
                match_level = "⚠️ **Moderate Match**"
                advice = "You have some relevant qualifications. Focus on transferable skills and growth potential!"
            else:
                match_level = "🔧 **Limited Match**"
                advice = "This role might be challenging. Consider developing key skills or targeting more aligned positions!"
            
            return f"""## 🎯 **Job Compatibility Analysis**

{match_level}

**Keyword Similarity:** {similarity:.0f}%

**Recommendation:** {advice}

**Next Steps:**
• Review job requirements carefully
• Highlight matching experience prominently
• Address any skill gaps in your cover letter
• Consider reaching out to employees at the company"""
        
        # Skill development
        elif any(keyword in message_lower for keyword in ['skill', 'learn', 'develop', 'study']):
            recommendations = []
            
            if any(term in message_lower for term in ['software', 'programming', 'developer']):
                recommendations = [
                    "🐍 **Python** - Versatile for web development, data science, automation",
                    "⚛️ **React/JavaScript** - Essential for modern web development",
                    "☁️ **Cloud Platforms (AWS/Azure)** - High demand across industries",
                    "🔧 **Git/Version Control** - Fundamental for any development work"
                ]
            elif any(term in message_lower for term in ['data', 'analytics', 'science']):
                recommendations = [
                    "📊 **Python/R** - Core languages for data analysis",
                    "📈 **SQL** - Essential for working with databases",
                    "🤖 **Machine Learning** - Growing field with excellent prospects",
                    "📊 **Tableau/Power BI** - Popular data visualization tools"
                ]
            else:
                recommendations = [
                    "💼 **Communication** - Critical for any career advancement",
                    "📊 **Data Analysis** - Valuable across all industries",
                    "👥 **Project Management** - Useful for leadership roles",
                    "💻 **Digital Marketing** - High demand in modern business"
                ]
            
            response = "🚀 **Skill Development Recommendations**\n\n"
            for rec in recommendations:
                response += f"• {rec}\n"
            
            response += f"""

**Learning Strategy:**
1. **Focus on 1-2 skills** at a time for effective learning
2. **Practice consistently** - even 30 minutes daily helps
3. **Build projects** to apply what you learn
4. **Get certified** to validate your skills
5. **Join communities** to learn from others

**Recommended Platforms:**
• **Coursera/edX** - University-quality courses
• **Udemy** - Practical, hands-on training  
• **YouTube** - Free tutorials and explanations
• **LinkedIn Learning** - Professional development"""
            
            return response
        
        # Interview advice
        elif any(keyword in message_lower for keyword in ['interview', 'interviewing']):
            tips = random.sample(self.knowledge_base['career_advice']['interview_tips'], 4)
            
            response = "🎯 **Interview Success Tips**\n\n"
            for tip in tips:
                response += f"• {tip}\n"
            
            response += """

**Common Interview Questions to Prepare:**
• "Tell me about yourself"
• "Why are you interested in this role?"
• "What are your greatest strengths/weaknesses?"
• "Describe a challenging situation and how you handled it"
• "Where do you see yourself in 5 years?"

**Remember:** Interviews are conversations, not interrogations. Be yourself and let your personality shine! 🌟"""
            
            return response
        
        # General career advice
        else:
            return """👋 **Hello! I'm your AI Career Assistant!**

I can help you with:

🔍 **Resume Analysis** - Upload your resume for detailed feedback and scoring
📝 **Resume Optimization** - Specific tips to improve your resume
🎯 **Job Matching** - Check how well you fit specific positions  
📚 **Skill Development** - Recommendations for career growth
💼 **Career Strategy** - Planning and professional development
🎤 **Interview Preparation** - Tips and common questions

**What would you like help with today?**

*Examples:*
• "Analyze my resume"
• "Am I suitable for this job?"
• "What skills should I learn?"
• "Give me interview tips"

Ready to boost your career? 🚀"""

# Utility Functions
def extract_text_from_file(uploaded_file) -> str:
    """Extract text from uploaded file"""
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
        return text
        
    except Exception as e:
        st.error(f"❌ Error processing {uploaded_file.name}: {e}")
        return ""

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts"""
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
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return float(similarity) * 100
        
    except Exception:
        return 0.0

def get_ml_prediction(resume_text: str) -> Dict[str, Any]:
    """Get ML prediction for resume classification"""
    try:
        if os.path.exists('resume_model.joblib'):
            model_data = joblib.load('resume_model.joblib')
            
            vectorizer = model_data['vectorizer']
            model = model_data['model']
            
            resume_vector = vectorizer.transform([resume_text.lower()])
            prediction = model.predict(resume_vector)[0]
            probabilities = model.predict_proba(resume_vector)[0]
            confidence = float(max(probabilities))
            
            if confidence >= 0.8:
                recommendation = "Excellent Fit"
            elif confidence >= 0.6:
                recommendation = "Strong Candidate"
            elif confidence >= 0.4:
                recommendation = "Potential Candidate"
            else:
                recommendation = "Needs Review"
            
            return {
                'category': prediction,
                'confidence': confidence,
                'confidence_percentage': f"{confidence*100:.1f}%",
                'hiring_recommendation': recommendation,
                'prediction_successful': True
            }
    except Exception:
        pass
    
    return {
        'category': 'Classification Unavailable',
        'confidence': 0.0,
        'confidence_percentage': '0.0%',
        'hiring_recommendation': 'Manual Review Required',
        'prediction_successful': False
    }

def extract_skills_advanced(text: str) -> Dict[str, List[str]]:
    """Extract skills from text"""
    skills_database = {
        'Programming': ['python', 'java', 'javascript', 'react', 'node', 'django', 'spring'],
        'Databases': ['mysql', 'postgresql', 'mongodb', 'oracle', 'redis'],
        'Cloud': ['aws', 'azure', 'docker', 'kubernetes', 'terraform'],
        'Tools': ['git', 'jenkins', 'jira', 'confluence', 'slack']
    }
    
    found_skills = {}
    text_lower = text.lower()
    
    for category, skills in skills_database.items():
        category_skills = []
        for skill in skills:
            if skill in text_lower:
                category_skills.append(skill.title())
        
        if category_skills:
            found_skills[category] = category_skills
    
    return found_skills

def create_score_chart(scores: List[float]):
    """Create score distribution chart"""
    if not scores:
        return None
    
    try:
        fig = px.histogram(
            x=scores,
            nbins=min(15, len(scores)),
            title="Score Distribution",
            labels={'x': 'Score (%)', 'y': 'Count'},
            color_discrete_sequence=['#2563eb']
        )
        
        mean_score = np.mean(scores)
        fig.add_vline(x=mean_score, line_dash="dash", line_color="#059669", 
                      annotation_text=f"Average: {mean_score:.1f}%")
        
        fig.update_layout(height=400, showlegend=False)
        return fig
    except Exception:
        return None

def create_hiring_chart(recommendations: List[str]):
    """Create hiring recommendations chart"""
    if not recommendations:
        return None
    
    try:
        rec_counts = pd.Series(recommendations).value_counts()
        
        color_map = {            'Excellent Fit': '#059669',
            'Strong Candidate': '#2563eb',
            'Good Candidate': '#0ea5e9',
            'Potential Candidate': '#d97706',
            'Needs Review': '#dc2626',
            'Manual Review Required': '#64748b'
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
            title="Hiring Recommendations",
            xaxis_title="Recommendation",
            yaxis_title="Count",
            height=400
        )
        
        return fig
    except Exception:
        return None
