"""
🎓 BULLETPROOF STUDENT DASHBOARD - ALL ERRORS FIXED
Complete student interface with comprehensive error handling
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import traceback

def student_dashboard():
    """Main student dashboard with bulletproof error handling"""
    
    try:
        # Dashboard header
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea, #764ba2); padding: 2rem; border-radius: 15px; color: white; margin-bottom: 2rem;">
            <h2 style="margin: 0; font-size: 2.5rem;">🎓 Student Career Co-Pilot</h2>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
                Your AI-powered career development companion
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Main tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📄 Resume Analysis", 
            "💬 Career Assistant", 
            "📊 Progress Tracking", 
            "🎯 Skill Development"
        ])
        
        with tab1:
            render_resume_analysis_section()
        
        with tab2:
            render_career_assistant_section()
        
        with tab3:
            render_progress_tracking_section()  # FIXED VERSION
        
        with tab4:
            render_skill_development_section()
    
    except Exception as e:
        st.error(f"❌ Student Dashboard Error: {e}")
        st.code(traceback.format_exc())
        
        # Show emergency interface
        st.markdown("## 🚨 Emergency Interface")
        emergency_student_interface()

def render_resume_analysis_section():
    """Resume analysis section"""
    
    try:
        st.markdown("## 📄 Resume Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📄 Upload Resume")
            resume_file = st.file_uploader(
                "Choose your resume:",
                type=['pdf', 'docx', 'txt'],
                key="student_resume"
            )
            
            if resume_file:
                st.success(f"✅ Uploaded: {resume_file.name}")
        
        with col2:
            st.markdown("### 🎯 Job Description")
            jd_text = st.text_area(
                "Paste job description:",
                height=200,
                key="student_jd"
            )
        
        if resume_file and jd_text:
            if st.button("🔍 Analyze Resume", type="primary"):
                with st.spinner("🤖 Analyzing..."):
                    # Mock analysis - safe implementation
                    import time
                    time.sleep(2)
                    
                    score = np.random.randint(65, 95)
                    
                    st.markdown(f"""
                    <div style="background: #28a745; color: white; padding: 2rem; border-radius: 15px; text-align: center;">
                        <h2 style="margin: 0;">🏆</h2>
                        <h3 style="margin: 0.5rem 0;">{score}%</h3>
                        <p style="margin: 0;">Match Score</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### ✅ Matched Skills")
                        skills = ['Python', 'SQL', 'JavaScript', 'Communication']
                        for skill in skills:
                            st.success(f"✓ {skill}")
                    
                    with col2:
                        st.markdown("#### ❌ Missing Skills")
                        missing = ['React', 'AWS', 'Docker']
                        for skill in missing:
                            st.warning(f"→ {skill}")
    
    except Exception as e:
        st.error(f"❌ Resume Analysis Error: {e}")
        st.info("Please try again or refresh the page.")

def render_career_assistant_section():
    """Career assistant section"""
    
    try:
        st.markdown("## 💬 Career Assistant")
        
        # Initialize chat safely
        if 'student_chat' not in st.session_state:
            st.session_state.student_chat = [
                {"role": "assistant", "content": "Hello! How can I help with your career today?"}
            ]
        
        # Display messages
        for msg in st.session_state.student_chat:
            role = msg.get('role', 'assistant')
            content = msg.get('content', '')
            
            if role == 'user':
                st.markdown(f"**You:** {content}")
            else:
                st.markdown(f"**Assistant:** {content}")
        
        # Input
        user_input = st.text_input("Ask a career question:")
        
        if st.button("Send") and user_input:
            st.session_state.student_chat.append({"role": "user", "content": user_input})
            
            # Simple response logic
            if 'resume' in user_input.lower():
                response = "To improve your resume: Use action verbs, quantify achievements, and tailor it for each job application."
            elif 'skills' in user_input.lower():
                response = "Focus on in-demand skills like Python, cloud computing, and data analysis. Consider online courses."
            elif 'interview' in user_input.lower():
                response = "For interviews: Research the company, practice common questions, and prepare examples using STAR method."
            else:
                response = "That's a great question! Focus on continuous learning, networking, and building practical experience."
            
            st.session_state.student_chat.append({"role": "assistant", "content": response})
            st.rerun()
    
    except Exception as e:
        st.error(f"❌ Career Assistant Error: {e}")
        st.info("Chat feature temporarily unavailable.")

def render_progress_tracking_section():
    """Progress tracking section - COMPLETELY FIXED VERSION"""
    
    st.markdown("## 📊 Progress Tracking")
    
    try:
        # Metrics - safe version
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Analyses Done", "3", "1")
        
        with col2:
            st.metric("Latest Score", "78%", "5%")
        
        with col3:
            st.metric("Skills Tracked", "12", "2")
        
        with col4:
            st.metric("Goals Set", "2", "1")
        
        # Goal setting form - bulletproof version
        st.markdown("### 🎯 Set Career Goals")
        
        with st.form("career_goals"):
            target_role = st.text_input("Target Role:", placeholder="e.g., Software Engineer")
            target_score = st.slider("Target Score:", 0, 100, 85)
            timeline = st.selectbox("Timeline:", ["1 month", "3 months", "6 months", "1 year"])
            
            if st.form_submit_button("💾 Save Goals"):
                # COMPLETELY SAFE way to store goals
                st.session_state.student_career_goals = {
                    'target_role': target_role or 'Not specified',
                    'target_score': target_score,
                    'timeline': timeline,
                    'set_date': datetime.now().isoformat()
                }
                st.success("✅ Goals saved successfully!")
        
        # Display current goals - BULLETPROOF VERSION
        st.markdown("### 🎯 Current Goals")
        
        # THE COMPLETE FIX - 100% safe access to goals
        goals = st.session_state.get('student_career_goals', None)
        
        if goals is not None and isinstance(goals, dict) and len(goals) > 0:
            # COMPLETELY SAFE way to access dictionary values with multiple safeguards
            goal_col1, goal_col2 = st.columns(2)
            
            with goal_col1:
                try:
                    target_role = goals.get('target_role', 'Not set')
                    if target_role is None:
                        target_role = 'Not set'
                    
                    target_score = goals.get('target_score', 0)
                    if target_score is None:
                        target_score = 0
                    
                    st.info(f"**Target Role:** {target_role}")
                    st.info(f"**Target Score:** {target_score}%")
                except Exception as e:
                    st.error(f"Error displaying role/score: {e}")
            
            with goal_col2:
                try:
                    timeline = goals.get('timeline', 'Not set')
                    if timeline is None:
                        timeline = 'Not set'
                    
                    set_date = goals.get('set_date', 'Unknown')
                    if set_date is None:
                        set_date = 'Unknown'
                    elif isinstance(set_date, str) and len(set_date) > 10:
                        set_date = set_date[:10]  # Get just the date part
                    
                    st.info(f"**Timeline:** {timeline}")
                    st.info(f"**Set Date:** {set_date}")
                except Exception as e:
                    st.error(f"Error displaying timeline/date: {e}")
        else:
            # No goals set yet - safe fallback
            st.info("🎯 No career goals set yet. Use the form above to set your goals!")
            st.markdown("""
            **Why set goals?**
            - Track your career progress
            - Stay motivated and focused
            - Measure improvement over time
            - Plan your learning path
            """)
        
        # Progress chart (mock data) - safe version
        st.markdown("### 📈 Progress Over Time")
        
        try:
            # Create safe mock data
            dates = pd.date_range(start='2024-01-01', periods=6, freq='M')
            scores = [65, 70, 72, 75, 78, 80]
            
            chart_data = pd.DataFrame({
                'Date': dates,
                'Score': scores
            })
            
            st.line_chart(chart_data.set_index('Date'))
        except Exception as e:
            st.error(f"Chart error: {e}")
            st.info("📊 Progress chart temporarily unavailable")
    
    except Exception as e:
        st.error(f"❌ Progress Tracking Error: {e}")
        st.code(traceback.format_exc())
        
        # Fallback simple interface
        st.markdown("### 📊 Basic Progress Info")
        st.info("• Complete resume analyses to track progress")
        st.info("• Set career goals to monitor improvement")
        st.info("• Check back regularly for updates")

def render_skill_development_section():
    """Skill development section"""
    
    try:
        st.markdown("## 🎯 Skill Development")
        
        # Skill categories
        categories = {
            'Programming': ['Python', 'JavaScript', 'Java', 'SQL'],
            'Web Development': ['HTML/CSS', 'React', 'Node.js', 'Bootstrap'],
            'Data Science': ['Data Analysis', 'Machine Learning', 'Statistics', 'Visualization'],
            'Cloud & DevOps': ['AWS', 'Azure', 'Docker', 'Git'],
            'Soft Skills': ['Communication', 'Leadership', 'Teamwork', 'Problem Solving']
        }
        
        # Self-assessment
        st.markdown("### 📝 Skills Self-Assessment")
        
        selected_category = st.selectbox("Choose category:", list(categories.keys()))
        
        if selected_category and selected_category in categories:
            skills = categories[selected_category]
            
            for skill in skills:
                level = st.slider(
                    f"{skill} proficiency:",
                    0, 100, 50,
                    key=f"skill_{skill.replace(' ', '_').lower()}"
                )
                
                if level < 50:
                    st.info(f"💡 Focus on developing {skill}")
                elif level >= 80:
                    st.success(f"🎉 Strong in {skill}!")
        
        # Learning resources
        st.markdown("### 📚 Recommended Resources")
        
        resources = [
            "🎓 **Coursera** - University-level courses",
            "💻 **freeCodeCamp** - Free programming tutorials", 
            "📖 **MDN Web Docs** - Web development reference",
            "🏆 **LeetCode** - Programming practice",
            "🌐 **Stack Overflow** - Q&A community",
            "📊 **Kaggle** - Data science competitions"
        ]
        
        for resource in resources:
            st.markdown(resource)
    
    except Exception as e:
        st.error(f"❌ Skill Development Error: {e}")
        st.info("Skill development features temporarily unavailable.")

def emergency_student_interface():
    """Emergency fallback interface"""
    
    st.markdown("### 🚨 Basic Features Available")
    
    # Simple file upload
    st.markdown("#### 📄 Quick Resume Check")
    file = st.file_uploader("Upload resume", type=['pdf', 'docx', 'txt'])
    
    if file:
        st.success(f"✅ File received: {file.name}")
        st.info("🔍 Analysis features temporarily disabled")
    
    # Basic tips
    st.markdown("#### 💡 Career Tips")
    
    tips = [
        "Keep your resume updated and tailored",
        "Develop both technical and soft skills", 
        "Network actively in your industry",
        "Practice interviewing regularly",
        "Stay current with industry trends"
    ]
    
    for i, tip in enumerate(tips, 1):
        st.info(f"{i}. {tip}")
    
    # Contact info
    st.markdown("#### 🆘 Need Help?")
    st.info("If you continue having issues, please refresh the page or contact support.")

# Initialize session state safely
def init_student_session():
    """Initialize student-specific session state safely"""
    
    defaults = {
        'student_chat': [],
        'student_career_goals': None,  # Initialize as None
        'analysis_history': [],
        'skill_assessments': {}
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Initialize on module load
init_student_session()
