import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from datetime import datetime
import joblib
import os

# Basic PDF processing (fallback)
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="🤖 AI-Powered Resume Matcher",
    page_icon="🎯",
    layout="wide"
)

# Enhanced CSS with ML styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .ml-badge {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.2rem;
        display: inline-block;
    }
    
    .confidence-high { background: linear-gradient(45deg, #28a745, #20c997); }
    .confidence-medium { background: linear-gradient(45deg, #ffc107, #fd7e14); }
    .confidence-low { background: linear-gradient(45deg, #dc3545, #e83e8c); }
    
    .score-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
    }
    
    .ml-insight {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .hiring-rec-strong { border-left-color: #28a745; background: #d4edda; }
    .hiring-rec-hire { border-left-color: #17a2b8; background: #d1ecf1; }
    .hiring-rec-conditional { border-left-color: #ffc107; background: #fff3cd; }
    .hiring-rec-no { border-left-color: #dc3545; background: #f8d7da; }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .ai-status {
        background: linear-gradient(45deg, #28a745, #20c997);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ML Model Loading Function
@st.cache_resource
def load_ml_model():
    """Load the trained ML model"""
    try:
        if os.path.exists('resume_model.joblib'):
            model_data = joblib.load('resume_model.joblib')
            st.sidebar.success("🤖 AI Model: LOADED")
            return model_data
        else:
            st.sidebar.error("🤖 AI Model: NOT FOUND")
            st.sidebar.info("Place 'resume_model.joblib' in the project folder")
            return None
    except Exception as e:
        st.sidebar.error(f"🤖 AI Model: ERROR - {str(e)}")
        return None

# Initialize ML model
ml_model = load_ml_model()

# Header with AI status
if ml_model:
    ai_status = "🤖 AI-POWERED"
    ai_subtitle = "Advanced ML Model Active - Intelligent Resume Analysis"
else:
    ai_status = "📊 BASIC MODE"
    ai_subtitle = "Upload resume_model.joblib to enable AI features"

st.markdown(f"""
<div class="main-header">
    <div style="display: inline-block;" class="ai-status">{ai_status}</div>
    <h1>🎯 Resume Matcher Pro</h1>
    <p>{ai_subtitle}</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'jd_text' not in st.session_state:
    st.session_state.jd_text = ""

if 'results' not in st.session_state:
    st.session_state.results = []

# Enhanced text extraction
def extract_text_simple(uploaded_file):
    """Enhanced text extraction with better error handling"""
    try:
        if uploaded_file.type == "text/plain":
            return str(uploaded_file.read(), "utf-8")
        
        elif uploaded_file.type == "application/pdf" and PDF_AVAILABLE:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        
        else:
            st.warning(f"⚠️ File type {uploaded_file.type} - trying text extraction")
            return str(uploaded_file.read(), "utf-8", errors="ignore")
    
    except Exception as e:
        st.error(f"❌ Error reading {uploaded_file.name}: {e}")
        return ""

# Enhanced similarity calculation
def calculate_similarity(text1, text2):
    """Calculate TF-IDF similarity with better preprocessing"""
    try:
        if not text1 or not text2:
            return 0.0
        
        # Clean texts
        text1_clean = re.sub(r'[^\w\s]', ' ', text1.lower())
        text2_clean = re.sub(r'[^\w\s]', ' ', text2.lower())
        
        vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=1500,
            ngram_range=(1, 2),
            min_df=1
        )
        
        tfidf_matrix = vectorizer.fit_transform([text1_clean, text2_clean])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity) * 100
    
    except Exception as e:
        return 0.0

# ML Prediction Function
def get_ml_prediction(resume_text):
    """Get ML model prediction with confidence"""
    if not ml_model:
        return {
            'category': 'No ML Model',
            'confidence': 0.0,
            'hiring_recommendation': 'Manual Review Required'
        }
    
    try:
        # Clean the resume text (same as training)
        resume_clean = resume_text.lower()
        
        # Transform using the saved vectorizer
        vectorizer = ml_model['vectorizer']
        model = ml_model['model']
        
        # Vectorize the resume
        resume_vector = vectorizer.transform([resume_clean])
        
        # Get prediction and probability
        prediction = model.predict(resume_vector)[0]
        probabilities = model.predict_proba(resume_vector)[0]
        confidence = max(probabilities)
        
        # Generate hiring recommendation based on confidence
        if confidence >= 0.9:
            hiring_rec = "STRONG HIRE"
        elif confidence >= 0.75:
            hiring_rec = "HIRE"
        elif confidence >= 0.6:
            hiring_rec = "CONDITIONAL HIRE" 
        else:
            hiring_rec = "NO HIRE"
        
        return {
            'category': prediction,
            'confidence': confidence,
            'hiring_recommendation': hiring_rec,
            'all_probabilities': dict(zip(ml_model.get('categories', []), probabilities))
        }
    
    except Exception as e:
        st.error(f"ML Prediction Error: {e}")
        return {
            'category': 'Prediction Error',
            'confidence': 0.0,
            'hiring_recommendation': 'Manual Review Required'
        }

# Enhanced skills extraction
def extract_skills_advanced(text):
    """Enhanced skills extraction with more categories"""
    skills_database = {
        'Programming Languages': [
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 
            'go', 'rust', 'swift', 'kotlin', 'scala', 'r', 'matlab'
        ],
        'Web Technologies': [
            'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 
            'flask', 'spring', 'laravel', 'bootstrap', 'jquery'
        ],
        'Databases': [
            'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'sqlite',
            'elasticsearch', 'cassandra', 'dynamodb'
        ],
        'Cloud & DevOps': [
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git',
            'terraform', 'ansible', 'linux', 'nginx'
        ],
        'Data Science': [
            'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'pandas',
            'numpy', 'scikit-learn', 'matplotlib', 'seaborn', 'jupyter'
        ],
        'Soft Skills': [
            'leadership', 'communication', 'project management', 'agile', 'scrum',
            'team collaboration', 'problem solving'
        ]
    }
    
    found_skills = {}
    text_lower = text.lower()
    
    for category, skills_list in skills_database.items():
        found_in_category = []
        for skill in skills_list:
            if skill in text_lower:
                found_in_category.append(skill)
        
        if found_in_category:
            found_skills[category] = found_in_category
    
    return found_skills

# Generate AI Insights
def generate_ai_insights(result, jd_text):
    """Generate AI-powered insights"""
    insights = []
    
    confidence = result['ml_confidence']
    category = result['ml_category']
    score = result['similarity_score']
    
    # Confidence-based insights
    if confidence > 0.9:
        insights.append("🎯 High-confidence ML prediction - Very reliable categorization")
    elif confidence > 0.7:
        insights.append("✅ Good ML confidence - Reliable prediction") 
    else:
        insights.append("⚠️ Lower ML confidence - Manual review recommended")
    
    # Score-based insights
    if score > 80:
        insights.append("🏆 Excellent similarity match - Strong alignment with job requirements")
    elif score > 60:
        insights.append("👍 Good match - Candidate shows relevant experience")
    else:
        insights.append("📚 Limited match - Consider for junior roles or with training")
    
    # Category-specific insights
    if "senior" in category.lower():
        insights.append("🎖️ Senior-level candidate - Suitable for leadership roles")
    elif "data" in category.lower():
        insights.append("📊 Data specialist - Strong analytical background")
    elif "engineer" in category.lower():
        insights.append("⚙️ Technical engineering background - Good for development roles")
    
    return insights

# Main interface
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("## 📋 Job Description")
    
    jd_method = st.radio("Input method:", ["Paste Text", "Upload File"])
    
    if jd_method == "Paste Text":
        jd_text = st.text_area("Paste job description:", height=200)
        if jd_text:
            st.session_state.jd_text = jd_text
            st.success("✅ JD loaded!")
    
    else:
        jd_file = st.file_uploader("Upload JD file:", type=['txt', 'pdf'])
        if jd_file:
            jd_text = extract_text_simple(jd_file)
            if jd_text:
                st.session_state.jd_text = jd_text
                st.success("✅ JD extracted!")
    
    st.markdown("## 📄 Upload Resumes")
    
    resume_files = st.file_uploader(
        "Choose resume files:",
        type=['txt', 'pdf'],
        accept_multiple_files=True
    )
    
    if resume_files and st.session_state.jd_text:
        if st.button("🚀 Analyze with AI", type="primary"):
            
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, resume_file in enumerate(resume_files):
                status_text.text(f"🔄 Processing: {resume_file.name}")
                progress_bar.progress((i + 1) / len(resume_files))
                
                # Extract resume text
                resume_text = extract_text_simple(resume_file)
                
                if resume_text:
                    # Calculate TF-IDF similarity
                    similarity_score = calculate_similarity(resume_text, st.session_state.jd_text)
                    
                    # Get ML prediction
                    ml_prediction = get_ml_prediction(resume_text)
                    
                    # Extract skills with categories
                    resume_skills = extract_skills_advanced(resume_text)
                    jd_skills = extract_skills_advanced(st.session_state.jd_text)
                    
                    # Calculate skills overlap
                    all_resume_skills = []
                    all_jd_skills = []
                    
                    for skills_list in resume_skills.values():
                        all_resume_skills.extend(skills_list)
                    
                    for skills_list in jd_skills.values():
                        all_jd_skills.extend(skills_list)
                    
                    matched_skills = list(set(all_resume_skills) & set(all_jd_skills))
                    
                    if all_jd_skills:
                        skills_match_pct = (len(matched_skills) / len(all_jd_skills)) * 100
                    else:
                        skills_match_pct = 0
                    
                    # Calculate composite score (TF-IDF + ML confidence)
                    if ml_model:
                        composite_score = (similarity_score * 0.6) + (ml_prediction['confidence'] * 100 * 0.4)
                    else:
                        composite_score = similarity_score
                    
                    # Extract candidate name
                    lines = resume_text.split('\n')[:10]
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
                    
                    result = {
                        'filename': resume_file.name,
                        'candidate_name': candidate_name,
                        'similarity_score': round(similarity_score, 1),
                        'composite_score': round(composite_score, 1),
                        'ml_category': ml_prediction['category'],
                        'ml_confidence': round(ml_prediction['confidence'], 3),
                        'hiring_recommendation': ml_prediction['hiring_recommendation'],
                        'skills_match': round(skills_match_pct, 1),
                        'matched_skills': matched_skills,
                        'resume_skills': resume_skills,
                        'jd_skills': jd_skills,
                        'resume_preview': resume_text[:800] + "..." if len(resume_text) > 800 else resume_text,
                        'all_probabilities': ml_prediction.get('all_probabilities', {})
                    }
                    
                    # Generate AI insights
                    result['ai_insights'] = generate_ai_insights(result, st.session_state.jd_text)
                    
                    results.append(result)
            
            # Sort by composite score
            results.sort(key=lambda x: x['composite_score'], reverse=True)
            
            # Add rankings
            for i, result in enumerate(results):
                result['ranking'] = i + 1
            
            st.session_state.results = results
            progress_bar.progress(1.0)
            status_text.text("✅ AI Analysis Complete!")
            
            st.balloons()  # Celebration animation!

with col2:
    st.markdown("## 📊 AI-Powered Results")
    
    if st.session_state.results:
        # Enhanced summary metrics
        scores = [r['composite_score'] for r in st.session_state.results]
        confidences = [r['ml_confidence'] for r in st.session_state.results if r['ml_confidence'] > 0]
        
        met_col1, met_col2, met_col3, met_col4 = st.columns(4)
        
        with met_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem; font-weight: bold;">{len(st.session_state.results)}</div>
                <div>Total Analyzed</div>
            </div>
            """, unsafe_allow_html=True)
        
        with met_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem; font-weight: bold;">{np.mean(scores):.1f}%</div>
                <div>Avg AI Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        with met_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem; font-weight: bold;">{max(scores):.1f}%</div>
                <div>Top Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        with met_col4:
            avg_conf = np.mean(confidences) if confidences else 0
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem; font-weight: bold;">{avg_conf:.2f}</div>
                <div>AI Confidence</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Results display
        st.markdown("### 🏆 Ranked Candidates")
        
        for result in st.session_state.results[:15]:  # Top 15
            
            score = result['composite_score']
            ml_conf = result['ml_confidence']
            hiring_rec = result['hiring_recommendation']
            
            # Determine styling based on hiring recommendation
            if hiring_rec == "STRONG HIRE":
                card_class = "hiring-rec-strong"
                icon = "🏆"
                badge_class = "confidence-high"
            elif hiring_rec == "HIRE":
                card_class = "hiring-rec-hire" 
                icon = "✅"
                badge_class = "confidence-high"
            elif hiring_rec == "CONDITIONAL HIRE":
                card_class = "hiring-rec-conditional"
                icon = "⚠️"
                badge_class = "confidence-medium"
            else:
                card_class = "hiring-rec-no"
                icon = "❌"
                badge_class = "confidence-low"
            
            with st.expander(f"{icon} #{result['ranking']} - {result['candidate_name']} ({score:.1f}%) - {hiring_rec}", expanded=(result['ranking'] <= 3)):
                
                # Header information
                header_col1, header_col2, header_col3 = st.columns(3)
                
                with header_col1:
                    st.markdown("**📄 Basic Info:**")
                    st.write(f"• File: {result['filename']}")
                    st.write(f"• Candidate: {result['candidate_name']}")
                    st.write(f"• Composite Score: {result['composite_score']:.1f}%")
                    st.write(f"• TF-IDF Score: {result['similarity_score']:.1f}%")
                
                with header_col2:
                    st.markdown("**🤖 AI Analysis:**")
                    st.markdown(f"<span class='ml-badge {badge_class}'>{result['ml_category']}</span>", unsafe_allow_html=True)
                    st.write(f"• Confidence: {ml_conf*100:.1f}%")
                    st.write(f"• Skills Match: {result['skills_match']:.1f}%")
                    st.markdown(f"**🎯 Recommendation:** {hiring_rec}")
                
                with header_col3:
                    st.markdown("**🎯 Top Skills:**")
                    matched_skills = result['matched_skills'][:6]
                    for skill in matched_skills:
                        st.markdown(f"<span class='ml-badge'>{skill}</span>", unsafe_allow_html=True)
                
                # AI Insights section
                if result['ai_insights']:
                    st.markdown("**💡 AI Insights:**")
                    for insight in result['ai_insights']:
                        st.markdown(f"<div class='ml-insight'>{insight}</div>", unsafe_allow_html=True)
                
                # Skills by category
                st.markdown("**🔧 Skills by Category:**")
                skills_col1, skills_col2 = st.columns(2)
                
                skill_categories = list(result['resume_skills'].keys())
                mid_point = len(skill_categories) // 2
                
                with skills_col1:
                    for category in skill_categories[:mid_point]:
                        st.write(f"**{category}:**")
                        for skill in result['resume_skills'][category][:4]:
                            st.write(f"• {skill}")
                
                with skills_col2:
                    for category in skill_categories[mid_point:]:
                        st.write(f"**{category}:**") 
                        for skill in result['resume_skills'][category][:4]:
                            st.write(f"• {skill}")
                
                # ML Probability breakdown (if available)
                if ml_model and 'all_probabilities' in result:
                    st.markdown("**🧠 ML Category Probabilities:**")
                    
                    probs = result['all_probabilities']
                    top_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
                    
                    for category, prob in top_probs:
                        prob_pct = prob * 100
                        st.progress(prob)
                        st.write(f"{category}: {prob_pct:.1f}%")
                
                # Resume preview
                st.markdown("**📄 Resume Preview:**")
                st.text_area(
                    "Content Preview",
                    result['resume_preview'],
                    height=200,
                    disabled=True,
                    key=f"preview_{result['ranking']}"
                )
        
        # Enhanced export section
        st.markdown("### 📤 Export AI Results")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            if st.button("📊 Download Detailed CSV"):
                # Enhanced CSV with ML data
                csv_data = []
                for result in st.session_state.results:
                    csv_data.append({
                        'Ranking': result['ranking'],
                        'Filename': result['filename'],
                        'Candidate Name': result['candidate_name'],
                        'Composite Score (%)': result['composite_score'],
                        'TF-IDF Score (%)': result['similarity_score'],
                        'ML Category': result['ml_category'],
                        'ML Confidence': f"{result['ml_confidence']*100:.1f}%",
                        'Hiring Recommendation': result['hiring_recommendation'],
                        'Skills Match (%)': result['skills_match'],
                        'Matched Skills': ', '.join(result['matched_skills'][:10]),
                        'Top AI Insight': result['ai_insights'][0] if result['ai_insights'] else 'None'
                    })
                
                df = pd.DataFrame(csv_data)
                csv_string = df.to_csv(index=False)
                
                st.download_button(
                    label="💾 Download AI-Enhanced Report",
                    data=csv_string,
                    file_name=f"ai_resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with export_col2:
            if st.button("🎯 Get Hiring Summary"):
                # Generate hiring summary
                strong_hires = [r for r in st.session_state.results if r['hiring_recommendation'] == 'STRONG HIRE']
                hires = [r for r in st.session_state.results if r['hiring_recommendation'] == 'HIRE']
                conditional = [r for r in st.session_state.results if r['hiring_recommendation'] == 'CONDITIONAL HIRE']
                
                summary = f"""
                # 🎯 AI Hiring Summary Report
                
                **Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                **Total Candidates:** {len(st.session_state.results)}
                
                ## 🏆 Recommendations:
                - **Strong Hire:** {len(strong_hires)} candidates
                - **Hire:** {len(hires)} candidates  
                - **Conditional Hire:** {len(conditional)} candidates
                
                ## 🎯 Top 3 Candidates:
                """
                
                for i, result in enumerate(st.session_state.results[:3]):
                    summary += f"\n{i+1}. **{result['candidate_name']}** ({result['composite_score']:.1f}%)\n"
                    summary += f"   - Category: {result['ml_category']}\n"
                    summary += f"   - Recommendation: {result['hiring_recommendation']}\n"
                    summary += f"   - Key Skills: {', '.join(result['matched_skills'][:3])}\n"
                
                st.download_button(
                    label="📋 Download Hiring Summary",
                    data=summary,
                    file_name=f"hiring_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
    
    else:
        st.info("📊 Upload JD and resumes to see AI-powered results")

# Enhanced sidebar
with st.sidebar:
    st.markdown("## 🤖 AI System Status")
    
    if ml_model:
        st.success("✅ ML Model: ACTIVE")
        st.info(f"📊 Categories: {len(ml_model.get('categories', []))}")
        st.info("🎯 Features: Enhanced Analysis")
        
        with st.expander("🔍 Model Details"):
            categories = ml_model.get('categories', [])
            st.write("**Supported Categories:**")
            for i, cat in enumerate(categories[:10], 1):
                st.write(f"{i}. {cat}")
            if len(categories) > 10:
                st.write(f"... and {len(categories)-10} more")
    else:
        st.error("❌ ML Model: NOT LOADED")
        st.warning("Place 'resume_model.joblib' in project folder")
        st.info("🔧 Running in basic mode")
    
    st.markdown("## 💡 AI Features")
    
    features = [
        "🤖 ML Category Prediction",
        "🎯 Hiring Recommendations", 
        "💡 AI-Powered Insights",
        "📊 Confidence Scoring",
        "🔧 Skills Categorization",
        "📈 Composite Scoring",
        "🎨 Enhanced Visualizations"
    ]
    
    for feature in features:
        if ml_model:
            st.write(f"✅ {feature}")
        else:
            st.write(f"⏸️ {feature}")
    
    st.markdown("## 🚀 Quick Actions")
    
    if st.button("🔄 Reload ML Model"):
        st.cache_resource.clear()
        st.experimental_rerun()
    
    if st.button("🧹 Clear All Data"):
        st.session_state.jd_text = ""
        st.session_state.results = []
        st.success("✅ Data cleared!")
        st.experimental_rerun()
    
    st.markdown("## 📊 Session Stats")
    
    if st.session_state.results:
        total_analyzed = len(st.session_state.results)
        strong_hires = len([r for r in st.session_state.results if r['hiring_recommendation'] == 'STRONG HIRE'])
        avg_confidence = np.mean([r['ml_confidence'] for r in st.session_state.results if r['ml_confidence'] > 0])
        
        st.metric("Analyzed", total_analyzed)
        st.metric("Strong Hires", strong_hires)
        st.metric("Avg AI Confidence", f"{avg_confidence:.2f}" if avg_confidence else "N/A")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>🤖 AI-Powered Resume Matcher Pro</strong></p>
    <p>Enhanced with Machine Learning • Intelligent Hiring Recommendations • Advanced Analytics</p>
</div>
""", unsafe_allow_html=True)
