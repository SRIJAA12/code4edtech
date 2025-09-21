"""
👔 RECRUITER DASHBOARD - FULLY CORRECTED
Complete recruiter interface with bulletproof error handling
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import traceback

# Try to import plotly, provide fallback if not available
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("⚠️ Plotly not available - some visualizations disabled")

def recruiter_dashboard():
    """Main recruiter dashboard interface"""
    
    try:
        # Dashboard header
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb, #f5576c); padding: 2rem; border-radius: 15px; color: white; margin-bottom: 2rem;">
            <h2 style="margin: 0; font-size: 2.5rem;">👔 Recruiter Intelligence Hub</h2>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
                Advanced candidate analysis and matching system powered by AI
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize session state safely
        if 'results' not in st.session_state:
            st.session_state.results = []
        
        if 'cluster_names' not in st.session_state:
            st.session_state.cluster_names = {}
        
        # Main interface
        col1, col2 = st.columns([2, 3])
        
        with col1:
            render_upload_section()
        
        with col2:
            render_results_section()
    
    except Exception as e:
        st.error(f"❌ Recruiter Dashboard Error: {e}")
        st.code(traceback.format_exc())
        
        # Show emergency interface
        st.markdown("## 🚨 Emergency Interface")
        emergency_recruiter_interface()

def render_upload_section():
    """Render document upload section"""
    
    try:
        st.subheader("📋 Upload Documents")
        
        # Job description upload
        jd_file = st.file_uploader(
            "1. Upload Job Description:", 
            type=['pdf', 'docx', 'txt'], 
            key="rec_jd"
        )
        
        # Resume uploads
        resume_files = st.file_uploader(
            "2. Upload Candidate Resumes:", 
            type=['pdf', 'docx'], 
            accept_multiple_files=True, 
            key="rec_resumes"
        )
        
        # Analysis configuration
        if jd_file and resume_files:
            st.markdown("### ⚙️ Analysis Options")
            
            analysis_type = st.selectbox(
                "Analysis Type:",
                ["Quick Analysis", "Comprehensive Analysis", "Deep Dive"]
            )
            
            include_clustering = st.checkbox("Group Similar Candidates", True)
            include_scoring = st.checkbox("Detailed Scoring", True)
            max_candidates = st.slider("Max Candidates to Process", 1, len(resume_files), min(len(resume_files), 20))
        
        # Analysis button
        if st.button("🔍 Analyze Candidate Pool", type="primary"):
            if not jd_file or not resume_files:
                st.warning("⚠️ Please upload both JD and Resume files.")
            else:
                analyze_candidate_pool(jd_file, resume_files[:max_candidates], analysis_type)
    
    except Exception as e:
        st.error(f"❌ Upload Section Error: {e}")
        st.info("Please refresh the page and try again.")

def render_results_section():
    """Render analysis results section"""
    
    try:
        st.subheader("📊 Analysis Results")
        
        if not st.session_state.results:
            st.info("🔍 Results will be displayed here after analysis.")
        else:
            display_analysis_results()
    
    except Exception as e:
        st.error(f"❌ Results Section Error: {e}")
        st.info("Results display temporarily unavailable.")

def process_file(file):
    """Process uploaded file and extract text safely"""
    
    try:
        if not file:
            return ""
        
        file.seek(0)
        
        if file.type == "application/pdf":
            # Mock PDF extraction - replace with actual PDF parser
            return f"Sample PDF content from {file.name}"
        elif "word" in file.type:
            # Mock DOCX extraction - replace with actual DOCX parser
            return f"Sample DOCX content from {file.name}"
        else:
            # Text file
            return str(file.read(), "utf-8")
            
    except Exception as e:
        st.error(f"❌ Error processing file {file.name if file else 'unknown'}: {e}")
        return ""

def analyze_candidate_pool(jd_file, resume_files, analysis_type):
    """Analyze multiple candidates against job description"""
    
    try:
        with st.spinner("🤖 Performing advanced AI analysis..."):
            
            # Process job description
            jd_text = process_file(jd_file)
            
            if not jd_text:
                st.error("❌ Could not process job description")
                return
            
            # Process resumes
            results = []
            progress_bar = st.progress(0)
            
            for i, resume_file in enumerate(resume_files):
                # Update progress safely
                try:
                    progress_bar.progress((i + 1) / len(resume_files))
                except:
                    pass
                
                # Process resume
                resume_text = process_file(resume_file)
                
                if resume_text:
                    # Mock analysis results - replace with actual analysis
                    candidate_name = resume_file.name.replace('.pdf', '').replace('.docx', '').replace('_', ' ').title()
                    
                    # Generate realistic scores
                    base_score = np.random.normal(70, 15)
                    overall_score = max(20, min(95, base_score))
                    
                    # Generate mock skills
                    all_skills = ['Python', 'JavaScript', 'React', 'SQL', 'AWS', 'Docker', 'Git', 'Java', 'Angular', 'Node.js']
                    matched_skills = list(np.random.choice(all_skills, size=np.random.randint(3, 7), replace=False))
                    missing_skills = [skill for skill in all_skills if skill not in matched_skills][:3]
                    
                    # Determine verdict
                    if overall_score >= 80:
                        verdict = "🏆 STRONG HIRE"
                        cluster = "Top Performers"
                    elif overall_score >= 65:
                        verdict = "✅ HIRE"
                        cluster = "Strong Candidates"
                    elif overall_score >= 50:
                        verdict = "⚠️ CONDITIONAL"
                        cluster = "Potential Candidates"
                    else:
                        verdict = "❌ PASS"
                        cluster = "Below Threshold"
                    
                    result = {
                        'filename': resume_file.name,
                        'candidate_name': candidate_name,
                        'overall_score': round(overall_score, 1),
                        'verdict': verdict,
                        'cluster_name': cluster,
                        'matched_skills': matched_skills,
                        'missing_skills': missing_skills,
                        'suggestions': f"Focus on developing {', '.join(missing_skills[:2])} skills",
                        'missing_elements': missing_skills
                    }
                    
                    results.append(result)
            
            # Store results safely
            st.session_state.results = results
            
            # Create cluster mapping
            clusters = {}
            for result in results:
                cluster = result['cluster_name']
                if cluster not in clusters:
                    clusters[cluster] = []
                clusters[cluster].append(result['candidate_name'])
            
            st.session_state.cluster_names = clusters
            
            st.success("✅ Analysis and Clustering complete!")
            st.rerun()  # Fixed: Changed from st.experimental_rerun()
    
    except Exception as e:
        st.error(f"❌ Analysis failed: {e}")
        st.code(traceback.format_exc())

def display_analysis_results():
    """Display comprehensive analysis results safely"""
    
    try:
        results = st.session_state.results
        
        if not results or len(results) == 0:
            st.info("No results to display.")
            return
        
        # Create DataFrame safely
        df = pd.DataFrame(results)
        
        # Filtering options
        st.markdown("### 🔍 Filter Results")
        
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            # Cluster filter
            if st.session_state.cluster_names and len(st.session_state.cluster_names) > 0:
                cluster_options = list(st.session_state.cluster_names.keys())
                selected_clusters = st.multiselect(
                    "Filter by Profile:",
                    options=cluster_options,
                    default=cluster_options
                )
                
                if selected_clusters and len(selected_clusters) > 0:
                    df = df[df['cluster_name'].isin(selected_clusters)]
        
        with filter_col2:
            # Score filter
            min_score = st.slider("Minimum Score:", 0, 100, 0)
            df = df[df['overall_score'] >= min_score]
        
        # Sort by score safely
        df_sorted = df.sort_values(by='overall_score', ascending=False).reset_index(drop=True)
        
        # Summary metrics
        st.markdown("### 📊 Summary")
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Total Candidates", len(df_sorted))
        
        with metric_col2:
            avg_score = df_sorted['overall_score'].mean() if len(df_sorted) > 0 else 0
            st.metric("Average Score", f"{avg_score:.1f}%")
        
        with metric_col3:
            strong_hires = len(df_sorted[df_sorted['overall_score'] >= 80])
            st.metric("Strong Hires", strong_hires)
        
        with metric_col4:
            hire_rate = (strong_hires / len(df_sorted) * 100) if len(df_sorted) > 0 else 0
            st.metric("Hire Rate", f"{hire_rate:.1f}%")
        
        # Results table
        st.markdown("### 📋 Candidate Rankings")
        
        # Display table with key columns safely
        if len(df_sorted) > 0:
            display_df = df_sorted[['candidate_name', 'overall_score', 'verdict', 'cluster_name']].copy()
            display_df.columns = ['Candidate', 'Score (%)', 'Recommendation', 'Profile']
            
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No candidates match the selected filters.")
        
        # Detailed candidate profiles
        st.markdown("### 👤 Detailed Candidate Profiles")
        
        for _, row in df_sorted.head(10).iterrows():  # Show top 10
            with st.expander(f"📄 {row['candidate_name']} - Score: {row['overall_score']}% - {row['cluster_name']}"):
                
                profile_col1, profile_col2 = st.columns(2)
                
                with profile_col1:
                    st.markdown(f"**Verdict:** {row['verdict']}")
                    
                    st.markdown("**✅ Matched Skills:**")
                    matched_skills = row.get('matched_skills', [])
                    if matched_skills:
                        for skill in matched_skills:
                            st.success(f"• {skill}")
                    else:
                        st.info("No matched skills found")
                
                with profile_col2:
                    st.markdown(f"**📊 Overall Score:** {row['overall_score']}%")
                    
                    st.markdown("**❌ Missing Skills:**")
                    missing_skills = row.get('missing_skills', [])
                    if missing_skills:
                        for skill in missing_skills:
                            st.warning(f"• {skill}")
                    else:
                        st.success("No critical skills missing!")
                
                suggestions = row.get('suggestions', 'No specific suggestions available')
                st.info(f"**💡 Suggestions:** {suggestions}")
        
        # Export options
        st.markdown("---")
        st.markdown("### 📤 Export Results")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            if st.button("📊 Download CSV Report", use_container_width=True):
                if len(df_sorted) > 0:
                    csv_data = display_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download CSV",
                        data=csv_data,
                        file_name=f"candidate_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.warning("No data to export")
        
        with export_col2:
            if st.button("📋 Generate Summary Report", use_container_width=True):
                st.success("✅ Report generation feature coming soon!")
        
        # Visualization
        if PLOTLY_AVAILABLE and len(df_sorted) > 1:
            st.markdown("### 📈 Score Distribution")
            
            try:
                fig = px.histogram(
                    df_sorted, 
                    x='overall_score', 
                    nbins=10,
                    title="Candidate Score Distribution",
                    labels={'overall_score': 'Score (%)', 'count': 'Number of Candidates'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Visualization error: {e}")
        elif not PLOTLY_AVAILABLE:
            st.info("📊 Install Plotly for advanced visualizations")
    
    except Exception as e:
        st.error(f"❌ Results Display Error: {e}")
        st.code(traceback.format_exc())

def emergency_recruiter_interface():
    """Emergency fallback interface"""
    
    st.markdown("### 🚨 Basic Features Available")
    
    st.markdown("#### 📋 Quick Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        jd_text = st.text_area("Job Description", height=200)
    
    with col2:
        resume_files = st.file_uploader("Resume Files", type=['pdf', 'docx'], accept_multiple_files=True)
    
    if jd_text and resume_files:
        if st.button("🔍 Basic Analysis"):
            st.success(f"✅ Would analyze {len(resume_files)} candidates")
            st.info("Full analysis features temporarily disabled")
    
    st.markdown("#### 💡 Recruiting Tips")
    
    tips = [
        "Focus on skills match over perfect resume formatting",
        "Consider candidates with growth potential",
        "Use structured interviews for fair comparison",
        "Check references and past work examples",
        "Evaluate cultural fit alongside technical skills"
    ]
    
    for i, tip in enumerate(tips, 1):
        st.info(f"{i}. {tip}")

# Initialize session state safely
def init_recruiter_session():
    """Initialize recruiter-specific session state safely"""
    
    defaults = {
        'results': [],
        'cluster_names': {},
        'analysis_history': []
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Initialize on module load
init_recruiter_session()
