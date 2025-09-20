"""
Core utilities for Resume Relevance System
Handles text extraction, preprocessing, similarity calculations, and ML operations
"""

import os
import re
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import sqlite3
from datetime import datetime
import joblib
import pickle
import json

# Document processing
import pymupdf  # PyMuPDF for PDF
from docx import Document  # python-docx for DOCX

# NLP and ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Optional: Sentence Transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextExtractor:
    """Professional text extraction from multiple file formats"""
    
    @staticmethod
    def extract_from_pdf(file_path: str) -> str:
        """Extract text from PDF using PyMuPDF"""
        try:
            doc = pymupdf.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return TextExtractor._clean_text(text)
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return ""
    
    @staticmethod
    def extract_from_docx(file_path: str) -> str:
        """Extract text from DOCX using python-docx"""
        try:
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return TextExtractor._clean_text(text)
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            return ""
    
    @staticmethod
    def extract_from_txt(file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            return TextExtractor._clean_text(text)
        except Exception as e:
            logger.error(f"TXT extraction failed: {e}")
            return ""
    
    @staticmethod
    def extract_from_uploaded_file(uploaded_file) -> str:
        """Extract text from Streamlit uploaded file"""
        try:
            if uploaded_file.type == "application/pdf":
                # Save temporarily and extract
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                text = TextExtractor.extract_from_pdf(temp_path)
                os.remove(temp_path)
                return text
            
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                text = TextExtractor.extract_from_docx(temp_path)
                os.remove(temp_path)
                return text
            
            elif uploaded_file.type == "text/plain":
                return str(uploaded_file.read(), "utf-8")
            
            else:
                logger.warning(f"Unsupported file type: {uploaded_file.type}")
                return ""
                
        except Exception as e:
            logger.error(f"File extraction failed: {e}")
            return ""
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\-\+\#\(\)\/\@\&]', ' ', text)
        
        # Remove multiple dots
        text = re.sub(r'\.{2,}', '.', text)
        
        # Clean up spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

class TextPreprocessor:
    """Advanced text preprocessing for NLP tasks"""
    
    def __init__(self):
        self.stop_words = self._load_stop_words()
        
    def _load_stop_words(self) -> set:
        """Load common English stop words"""
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'would', 'have', 'had', 'been', 'this',
            'these', 'they', 'their', 'there', 'where', 'when', 'what', 'who'
        }
    
    def preprocess(self, text: str) -> str:
        """Comprehensive text preprocessing"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Normalize common terms
        text = self._normalize_tech_terms(text)
        
        # Remove stop words (optional for technical resumes)
        # words = [word for word in text.split() if word not in self.stop_words]
        # text = ' '.join(words)
        
        return text
    
    def _normalize_tech_terms(self, text: str) -> str:
        """Normalize technical terms and abbreviations"""
        normalizations = {
            'javascript': 'javascript js',
            'js': 'javascript',
            'python': 'python py',
            'artificial intelligence': 'ai artificial intelligence',
            'machine learning': 'ml machine learning',
            'c++': 'cpp c++',
            'c#': 'csharp c#',
            'node.js': 'nodejs node.js',
            'react.js': 'reactjs react',
            'vue.js': 'vuejs vue'
        }
        
        for original, replacement in normalizations.items():
            text = text.replace(original, replacement)
        
        return text
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract technical skills from text"""
        skills_patterns = {
            # Programming Languages
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby',
            'go', 'rust', 'kotlin', 'swift', 'scala', 'r', 'matlab', 'sql',
            
            # Web Technologies
            'html', 'css', 'react', 'angular', 'vue', 'nodejs', 'express', 'django',
            'flask', 'spring', 'laravel', 'bootstrap', 'jquery',
            
            # Databases
            'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'sqlite',
            'elasticsearch', 'cassandra', 'dynamodb',
            
            # Cloud & DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git',
            'terraform', 'ansible', 'linux',
            
            # Data Science & AI
            'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'keras',
            'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'jupyter',
            
            # Other Technologies
            'api', 'rest', 'graphql', 'microservices', 'agile', 'scrum'
        }
        
        found_skills = []
        text_lower = text.lower()
        
        for skill in skills_patterns:
            if skill in text_lower:
                found_skills.append(skill)
        
        return list(set(found_skills))

class SimilarityCalculator:
    """Calculate similarity between texts using multiple methods"""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.sentence_model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize similarity calculation models"""
        # TF-IDF Vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        # Sentence Transformers (if available)
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✅ Sentence Transformers model loaded")
            except Exception as e:
                logger.warning(f"Failed to load Sentence Transformers: {e}")
                self.sentence_model = None
        else:
            logger.info("Sentence Transformers not available - using TF-IDF only")
    
    def calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Calculate TF-IDF cosine similarity"""
        try:
            if not text1 or not text2:
                return 0.0
            
            # Fit and transform texts
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return float(similarity)
        
        except Exception as e:
            logger.error(f"TF-IDF similarity calculation failed: {e}")
            return 0.0
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using Sentence Transformers"""
        if not self.sentence_model:
            return self.calculate_tfidf_similarity(text1, text2)
        
        try:
            # Generate embeddings
            embeddings = self.sentence_model.encode([text1, text2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            return float(similarity)
        
        except Exception as e:
            logger.error(f"Semantic similarity calculation failed: {e}")
            return self.calculate_tfidf_similarity(text1, text2)
    
    def calculate_keyword_overlap(self, text1: str, text2: str) -> float:
        """Calculate simple keyword overlap similarity"""
        try:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
        
        except Exception as e:
            logger.error(f"Keyword overlap calculation failed: {e}")
            return 0.0
    
    def calculate_comprehensive_similarity(self, resume_text: str, jd_text: str) -> Dict[str, float]:
        """Calculate multiple similarity metrics"""
        
        # Preprocess texts
        preprocessor = TextPreprocessor()
        resume_clean = preprocessor.preprocess(resume_text)
        jd_clean = preprocessor.preprocess(jd_text)
        
        # Calculate different similarities
        similarities = {
            'tfidf_similarity': self.calculate_tfidf_similarity(resume_clean, jd_clean),
            'semantic_similarity': self.calculate_semantic_similarity(resume_clean, jd_clean),
            'keyword_overlap': self.calculate_keyword_overlap(resume_clean, jd_clean)
        }
        
        # Calculate weighted average
        weights = {'tfidf_similarity': 0.4, 'semantic_similarity': 0.4, 'keyword_overlap': 0.2}
        weighted_score = sum(similarities[key] * weights[key] for key in similarities)
        similarities['overall_similarity'] = weighted_score
        
        # Extract skills overlap
        resume_skills = preprocessor.extract_skills(resume_text)
        jd_skills = preprocessor.extract_skills(jd_text)
        
        if jd_skills:
            skills_overlap = len(set(resume_skills).intersection(set(jd_skills))) / len(jd_skills)
        else:
            skills_overlap = 0.0
        
        similarities['skills_overlap'] = skills_overlap
        
        return similarities

class DatabaseManager:
    """Manage SQLite database for storing results"""
    
    def __init__(self, db_path: str = "resume_relevance.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                jd_filename TEXT,
                resume_filename TEXT,
                candidate_name TEXT,
                overall_similarity REAL,
                tfidf_similarity REAL,
                semantic_similarity REAL,
                keyword_overlap REAL,
                skills_overlap REAL,
                similarity_percentage REAL,
                ranking INTEGER,
                notes TEXT
            )
        ''')
        
        # Create job descriptions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS job_descriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE,
                content TEXT,
                extracted_skills TEXT,
                created_date DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_results(self, results: List[Dict]) -> bool:
        """Save evaluation results to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for result in results:
                cursor.execute('''
                    INSERT INTO results 
                    (jd_filename, resume_filename, candidate_name, overall_similarity, 
                     tfidf_similarity, semantic_similarity, keyword_overlap, skills_overlap, 
                     similarity_percentage, ranking, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result.get('jd_filename', ''),
                    result.get('resume_filename', ''),
                    result.get('candidate_name', ''),
                    result.get('overall_similarity', 0.0),
                    result.get('tfidf_similarity', 0.0),
                    result.get('semantic_similarity', 0.0),
                    result.get('keyword_overlap', 0.0),
                    result.get('skills_overlap', 0.0),
                    result.get('similarity_percentage', 0.0),
                    result.get('ranking', 0),
                    result.get('notes', '')
                ))
            
            conn.commit()
            conn.close()
            return True
        
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return False
    
    def get_results(self, limit: int = 100) -> List[Dict]:
        """Retrieve results from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM results 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            results = []
            for row in rows:
                results.append(dict(zip(columns, row)))
            
            conn.close()
            return results
        
        except Exception as e:
            logger.error(f"Failed to retrieve results: {e}")
            return []
    
    def save_job_description(self, filename: str, content: str, skills: List[str]) -> bool:
        """Save job description to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO job_descriptions (filename, content, extracted_skills)
                VALUES (?, ?, ?)
            ''', (filename, content, json.dumps(skills)))
            
            conn.commit()
            conn.close()
            return True
        
        except Exception as e:
            logger.error(f"Failed to save job description: {e}")
            return False

class MLModelTrainer:
    """Train custom ML models for resume relevance scoring"""
    
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.label_encoder = None
        self.preprocessor = TextPreprocessor()
    
    def load_dataset(self, csv_path: str = "resume_dataset.csv") -> pd.DataFrame:
        """Load and prepare the resume dataset"""
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"✅ Loaded dataset with {len(df)} resumes")
            logger.info(f"📊 Categories: {df['Category'].nunique()} unique job categories")
            
            return df
        
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return None
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for ML training"""
        
        # Clean resume text
        df['Resume_Clean'] = df['Resume'].apply(self.preprocessor.preprocess)
        
        # Extract features using TF-IDF
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        X = self.vectorizer.fit_transform(df['Resume_Clean'])
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df['Category'])
        
        logger.info(f"✅ Feature matrix shape: {X.shape}")
        logger.info(f"✅ Number of classes: {len(self.label_encoder.classes_)}")
        
        return X.toarray(), y
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Train multiple ML models and return performance metrics"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define models
        models_config = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        results = {}
        
        for name, model in models_config.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            # Store model and results
            self.models[name] = model
            results[name] = {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'model': model
            }
            
            logger.info(f"✅ {name} - Train: {train_score:.3f}, Test: {test_score:.3f}")
        
        # Select best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
        self.best_model = results[best_model_name]['model']
        
        logger.info(f"🏆 Best model: {best_model_name}")
        
        return results
    
    def save_model(self, model_path: str = "models/custom_model.joblib"):
        """Save trained model and preprocessing components"""
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            model_data = {
                'model': self.best_model,
                'vectorizer': self.vectorizer,
                'label_encoder': self.label_encoder,
                'preprocessor': self.preprocessor
            }
            
            joblib.dump(model_data, model_path)
            logger.info(f"✅ Model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load_model(self, model_path: str = "models/custom_model.joblib") -> bool:
        """Load pre-trained model"""
        try:
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}")
                return False
            
            model_data = joblib.load(model_path)
            
            self.best_model = model_data['model']
            self.vectorizer = model_data['vectorizer']
            self.label_encoder = model_data['label_encoder']
            self.preprocessor = model_data['preprocessor']
            
            logger.info(f"✅ Model loaded from {model_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict_category(self, resume_text: str) -> Dict[str, Union[str, float]]:
        """Predict resume category using trained model"""
        
        if not hasattr(self, 'best_model') or self.best_model is None:
            return {'category': 'Unknown', 'confidence': 0.0}
        
        try:
            # Preprocess text
            clean_text = self.preprocessor.preprocess(resume_text)
            
            # Vectorize
            X = self.vectorizer.transform([clean_text])
            
            # Predict
            prediction = self.best_model.predict(X)[0]
            confidence = max(self.best_model.predict_proba(X)[0])
            
            # Decode category
            category = self.label_encoder.inverse_transform([prediction])[0]
            
            return {
                'category': category,
                'confidence': float(confidence)
            }
        
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {'category': 'Unknown', 'confidence': 0.0}

class ResumeAnalyzer:
    """Main class for comprehensive resume analysis"""
    
    def __init__(self):
        self.text_extractor = TextExtractor()
        self.similarity_calculator = SimilarityCalculator()
        self.db_manager = DatabaseManager()
        self.ml_trainer = MLModelTrainer()
        
        # Try to load pre-trained model
        self.ml_trainer.load_model()
    
    def analyze_batch(self, jd_text: str, resume_files: List, 
                     min_score_threshold: float = 0.0, 
                     mode: str = "tfidf") -> List[Dict]:
        """Analyze multiple resumes against a job description"""
        
        results = []
        
        logger.info(f"🔄 Analyzing {len(resume_files)} resumes...")
        
        for i, resume_file in enumerate(resume_files):
            try:
                # Extract resume text
                resume_text = self.text_extractor.extract_from_uploaded_file(resume_file)
                
                if not resume_text:
                    logger.warning(f"Could not extract text from {resume_file.name}")
                    continue
                
                # Calculate similarities
                if mode == "semantic" and SENTENCE_TRANSFORMERS_AVAILABLE:
                    similarities = self.similarity_calculator.calculate_comprehensive_similarity(resume_text, jd_text)
                    primary_score = similarities['semantic_similarity']
                else:
                    similarities = self.similarity_calculator.calculate_comprehensive_similarity(resume_text, jd_text)
                    primary_score = similarities['tfidf_similarity']
                
                # Convert to percentage
                similarity_percentage = primary_score * 100
                
                # Apply threshold filter
                if similarity_percentage < min_score_threshold:
                    continue
                
                # Extract candidate name (simple heuristic)
                candidate_name = self._extract_candidate_name(resume_text)
                
                # Get ML prediction if model is available
                ml_prediction = self.ml_trainer.predict_category(resume_text)
                
                # Create result
                result = {
                    'resume_filename': resume_file.name,
                    'candidate_name': candidate_name,
                    'resume_text': resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text,
                    'similarity_percentage': round(similarity_percentage, 2),
                    'ml_category': ml_prediction['category'],
                    'ml_confidence': round(ml_prediction['confidence'], 3),
                    'ranking': i + 1,  # Will be updated after sorting
                    **similarities
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error analyzing {resume_file.name}: {e}")
                continue
        
        # Sort by similarity score
        results.sort(key=lambda x: x['similarity_percentage'], reverse=True)
        
        # Update rankings
        for i, result in enumerate(results):
            result['ranking'] = i + 1
        
        logger.info(f"✅ Successfully analyzed {len(results)} resumes")
        
        return results
    
    def _extract_candidate_name(self, resume_text: str) -> str:
        """Extract candidate name from resume text (simple heuristic)"""
        lines = resume_text.strip().split('\n')[:5]  # Check first 5 lines
        
        for line in lines:
            line = line.strip()
            # Look for lines that might be names (2-4 words, title case, no numbers)
            if (2 <= len(line.split()) <= 4 and 
                line.istitle() and 
                not any(char.isdigit() for char in line) and
                len(line) > 5):
                return line
        
        return "Name not found"
    
    def get_similarity_insights(self, results: List[Dict]) -> Dict:
        """Generate insights from similarity analysis"""
        
        if not results:
            return {}
        
        scores = [r['similarity_percentage'] for r in results]
        
        insights = {
            'total_resumes': len(results),
            'average_score': round(np.mean(scores), 2),
            'highest_score': round(max(scores), 2),
            'lowest_score': round(min(scores), 2),
            'score_std': round(np.std(scores), 2),
            'top_candidate': results[0]['candidate_name'] if results else None,
            'high_scoring_count': len([s for s in scores if s >= 70]),
            'medium_scoring_count': len([s for s in scores if 50 <= s < 70]),
            'low_scoring_count': len([s for s in scores if s < 50])
        }
        
        return insights

# Utility functions for Streamlit integration
def initialize_components():
    """Initialize all components for the Streamlit app"""
    return ResumeAnalyzer()

def export_results_to_csv(results: List[Dict]) -> str:
    """Export results to CSV format"""
    try:
        df = pd.DataFrame(results)
        
        # Select relevant columns for export
        export_columns = [
            'ranking', 'resume_filename', 'candidate_name', 
            'similarity_percentage', 'ml_category', 'ml_confidence',
            'tfidf_similarity', 'semantic_similarity', 'skills_overlap'
        ]
        
        df_export = df[export_columns]
        csv_string = df_export.to_csv(index=False)
        
        return csv_string
    
    except Exception as e:
        logger.error(f"CSV export failed: {e}")
        return ""

# Main execution
if __name__ == "__main__":
    # Test the components
    analyzer = ResumeAnalyzer()
    
    # Test ML training if dataset is available
    if os.path.exists("resume_dataset.csv"):
        df = analyzer.ml_trainer.load_dataset()
        if df is not None:
            X, y = analyzer.ml_trainer.prepare_features(df)
            results = analyzer.ml_trainer.train_models(X, y)
            analyzer.ml_trainer.save_model()
            print("✅ ML model training completed")
    
    print("✅ Utils module loaded successfully")
