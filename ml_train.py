"""
Custom ML Model Training Script for Resume Relevance System
Trains models using the real resume dataset
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Import our utils
from utils import TextPreprocessor, DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedMLTrainer:
    """Enhanced ML trainer with comprehensive model evaluation"""
    
    def __init__(self, dataset_path: str = "resume_dataset.csv"):
        self.dataset_path = dataset_path
        self.preprocessor = TextPreprocessor()
        self.vectorizer = None
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.best_model = None
        self.best_model_name = ""
        
    def load_and_explore_dataset(self) -> pd.DataFrame:
        """Load dataset and perform exploratory data analysis"""
        
        try:
            df = pd.read_csv(self.dataset_path)
            logger.info(f"✅ Dataset loaded: {len(df)} resumes")
            
            # Basic statistics
            logger.info(f"📊 Categories: {df['Category'].nunique()}")
            logger.info(f"📊 Category distribution:\n{df['Category'].value_counts()}")
            
            # Data quality checks
            null_count = df['Resume'].isnull().sum()
            logger.info(f"📊 Null resumes: {null_count}")
            
            # Text length analysis
            df['text_length'] = df['Resume'].str.len()
            logger.info(f"📊 Average resume length: {df['text_length'].mean():.0f} characters")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {e}")
            return None
    
    def prepare_enhanced_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare enhanced features with multiple vectorization strategies"""
        
        logger.info("🔧 Preparing enhanced features...")
        
        # Clean text
        df['Resume_Clean'] = df['Resume'].apply(self.preprocessor.preprocess)
        
        # Multiple vectorization strategies
        
        # 1. TF-IDF with different n-grams
        tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3),  # Include trigrams
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        
        X_tfidf = tfidf_vectorizer.fit_transform(df['Resume_Clean'])
        
        # 2. Additional feature engineering
        feature_names = list(tfidf_vectorizer.get_feature_names_out())
        
        # Add length-based features
        length_features = np.array([
            df['text_length'].values,
            df['Resume_Clean'].str.split().str.len().values,  # word count
            df['Resume_Clean'].str.count('\.').values,        # sentence count
        ]).T
        
        feature_names.extend(['text_length', 'word_count', 'sentence_count'])
        
        # Combine features
        X_combined = np.hstack([X_tfidf.toarray(), length_features])
        
        # Encode labels
        y = self.label_encoder.fit_transform(df['Category'])
        
        self.vectorizer = tfidf_vectorizer  # Save for later use
        
        logger.info(f"✅ Feature matrix: {X_combined.shape}")
        logger.info(f"✅ Classes: {len(self.label_encoder.classes_)}")
        
        return X_combined, y, feature_names
    
    def train_comprehensive_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict]:
        """Train multiple models with hyperparameter tuning"""
        
        logger.info("🚀 Training comprehensive models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Model configurations with hyperparameter grids
        models_config = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l2'],
                    'solver': ['liblinear', 'lbfgs']
                }
            },
            'SVM': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                }
            },
            'Naive Bayes': {
                'model': MultinomialNB(),
                'params': {
                    'alpha': [0.1, 0.5, 1.0, 2.0]
                }
            }
        }
        
        results = {}
        
        for name, config in models_config.items():
            logger.info(f"🔧 Training {name}...")
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                config['model'],
                config['params'],
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            
            # Fit model
            grid_search.fit(X_train, y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            
            # Evaluate
            train_score = best_model.score(X_train, y_train)
            test_score = best_model.score(X_test, y_test)
            
            # Cross-validation score
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
            
            # Store results
            results[name] = {
                'model': best_model,
                'best_params': grid_search.best_params_,
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': best_model.predict(X_test),
                'y_test': y_test
            }
            
            self.models[name] = best_model
            
            logger.info(f"✅ {name} - Test: {test_score:.3f}, CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
        
        # Select best model based on test accuracy
        self.best_model_name = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
        self.best_model = results[self.best_model_name]['model']
        
        logger.info(f"🏆 Best model: {self.best_model_name} ({results[self.best_model_name]['test_accuracy']:.3f})")
        
        return results
    
    def generate_model_report(self, results: Dict[str, Dict]) -> str:
        """Generate comprehensive model evaluation report"""
        
        report = "# 📊 Model Training Report\n\n"
        report += f"**Dataset:** {self.dataset_path}\n"
        report += f"**Number of categories:** {len(self.label_encoder.classes_)}\n"
        report += f"**Categories:** {', '.join(self.label_encoder.classes_)}\n\n"
        
        report += "## 🏆 Model Performance Comparison\n\n"
        report += "| Model | Test Accuracy | CV Mean±Std | Best Parameters |\n"
        report += "|-------|---------------|-------------|------------------|\n"
        
        for name, result in results.items():
            report += f"| {name} | {result['test_accuracy']:.3f} | {result['cv_mean']:.3f}±{result['cv_std']:.3f} | {result['best_params']} |\n"
        
        report += f"\n**🥇 Best Model:** {self.best_model_name}\n\n"
        
        # Detailed classification report for best model
        best_result = results[self.best_model_name]
        y_pred = best_result['predictions']
        y_test = best_result['y_test']
        
        # Convert back to category names
        y_test_names = self.label_encoder.inverse_transform(y_test)
        y_pred_names = self.label_encoder.inverse_transform(y_pred)
        
        report += "## 📋 Detailed Classification Report (Best Model)\n\n"
        report += "```\n"
        report += classification_report(y_test_names, y_pred_names)
        report += "```\n"
        
        return report
    
    def save_model_and_components(self, model_dir: str = "models"):
        """Save trained model and all components"""
        
        try:
            # Create model directory
            Path(model_dir).mkdir(exist_ok=True)
            
            # Save model components
            model_data = {
                'model': self.best_model,
                'model_name': self.best_model_name,
                'vectorizer': self.vectorizer,
                'label_encoder': self.label_encoder,
                'preprocessor': self.preprocessor,
                'categories': list(self.label_encoder.classes_)
            }
            
            model_path = Path(model_dir) / "enhanced_resume_model.joblib"
            joblib.dump(model_data, model_path)
            
            logger.info(f"✅ Model saved to {model_path}")
            
            # Save model metadata
            metadata = {
                'model_type': self.best_model_name,
                'categories': list(self.label_encoder.classes_),
                'n_categories': len(self.label_encoder.classes_),
                'vectorizer_features': self.vectorizer.max_features if self.vectorizer else 0,
                'created_date': pd.Timestamp.now().isoformat()
            }
            
            metadata_path = Path(model_dir) / "model_metadata.json"
            pd.Series(metadata).to_json(metadata_path, orient='index', indent=2)
            
            return str(model_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
            return None
    
    def create_visualizations(self, results: Dict[str, Dict], output_dir: str = "plots"):
        """Create visualization plots for model performance"""
        
        try:
            Path(output_dir).mkdir(exist_ok=True)
            
            # 1. Model accuracy comparison
            plt.figure(figsize=(12, 6))
            
            models = list(results.keys())
            test_scores = [results[name]['test_accuracy'] for name in models]
            cv_scores = [results[name]['cv_mean'] for name in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            plt.bar(x - width/2, test_scores, width, label='Test Accuracy', alpha=0.8)
            plt.bar(x + width/2, cv_scores, width, label='CV Mean', alpha=0.8)
            
            plt.xlabel('Models')
            plt.ylabel('Accuracy')
            plt.title('Model Performance Comparison')
            plt.xticks(x, models, rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(Path(output_dir) / 'model_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Confusion matrix for best model
            best_result = results[self.best_model_name]
            y_test = best_result['y_test']
            y_pred = best_result['predictions']
            
            cm = confusion_matrix(y_test, y_pred)
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.label_encoder.classes_,
                       yticklabels=self.label_encoder.classes_)
            plt.title(f'Confusion Matrix - {self.best_model_name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.savefig(Path(output_dir) / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Visualizations saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create visualizations: {e}")
    
    def run_complete_training(self) -> bool:
        """Run the complete training pipeline"""
        
        logger.info("🚀 Starting comprehensive ML training pipeline...")
        
        try:
            # 1. Load and explore dataset
            df = self.load_and_explore_dataset()
            if df is None:
                return False
            
            # 2. Prepare features
            X, y, feature_names = self.prepare_enhanced_features(df)
            
            # 3. Train models
            results = self.train_comprehensive_models(X, y)
            
            # 4. Generate report
            report = self.generate_model_report(results)
            with open("model_training_report.md", "w") as f:
                f.write(report)
            
            # 5. Save model
            model_path = self.save_model_and_components()
            
            # 6. Create visualizations
            self.create_visualizations(results)
            
            logger.info("✅ Complete training pipeline finished successfully!")
            logger.info(f"📊 Report saved: model_training_report.md")
            logger.info(f"🤖 Model saved: {model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Training pipeline failed: {e}")
            return False

def main():
    """Main training function"""
    
    print("🎯 Enhanced ML Training for Resume Relevance System")
    print("=" * 60)
    
    # Check if dataset exists
    dataset_path = "resume_dataset.csv"
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("Please ensure the resume dataset CSV file is in the current directory")
        return
    
    # Initialize trainer
    trainer = EnhancedMLTrainer(dataset_path)
    
    # Run training
    success = trainer.run_complete_training()
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("✅ Model ready for use in Streamlit app")
    else:
        print("\n❌ Training failed. Check logs for details.")

if __name__ == "__main__":
    main()
