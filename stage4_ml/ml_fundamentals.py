"""
Stage 4: Machine Learning Fundamentals
Author: Jajitha
Course: Creating Your First Python Program - UST
Description: Scikit-learn, ML models and evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)
from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')


# =============================================
# 1. DATA PREPARATION
# =============================================

def create_gks_ml_dataset():
    """Create ML dataset for GKS prediction"""
    np.random.seed(42)
    n = 500
    
    gpa = np.random.uniform(3.0, 4.0, n)
    topik_level = np.random.choice(range(7), n)
    research_papers = np.random.choice(range(4), n)
    github_projects = np.random.randint(0, 15, n)
    has_lor = np.random.choice([0, 1], n)
    study_hours = np.random.uniform(2, 8, n)
    
    # Create realistic selection probability
    selection_score = (
        (gpa - 3.0) * 20 +
        topik_level * 8 +
        research_papers * 15 +
        github_projects * 2 +
        has_lor * 20 +
        study_hours * 2 +
        np.random.normal(0, 10, n)
    )
    
    threshold = np.percentile(selection_score, 75)
    selected = (selection_score >= threshold).astype(int)
    
    df = pd.DataFrame({
        'gpa': gpa,
        'topik_level': topik_level,
        'research_papers': research_papers,
        'github_projects': github_projects,
        'has_lor': has_lor,
        'study_hours': study_hours,
        'selected': selected
    })
    
    return df


# =============================================
# 2. MODEL TRAINING
# =============================================

class GKSPredictor:
    """
    ML model to predict GKS scholarship selection.
    
    Author: Jajitha
    Course: Creating Your First Python Program - UST
    """
    
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=5,
                random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            ),
            'SVM': SVC(
                probability=True,
                random_state=42
            )
        }
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        self.feature_names = None
    
    def prepare_data(self, df):
        """Prepare data for training"""
        feature_cols = [
            'gpa', 'topik_level', 'research_papers',
            'github_projects', 'has_lor', 'study_hours'
        ]
        self.feature_names = feature_cols
        
        X = df[feature_cols].values
        y = df['selected'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print(f"Selection rate: {y.mean():.2%}")
        
        return (
            X_train_scaled, X_test_scaled,
            y_train, y_test,
            X_train, X_test
        )
    
    def train_all_models(self, X_train, X_test, y_train, y_test):
        """Train and evaluate all models"""
        print("\nTraining models...")
        print("-"*50)
        
        best_score = 0
        
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=5, scoring='accuracy'
            )
            
            self.results[name] = {
                'accuracy': accuracy,
                'auc': auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'model': model,
                'y_pred': y_pred
            }
            
            print(f"{name}:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  AUC: {auc:.4f}")
            print(f"  CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            if accuracy > best_score:
                best_score = accuracy
                self.best_model = model
                self.best_model_name = name
        
        print(f"\n🏆 Best Model: {self.best_model_name}")
        print(f"   Accuracy: {best_score:.4f}")
    
    def get_feature_importance(self):
        """Get feature importance from best model"""
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            return feature_importance
        return None
    
    def predict_selection(self, applicant_data):
        """
        Predict GKS selection for new applicant
        
        Args:
            applicant_data: dict with applicant features
            
        Returns:
            prediction and probability
        """
        features = np.array([[
            applicant_data['gpa'],
            applicant_data['topik_level'],
            applicant_data['research_papers'],
            applicant_data['github_projects'],
            applicant_data['has_lor'],
            applicant_data['study_hours']
        ]])
        
        features_scaled = self.scaler.transform(features)
        prediction = self.best_model.predict(features_scaled)[0]
        probability = self.best_model.predict_proba(
            features_scaled
        )[0][1]
        
        return prediction, probability
    
    def visualize_results(self, X_test, y_test):
        """Visualize model comparison and results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Model comparison
        model_names = list(self.results.keys())
        accuracies = [r['accuracy'] for r in self.results.values()]
        aucs = [r['auc'] for r in self.results.values()]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0, 0].bar(
            x - width/2, accuracies,
            width, label='Accuracy',
            color='#2196F3', alpha=0.8
        )
        axes[0, 0].bar(
            x + width/2, aucs,
            width, label='AUC',
            color='#4CAF50', alpha=0.8
        )
        axes[0, 0].set_title(
            'Model Comparison',
            fontweight='bold'
        )
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(
            model_names,
            rotation=45,
            ha='right'
        )
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 1.1)
        
        # Confusion matrix for best model
        best_results = self.results[self.best_model_name]
        cm = confusion_matrix(y_test, best_results['y_pred'])
        
        im = axes[0, 1].imshow(cm, cmap='Blues')
        axes[0, 1].set_title(
            f'Confusion Matrix\n({self.best_model_name})',
            fontweight='bold'
        )
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')
        axes[0, 1].set_xticks([0, 1])
        axes[0, 1].set_yticks([0, 1])
        axes[0, 1].set_xticklabels(['Rejected', 'Selected'])
        axes[0, 1].set_yticklabels(['Rejected', 'Selected'])
        
        for i in range(2):
            for j in range(2):
                axes[0, 1].text(
                    j, i, str(cm[i, j]),
                    ha='center', va='center',
                    fontsize=16, fontweight='bold'
                )
        
        # Feature importance
        importance = self.get_feature_importance()
        if importance is not None:
            axes[1, 0].barh(
                importance['feature'],
                importance['importance'],
                color='#FF9800',
                alpha=0.8
            )
            axes[1, 0].set_title(
                'Feature Importance',
                fontweight='bold'
            )
            axes[1, 0].set_xlabel('Importance Score')
        
        # CV scores comparison
        cv_means = [r['cv_mean'] for r in self.results.values()]
        cv_stds = [r['cv_std'] for r in self.results.values()]
        
        axes[1, 1].barh(
            model_names,
            cv_means,
            xerr=cv_stds,
            color='#9C27B0',
            alpha=0.8,
            capsize=5
        )
        axes[1, 1].set_title(
            'Cross Validation Scores',
            fontweight='bold'
        )
        axes[1, 1].set_xlabel('CV Accuracy')
        
        plt.suptitle(
            'GKS Selection Prediction - ML Analysis\nAuthor: Jajitha',
            fontsize=14,
            fontweight='bold'
        )
        
        plt.tight_layout()
        plt.savefig('results/ml_results.png', dpi=150)
        plt.show()
        print("ML results visualization saved!")


# =============================================
# MAIN
# =============================================

if __name__ == "__main__":
    print("="*50)
    print("MACHINE LEARNING FUNDAMENTALS - STAGE 4")
    print("Author: Jajitha")
    print("="*50)
    
    # Create dataset
    print("\nCreating GKS dataset...")
    df = create_gks_ml_dataset()
    print(f"Dataset created: {df.shape}")
    
    # Initialize predictor
    predictor = GKSPredictor()
    
    # Prepare data
    print("\nPreparing data...")
    (
        X_train, X_test,
        y_train, y_test,
        X_train_raw, X_test_raw
    ) = predictor.prepare_data(df)
    
    # Train models
    predictor.train_all_models(
        X_train, X_test,
        y_train, y_test
    )
    
    # Best model report
    best = predictor.results[predictor.best_model_name]
    print(f"\nDetailed Report - {predictor.best_model_name}:")
    print(classification_report(
        y_test,
        best['y_pred'],
        target_names=['Rejected', 'Selected']
    ))
    
    # Predict for Jajitha's current profile
    print("\n" + "="*50)
    print("PREDICTION FOR JAJITHA - 2027 PROFILE")
    print("="*50)
    
    current_profile = {
        'gpa': 3.8,
        'topik_level': 2,
        'research_papers': 2,
        'github_projects': 10,
        'has_lor': 1,
        'study_hours': 6
    }
    
    prediction, probability = predictor.predict_selection(
        current_profile
    )
    
    print("\nProfile:")
    for key, value in current_profile.items():
        print(f"  {key}: {value}")
    
    print(f"\nPrediction: {'✅ SELECTED' if prediction == 1 else '❌ Rejected'}")
    print(f"Selection Probability: {probability:.2%}")
    
    if probability >= 0.7:
        print("🎉 Strong candidate for GKS 2027!")
    elif probability >= 0.5:
        print("💪 Good chance! Keep improving!")
    else:
        print("📚 Keep working hard! You'll get there!")
    
    # Visualize
    predictor.visualize_results(X_test, y_test)
    
    print("\n" + "="*50)
    print("Stage 4 Complete! Moving to Stage 5... 🚀")
    print("="*50)
