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
