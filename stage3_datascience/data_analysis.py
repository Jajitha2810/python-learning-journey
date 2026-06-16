"""
Stage 3: Data Science Foundation
Author: Jajitha
Course: Creating Your First Python Program - UST
Description: NumPy, Pandas and Data Visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
import random


# =============================================
# 1. NUMPY FUNDAMENTALS
# =============================================

def demonstrate_numpy():
    """Demonstrate NumPy operations"""
    print("\n" + "="*50)
    print("1. NUMPY FUNDAMENTALS")
    print("="*50)
    
    # Arrays
    scores = np.array([85, 92, 78, 95, 88, 91, 87, 93])
    print(f"Test scores: {scores}")
    print(f"Mean: {np.mean(scores):.2f}")
    print(f"Std Dev: {np.std(scores):.2f}")
    print(f"Max: {np.max(scores)}")
    print(f"Min: {np.min(scores)}")
    
    # 2D arrays
    study_hours = np.array([
        [2, 3, 4, 2, 5],  # Week 1
        [3, 4, 3, 5, 4],  # Week 2
        [4, 5, 4, 6, 5],  # Week 3
        [5, 6, 5, 7, 6],  # Week 4
    ])
    
    print(f"\nStudy hours matrix shape: {study_hours.shape}")
    print(f"Total hours: {np.sum(study_hours)}")
    print(f"Weekly averages: {np.mean(study_hours, axis=1)}")
    print(f"Daily averages: {np.mean(study_hours, axis=0)}")
    
    # Array operations
    topik_progress = np.linspace(0, 100, 12)
    print(f"\nTOPIK progress over 12 months:")
    print(np.round(topik_progress, 1))


# =============================================
# 2. PANDAS DATA ANALYSIS
# =============================================

def create_gks_dataset():
    """Create sample GKS applicant dataset"""
    np.random.seed(42)
    n_applicants = 50
    
    data = {
        'applicant_id': [f'APP{i:03d}' for i in range(1, n_applicants + 1)],
        'name': [f'Applicant_{i}' for i in range(1, n_applicants + 1)],
        'country': np.random.choice(
            ['India', 'Vietnam', 'Indonesia', 'Thailand', 'Malaysia'],
            n_applicants
        ),
        'gpa': np.round(np.random.uniform(3.0, 4.0, n_applicants), 2),
        'topik_level': np.random.choice([0, 1, 2, 3, 4, 5, 6], n_applicants),
        'research_papers': np.random.choice([0, 1, 2, 3], n_applicants),
        'github_projects': np.random.randint(0, 15, n_applicants),
        'has_lor': np.random.choice([True, False], n_applicants),
        'track': np.random.choice(['General', 'R&D'], n_applicants),
        'result': np.random.choice(
            ['Selected', 'Rejected', 'Pending'],
            n_applicants,
            p=[0.2, 0.6, 0.2]
        )
    }
    
    return pd.DataFrame(data)


def demonstrate_pandas():
