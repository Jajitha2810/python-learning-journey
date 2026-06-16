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
    """Demonstrate Pandas operations"""
    print("\n" + "="*50)
    print("2. PANDAS DATA ANALYSIS")
    print("="*50)
    
    df = create_gks_dataset()
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    print(f"\nDataset info:")
    print(df.dtypes)
    
    print(f"\nBasic statistics:")
    print(df[['gpa', 'topik_level', 'research_papers', 'github_projects']].describe())
    
    # Filtering
    selected = df[df['result'] == 'Selected']
    print(f"\nSelected applicants: {len(selected)}")
    print(f"Average GPA of selected: {selected['gpa'].mean():.2f}")
    print(f"Average TOPIK of selected: {selected['topik_level'].mean():.2f}")
    
    # Groupby analysis
    print(f"\nResults by track:")
    print(df.groupby('track')['result'].value_counts())
    
    print(f"\nAverage stats by result:")
    print(df.groupby('result')[
        ['gpa', 'topik_level', 'research_papers']
    ].mean().round(2))
    
    # Correlation
    print(f"\nCorrelation with selection:")
    df['is_selected'] = (df['result'] == 'Selected').astype(int)
    correlations = df[[
        'gpa', 'topik_level',
        'research_papers', 'github_projects',
        'is_selected'
    ]].corr()['is_selected'].drop('is_selected')
    print(correlations.sort_values(ascending=False))
    
    return df


# =============================================
# 3. DATA VISUALIZATION
# =============================================

def demonstrate_visualization(df):
    """Create comprehensive data visualizations"""
    print("\n" + "="*50)
    print("3. DATA VISUALIZATION")
    print("="*50)
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    
    # 1. Results distribution
    ax1 = fig.add_subplot(gs[0, 0])
    result_counts = df['result'].value_counts()
    colors = ['#4CAF50', '#F44336', '#FF9800']
    ax1.pie(
        result_counts.values,
        labels=result_counts.index,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90
    )
    ax1.set_title('GKS Results Distribution', fontweight='bold')
    
    # 2. GPA distribution
    ax2 = fig.add_subplot(gs[0, 1])
    for result, color in zip(
        ['Selected', 'Rejected'],
        ['#4CAF50', '#F44336']
    ):
        subset = df[df['result'] == result]['gpa']
        ax2.hist(
            subset,
            alpha=0.7,
            color=color,
            label=result,
            bins=10
        )
    ax2.set_title('GPA Distribution by Result', fontweight='bold')
    ax2.set_xlabel('GPA')
    ax2.set_ylabel('Count')
    ax2.legend()
    
    # 3. TOPIK levels
    ax3 = fig.add_subplot(gs[0, 2])
    topik_counts = df.groupby(['topik_level', 'result']).size().unstack(fill_value=0)
    topik_counts.plot(
        kind='bar',
        ax=ax3,
        color=['#FF9800', '#F44336', '#4CAF50']
    )
    ax3.set_title('TOPIK Level vs Result', fontweight='bold')
    ax3.set_xlabel('TOPIK Level')
    ax3.set_ylabel('Count')
    ax3.tick_params(axis='x', rotation=0)
    
    # 4. Research papers impact
    ax4 = fig.add_subplot(gs[1, 0])
    paper_counts = df.groupby(['research_papers', 'result']).size().unstack(fill_value=0)
    paper_counts.plot(
        kind='bar',
        ax=ax4,
        color=['#FF9800', '#F44336', '#4CAF50']
    )
    ax4.set_title('Research Papers vs Result', fontweight='bold')
    ax4.set_xlabel('Number of Papers')
    ax4.set_ylabel('Count')
    ax4.tick_params(axis='x', rotation=0)
    
    # 5. Country distribution
    ax5 = fig.add_subplot(gs[1, 1])
    country_counts = df['country'].value_counts()
    ax5.barh(
        country_counts.index,
        country_counts.values,
        color='#2196F3'
    )
    ax5.set_title('Applicants by Country', fontweight='bold')
    ax5.set_xlabel('Count')
    
    # 6. GPA vs TOPIK scatter
    ax6 = fig.add_subplot(gs[1, 2])
    colors_map = {
        'Selected': '#4CAF50',
        'Rejected': '#F44336',
        'Pending': '#FF9800'
    }
    for result, color in colors_map.items():
        subset = df[df['result'] == result]
        ax6.scatter(
            subset['gpa'],
            subset['topik_level'],
            c=color,
            label=result,
            alpha=0.7,
            s=50
        )
    ax6.set_title('GPA vs TOPIK Level', fontweight='bold')
    ax6.set_xlabel('GPA')
    ax6.set_ylabel('TOPIK Level')
    ax6.legend()
    
    plt.suptitle(
        'GKS Application Data Analysis - Jajitha',
        fontsize=16,
        fontweight='bold',
        y=1.02
    )
    
    plt.tight_layout()
    plt.savefig('results/data_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Visualization saved!")


# =============================================
# 4. STUDY PROGRESS TRACKER
# =============================================

def track_study_progress():
    """Track and visualize study progress"""
    print("\n" + "="*50)
    print("4. STUDY PROGRESS TRACKER")
    print("="*50)
    
    # Generate study data
    dates = pd.date_range(
        start='2026-06-14',
        periods=30,
        freq='D'
    )
    
    study_data = pd.DataFrame({
        'date': dates,
        'topik_hours': np.random.uniform(0.5, 3, 30),
        'python_hours': np.random.uniform(0.5, 2, 30),
        'research_hours': np.random.uniform(0, 2, 30),
        'korean_vocab': np.random.randint(10, 50, 30)
    })
    
    study_data['total_hours'] = (
        study_data['topik_hours'] +
        study_data['python_hours'] +
        study_data['research_hours']
    )
    
    study_data['cumulative_vocab'] = (
        study_data['korean_vocab'].cumsum()
    )
    
    print(f"Total study hours: {study_data['total_hours'].sum():.1f}")
    print(f"Average daily hours: {study_data['total_hours'].mean():.1f}")
    print(f"Total Korean vocab learned: {study_data['cumulative_vocab'].iloc[-1]}")
    
    # Plot progress
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Daily hours
    axes[0, 0].fill_between(
        study_data['date'],
        study_data['total_hours'],
        alpha=0.7,
        color='#2196F3'
    )
    axes[0, 0].set_title('Daily Study Hours', fontweight='bold')
    axes[0, 0].set_ylabel('Hours')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Subject breakdown
    axes[0, 1].stackplot(
        study_data['date'],
        study_data['topik_hours'],
        study_data['python_hours'],
        study_data['research_hours'],
        labels=['TOPIK', 'Python', 'Research'],
        colors=['#4CAF50', '#2196F3', '#FF9800'],
        alpha=0.8
    )
    axes[0, 1].set_title('Study Hours by Subject', fontweight='bold')
    axes[0, 1].legend(loc='upper left')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Cumulative vocab
    axes[1, 0].plot(
        study_data['date'],
        study_data['cumulative_vocab'],
        color='#9C27B0',
        linewidth=2
    )
    axes[1, 0].fill_between(
        study_data['date'],
        study_data['cumulative_vocab'],
        alpha=0.3,
        color='#9C27B0'
    )
    axes[1, 0].set_title('Cumulative Korean Vocabulary', fontweight='bold')
    axes[1, 0].set_ylabel('Words Learned')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Weekly summary
    weekly = study_data.resample('W', on='date')[
        'total_hours'
    ].sum()
    axes[1, 1].bar(
        range(len(weekly)),
        weekly.values,
        color='#F44336',
        alpha=0.8
    )
    axes[1, 1].set_title('Weekly Study Hours', fontweight='bold')
    axes[1, 1].set_ylabel('Total Hours')
    axes[1, 1].set_xlabel('Week')
    
    plt.suptitle(
        'Study Progress Tracker - Jajitha (GKS 2027)',
        fontsize=14,
        fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig('results/study_progress.png', dpi=150)
    plt.show()
    print("Study progress chart saved!")
    
    return study_data


# =============================================
# MAIN
# =============================================

if __name__ == "__main__":
    print("="*50)
    print("DATA SCIENCE FOUNDATION - STAGE 3")
    print("Author: Jajitha")
    print("="*50)
    
    demonstrate_numpy()
    df = demonstrate_pandas()
    demonstrate_visualization(df)
    study_data = track_study_progress()
    
    print("\n" + "="*50)
    print("Stage 3 Complete! Moving to Stage 4... 🚀")
    print("="*50)
