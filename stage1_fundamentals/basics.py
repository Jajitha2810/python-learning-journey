"""
Stage 1: Python Fundamentals
Author: Jajitha
Course: Creating Your First Python Program - UST
Description: Core Python concepts from scratch
"""

# =============================================
# 1. VARIABLES AND DATA TYPES
# =============================================

def demonstrate_variables():
    """Demonstrate Python variable types"""
    print("\n" + "="*50)
    print("1. VARIABLES AND DATA TYPES")
    print("="*50)
    
    # Basic data types
    name = "Jajitha"
    age = 28
    gpa = 3.8
    is_student = True
    
    print(f"Name: {name} (type: {type(name).__name__})")
    print(f"Age: {age} (type: {type(age).__name__})")
    print(f"GPA: {gpa} (type: {type(gpa).__name__})")
    print(f"Is Student: {is_student} (type: {type(is_student).__name__})")
    
    # Collections
    skills = ["Python", "NLP", "Machine Learning", "Korean"]
    goals = ("GKS Scholarship", "Korea", "Research")
    profile = {
        "name": name,
        "age": age,
        "skills": skills,
        "dream": "Study in Korea"
    }
    unique_languages = {"Python", "Korean", "English", "Telugu"}
    
    print(f"\nSkills list: {skills}")
    print(f"Goals tuple: {goals}")
    print(f"Profile dict: {profile}")
    print(f"Languages set: {unique_languages}")


# =============================================
# 2. CONTROL FLOW
# =============================================

def demonstrate_control_flow():
    """Demonstrate if/else and loops"""
    print("\n" + "="*50)
    print("2. CONTROL FLOW")
    print("="*50)
    
    # If/elif/else
    gks_attempts = 3
    
    if gks_attempts == 1:
        print("First attempt - Learning the process!")
    elif gks_attempts == 2:
        print("Second attempt - Getting stronger!")
    elif gks_attempts == 3:
        print("Third attempt - This is the one! 🇰🇷")
    else:
        print("Never giving up!")
    
    # For loop
    print("\nSkills to master for GKS:")
    skills = ["TOPIK", "Research Paper", "Python", "NLP"]
    for i, skill in enumerate(skills, 1):
        print(f"  {i}. {skill}")
    
    # While loop
    print("\nCountdown to Korea:")
    count = 5
    while count > 0:
        print(f"  {count}...")
        count -= 1
    print("  🚀 Let's go!")
    
    # List comprehension
    scores = [85, 92, 78, 95, 88, 91]
    high_scores = [s for s in scores if s >= 90]
    print(f"\nAll scores: {scores}")
    print(f"High scores (90+): {high_scores}")


# =============================================
# 3. FUNCTIONS
# =============================================

def calculate_gks_readiness(
    topik_level,
    research_papers,
    github_projects,
    professor_lor
):
    """
    Calculate GKS application readiness score
    
    Args:
        topik_level: TOPIK level achieved (0-6)
        research_papers: Number of published papers
        github_projects: Number of GitHub projects
        professor_lor: Has professor LOR (True/False)
    
    Returns:
        readiness_score: Score out of 100
        feedback: List of improvement suggestions
    """
    score = 0
    feedback = []
    
    # TOPIK scoring (max 30 points)
    topik_score = min(topik_level * 5, 30)
    score += topik_score
    if topik_level == 0:
        feedback.append("Start TOPIK preparation!")
    
    # Research papers (max 30 points)
    paper_score = min(research_papers * 10, 30)
    score += paper_score
    if research_papers == 0:
        feedback.append("Work on research publications!")
    
    # GitHub projects (max 20 points)
    github_score = min(github_projects * 4, 20)
    score += github_score
    if github_projects < 3:
        feedback.append("Add more GitHub projects!")
    
    # Professor LOR (20 points)
    if professor_lor:
        score += 20
    else:
        feedback.append("Secure professor LOR!")
    
    return score, feedback


def demonstrate_functions():
    """Demonstrate Python functions"""
    print("\n" + "="*50)
    print("3. FUNCTIONS")
    print("="*50)
    
    # Current readiness
    score, feedback = calculate_gks_readiness(
        topik_level=0,
        research_papers=0,
        github_projects=5,
        professor_lor=True
    )
    
    print(f"Current GKS Readiness Score: {score}/100")
    print("\nAreas to improve:")
    for item in feedback:
        print(f"  → {item}")
    
    # Target readiness
    target_score, _ = calculate_gks_readiness(
        topik_level=4,
        research_papers=2,
        github_projects=10,
        professor_lor=True
    )
    print(f"\nTarget Score (2027): {target_score}/100")


# =============================================
# 4. FILE HANDLING
# =============================================

def demonstrate_file_handling():
    """Demonstrate file read/write operations"""
    print("\n" + "="*50)
    print("4. FILE HANDLING")
    print("="*50)
    
    # Write to file
    goals = [
        "Pass TOPIK Level 2 by December 2026",
        "Publish 2 research papers by 2027",
        "Complete GitHub portfolio",
        "Secure GKS scholarship 2027",
        "Move to Korea and study AI"
    ]
    
    with open('goals.txt', 'w') as f:
        f.write("MY GOALS - Jajitha\n")
        f.write("="*30 + "\n")
        for i, goal in enumerate(goals, 1):
            f.write(f"{i}. {goal}\n")
    
    print("Goals written to goals.txt!")
    
    # Read from file
    print("\nReading goals from file:")
    with open('goals.txt', 'r') as f:
        content = f.read()
    print(content)


# =============================================
# 5. EXCEPTION HANDLING
# =============================================

def demonstrate_exceptions():
    """Demonstrate error handling"""
    print("\n" + "="*50)
    print("5. EXCEPTION HANDLING")
    print("="*50)
    
    # Try/except
    def safe_divide(a, b):
        try:
            result = a / b
            return result
        except ZeroDivisionError:
            print("Error: Cannot divide by zero!")
            return None
        except TypeError:
            print("Error: Please provide numbers!")
            return None
        finally:
            print("Division operation attempted.")
    
    print(f"10 / 2 = {safe_divide(10, 2)}")
    print(f"10 / 0 = {safe_divide(10, 0)}")
    
    # Custom exception
    class GKSApplicationError(Exception):
        """Custom exception for GKS application errors"""
        pass
    
    def validate_topik_level(level):
        if not 0 <= level <= 6:
            raise GKSApplicationError(
                f"Invalid TOPIK level: {level}. Must be 0-6!"
            )
        return True
    
    try:
        validate_topik_level(8)
    except GKSApplicationError as e:
        print(f"\nGKS Error caught: {e}")


# =============================================
# MAIN
# =============================================

if __name__ == "__main__":
    print("="*50)
    print("PYTHON FUNDAMENTALS - STAGE 1")
    print("Author: Jajitha")
    print("="*50)
    
    demonstrate_variables()
    demonstrate_control_flow()
    demonstrate_functions()
    demonstrate_file_handling()
    demonstrate_exceptions()
    
    print("\n" + "="*50)
    print("Stage 1 Complete! Moving to Stage 2... 🚀")
    print("="*50)
