"""
Stage 2: Intermediate Python - OOP Concepts
Author: Jajitha
Course: Creating Your First Python Program - UST
Description: Object Oriented Programming in Python
"""

from datetime import datetime
from abc import ABC, abstractmethod


# =============================================
# 1. CLASSES AND OBJECTS
# =============================================

class ScholarshipApplication:
    """
    Represents a scholarship application.
    Demonstrates basic OOP concepts.
    """
    
    # Class variable
    total_applications = 0
    
    def __init__(
        self,
        applicant_name,
        scholarship_name,
        university,
        year
    ):
        # Instance variables
        self.applicant_name = applicant_name
        self.scholarship_name = scholarship_name
        self.university = university
        self.year = year
        self.status = "Pending"
        self.documents = []
        self.created_at = datetime.now()
        
        # Increment class variable
        ScholarshipApplication.total_applications += 1
        self.application_id = (
            ScholarshipApplication.total_applications
        )
    
    def add_document(self, document):
        """Add document to application"""
        self.documents.append(document)
        print(f"Document added: {document}")
    
    def submit(self):
        """Submit the application"""
        if len(self.documents) < 3:
            print("❌ Need at least 3 documents!")
            return False
        self.status = "Submitted"
        print(f"✅ Application {self.application_id} submitted!")
        return True
    
    def __str__(self):
        return (
            f"Application #{self.application_id}\n"
            f"Applicant: {self.applicant_name}\n"
            f"Scholarship: {self.scholarship_name}\n"
            f"University: {self.university}\n"
            f"Year: {self.year}\n"
            f"Status: {self.status}\n"
            f"Documents: {len(self.documents)}"
        )
    
    def __repr__(self):
        return (
            f"ScholarshipApplication("
            f"'{self.applicant_name}', "
            f"'{self.scholarship_name}', "
            f"{self.year})"
        )


# =============================================
# 2. INHERITANCE
# =============================================

class BaseResearcher(ABC):
    """Abstract base class for researchers"""
    
    def __init__(self, name, field):
        self.name = name
        self.field = field
        self.publications = []
    
    @abstractmethod
    def conduct_research(self):
        """Abstract method - must be implemented"""
        pass
    
    def publish_paper(self, title, journal):
        """Publish a research paper"""
        paper = {
            'title': title,
            'journal': journal,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'author': self.name
        }
        self.publications.append(paper)
        print(f"📄 Paper published: {title}")
        return paper
    
    def get_publication_count(self):
        return len(self.publications)


class NLPResearcher(BaseResearcher):
    """
    NLP Researcher class inheriting from BaseResearcher.
    Demonstrates single inheritance.
    """
    
    def __init__(self, name, specialization):
        super().__init__(name, "Natural Language Processing")
        self.specialization = specialization
        self.models_trained = []
        self.datasets_used = []
    
    def conduct_research(self):
        """Implement abstract method"""
        print(f"\n🔬 {self.name} conducting NLP research...")
        print(f"   Specialization: {self.specialization}")
        print(f"   Field: {self.field}")
    
    def train_model(self, model_name, dataset):
        """Train an NLP model"""
        self.models_trained.append(model_name)
        self.datasets_used.append(dataset)
        print(f"🤖 Model trained: {model_name} on {dataset}")
    
    def get_research_summary(self):
        """Get research summary"""
        return {
            'researcher': self.name,
            'field': self.field,
            'specialization': self.specialization,
            'publications': self.get_publication_count(),
            'models_trained': len(self.models_trained)
        }


class HCIResearcher(BaseResearcher):
    """
    HCI Researcher class.
    Demonstrates multiple specializations.
    """
    
    def __init__(self, name):
        super().__init__(
            name,
            "Human Computer Interaction"
        )
        self.user_studies = []
        self.prototypes = []
    
    def conduct_research(self):
        """Implement abstract method"""
        print(f"\n🔬 {self.name} conducting HCI research...")
        print(f"   Field: {self.field}")
    
    def conduct_user_study(self, study_name, participants):
        """Conduct a user study"""
        study = {
            'name': study_name,
            'participants': participants,
            'date': datetime.now().strftime('%Y-%m-%d')
        }
        self.user_studies.append(study)
        print(
            f"👥 User study conducted: {study_name} "
            f"with {participants} participants"
        )


class AIResearcher(NLPResearcher, HCIResearcher):
    """
    AI Researcher combining NLP and HCI.
    Demonstrates multiple inheritance.
    """
    
    def __init__(self, name, specialization):
        NLPResearcher.__init__(
            self, name, specialization
        )
        self.field = "AI - NLP & HCI"
        self.projects = []
    
    def conduct_research(self):
        """Combined research approach"""
        print(f"\n🔬 {self.name} conducting AI research...")
        print(f"   Combining NLP and HCI approaches")
        print(f"   Specialization: {self.specialization}")
    
    def start_project(self, project_name):
        """Start a new research project"""
        project = {
            'name': project_name,
            'started': datetime.now().strftime('%Y-%m-%d'),
            'status': 'Active'
        }
        self.projects.append(project)
        print(f"🚀 Project started: {project_name}")


# =============================================
# 3. ENCAPSULATION
# =============================================

class GKSProfile:
    """
    GKS applicant profile with encapsulation.
    Private attributes with getters/setters.
    """
    
    def __init__(self, name):
        self.name = name
        self.__topik_level = 0
        self.__gpa = 0.0
        self.__research_papers = 0
        self.__attempts = 0
    
    # TOPIK level property
    @property
    def topik_level(self):
        return self.__topik_level
    
    @topik_level.setter
    def topik_level(self, level):
        if 0 <= level <= 6:
            self.__topik_level = level
            print(f"TOPIK level updated to: {level}")
        else:
            raise ValueError(
                "TOPIK level must be between 0 and 6!"
            )
    
    # GPA property
    @property
    def gpa(self):
        return self.__gpa
    
    @gpa.setter
    def gpa(self, value):
        if 0.0 <= value <= 4.0:
            self.__gpa = value
        else:
            raise ValueError("GPA must be between 0.0 and 4.0!")
    
    @property
    def research_papers(self):
        return self.__research_papers
    
    def add_research_paper(self):
        """Add a research paper"""
        self.__research_papers += 1
        print(
            f"Research paper added! "
            f"Total: {self.__research_papers}"
        )
    
    def record_attempt(self):
        """Record GKS application attempt"""
        self.__attempts += 1
        print(
            f"GKS attempt {self.__attempts} recorded! "
            f"Never give up! 🇰🇷"
        )
    
    def get_profile_summary(self):
        """Get complete profile summary"""
        readiness = (
            (self.__topik_level / 6 * 30) +
            (min(self.__research_papers * 10, 30)) +
            (min(self.__gpa / 4.0 * 20, 20))
        )
        
        return {
            'name': self.name,
            'topik_level': self.__topik_level,
            'gpa': self.__gpa,
            'research_papers': self.__research_papers,
            'gks_attempts': self.__attempts,
            'readiness_score': f"{readiness:.1f}/80"
        }


# =============================================
# MAIN
# =============================================

if __name__ == "__main__":
    print("="*50)
    print("INTERMEDIATE PYTHON - OOP CONCEPTS")
    print("Author: Jajitha")
    print("="*50)
    
    # Scholarship Application
    print("\n1. SCHOLARSHIP APPLICATION:")
    app = ScholarshipApplication(
        "Jajitha",
        "GKS Scholarship",
        "Kunsan National University",
        2027
    )
    app.add_document("Transcript")
    app.add_document("Research Proposal")
    app.add_document("Professor LOR")
    app.add_document("TOPIK Certificate")
    app.submit()
    print(f"\n{app}")
    
    # AI Researcher
    print("\n2. AI RESEARCHER:")
    researcher = AIResearcher(
        "Jajitha",
        "NLP and LLMs"
    )
    researcher.conduct_research()
    researcher.train_model("BERT-Korean", "Korean corpus")
    researcher.publish_paper(
        "Korean NLP with BERT",
        "ACL 2027"
    )
    researcher.conduct_user_study(
        "LLM User Interface Study",
        50
    )
    researcher.start_project(
        "Korean Language Model HCI"
    )
    
    # GKS Profile
    print("\n3. GKS PROFILE:")
    profile = GKSProfile("Jajitha")
    profile.topik_level = 2
    profile.gpa = 3.8
    profile.add_research_paper()
    profile.add_research_paper()
    profile.record_attempt()
    profile.record_attempt()
    profile.record_attempt()
    
    summary = profile.get_profile_summary()
    print("\nProfile Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*50)
    print("Stage 2 Complete! Moving to Stage 3... 🚀")
    print("="*50)
