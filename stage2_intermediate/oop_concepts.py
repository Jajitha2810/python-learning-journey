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
    NLP Researcher class inheriting from
