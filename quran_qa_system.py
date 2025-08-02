#!/usr/bin/env python3
"""
Qur'an QA System for Shared Task 2024
=====================================

This system addresses the Qur'an QA shared task by implementing:
1. Question preprocessing and analysis
2. Retrieval of relevant Qur'anic passages and Hadiths
3. Ranking of answer-bearing sources
4. Support for both factoid and non-factoid questions

The system can handle cases where questions may not have answers in the provided sources.
"""

import re
import json
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import arabic_reshaper
from bidi.algorithm import get_display
import nltk
from nltk.corpus import stopwords
import requests
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Source:
    """Represents a Qur'anic passage or Hadith source"""
    id: str
    text: str
    source_type: str  # 'quran' or 'hadith'
    surah: Optional[str] = None
    ayah: Optional[str] = None
    hadith_number: Optional[str] = None
    book: Optional[str] = None
    chapter: Optional[str] = None

@dataclass
class Question:
    """Represents a question in MSA"""
    text: str
    question_type: str  # 'factoid' or 'non-factoid'
    keywords: List[str]
    entities: List[str]

class QuranQASystem:
    """
    Main system for Qur'an QA task
    """
    
    def __init__(self, quran_data_path: str = None, hadith_data_path: str = None):
        """
        Initialize the Qur'an QA system
        
        Args:
            quran_data_path: Path to Qur'anic passages data
            hadith_data_path: Path to Sahih Bukhari Hadiths data
        """
        self.quran_sources = []
        self.hadith_sources = []
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words=None  # We'll handle Arabic stop words manually
        )
        self.question_analyzer = QuestionAnalyzer()
        self.retriever = RetrievalEngine()
        self.ranker = RankingEngine()
        
        # Load data if provided
        if quran_data_path:
            self.load_quran_data(quran_data_path)
        if hadith_data_path:
            self.load_hadith_data(hadith_data_path)
    
    def load_quran_data(self, data_path: str):
        """Load Qur'anic passages data"""
        try:
            # This would load from your actual data format
            # For now, creating sample data structure
            logger.info("Loading Qur'anic passages...")
            # Implementation would depend on your data format
            pass
        except Exception as e:
            logger.error(f"Error loading Qur'an data: {e}")
    
    def load_hadith_data(self, data_path: str):
        """Load Sahih Bukhari Hadiths data"""
        try:
            logger.info("Loading Sahih Bukhari Hadiths...")
            # Implementation would depend on your data format
            pass
        except Exception as e:
            logger.error(f"Error loading Hadith data: {e}")
    
    def process_question(self, question_text: str) -> Question:
        """
        Process and analyze the input question
        
        Args:
            question_text: Raw question text in MSA
            
        Returns:
            Processed Question object
        """
        return self.question_analyzer.analyze(question_text)
    
    def retrieve_sources(self, question: Question, max_results: int = 20) -> List[Source]:
        """
        Retrieve relevant sources for the given question
        
        Args:
            question: Processed question object
            max_results: Maximum number of results to return
            
        Returns:
            List of relevant sources ranked by relevance
        """
        # Get candidates from both Qur'an and Hadith collections
        quran_candidates = self.retriever.retrieve_quran(question, self.quran_sources)
        hadith_candidates = self.retriever.retrieve_hadith(question, self.hadith_sources)
        
        # Combine and rank all candidates
        all_candidates = quran_candidates + hadith_candidates
        
        if not all_candidates:
            logger.info("No relevant sources found for the question")
            return []
        
        # Rank the candidates
        ranked_sources = self.ranker.rank_sources(question, all_candidates)
        
        return ranked_sources[:max_results]
    
    def answer_question(self, question_text: str) -> Dict:
        """
        Main method to answer a question
        
        Args:
            question_text: Question in MSA
            
        Returns:
            Dictionary containing the answer with ranked sources
        """
        try:
            # Process the question
            question = self.process_question(question_text)
            logger.info(f"Processed question: {question.text}")
            logger.info(f"Question type: {question.question_type}")
            logger.info(f"Keywords: {question.keywords}")
            
            # Retrieve relevant sources
            sources = self.retrieve_sources(question)
            
            # Prepare response
            response = {
                "question": question_text,
                "question_type": question.question_type,
                "keywords": question.keywords,
                "entities": question.entities,
                "sources": [],
                "has_answer": len(sources) > 0
            }
            
            # Format sources for output
            for i, source in enumerate(sources):
                source_info = {
                    "rank": i + 1,
                    "id": source.id,
                    "text": source.text,
                    "source_type": source.source_type,
                    "relevance_score": getattr(source, 'score', 0.0)
                }
                
                if source.source_type == 'quran':
                    source_info.update({
                        "surah": source.surah,
                        "ayah": source.ayah
                    })
                elif source.source_type == 'hadith':
                    source_info.update({
                        "hadith_number": source.hadith_number,
                        "book": source.book,
                        "chapter": source.chapter
                    })
                
                response["sources"].append(source_info)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                "question": question_text,
                "error": str(e),
                "sources": [],
                "has_answer": False
            }

class QuestionAnalyzer:
    """Analyzes and processes questions in MSA"""
    
    def __init__(self):
        self.arabic_stop_words = self._load_arabic_stop_words()
        self.question_patterns = {
            'factoid': [
                r'مَنْ', r'مَا', r'أَيْنَ', r'مَتَى', r'كَيْفَ', r'لِمَاذَا',
                r'مَنْ هُوَ', r'مَا هُوَ', r'أَيْنَ هُوَ', r'مَتَى يَكُونُ'
            ],
            'non-factoid': [
                r'كَيْفَ', r'لِمَاذَا', r'لِمَاذَا يَكُونُ', r'كَيْفَ يَكُونُ',
                r'مَا هِيَ', r'مَا هُوَ', r'كَيْفَ يَعْمَلُ'
            ]
        }
    
    def _load_arabic_stop_words(self) -> set:
        """Load Arabic stop words"""
        return {
            'في', 'من', 'إلى', 'على', 'عن', 'مع', 'هذا', 'هذه', 'ذلك', 'تلك',
            'الذي', 'التي', 'الذين', 'اللاتي', 'هو', 'هي', 'هم', 'هن', 'أنا',
            'نحن', 'أنت', 'أنتم', 'أنتن', 'كان', 'كانت', 'يكون', 'تكون',
            'و', 'أو', 'لكن', 'إذا', 'إن', 'أن', 'لا', 'ما', 'هل', 'أي'
        }
    
    def analyze(self, question_text: str) -> Question:
        """
        Analyze the question to extract features
        
        Args:
            question_text: Raw question text
            
        Returns:
            Question object with extracted features
        """
        # Clean and normalize the text
        cleaned_text = self._clean_text(question_text)
        
        # Determine question type
        question_type = self._determine_question_type(cleaned_text)
        
        # Extract keywords
        keywords = self._extract_keywords(cleaned_text)
        
        # Extract named entities
        entities = self._extract_entities(cleaned_text)
        
        return Question(
            text=cleaned_text,
            question_type=question_type,
            keywords=keywords,
            entities=entities
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize Arabic text"""
        # Remove diacritics
        text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _determine_question_type(self, text: str) -> str:
        """Determine if question is factoid or non-factoid"""
        text_lower = text.lower()
        
        # Check for factoid patterns
        for pattern in self.question_patterns['factoid']:
            if re.search(pattern, text_lower):
                return 'factoid'
        
        # Check for non-factoid patterns
        for pattern in self.question_patterns['non-factoid']:
            if re.search(pattern, text_lower):
                return 'non-factoid'
        
        # Default to non-factoid for complex questions
        return 'non-factoid'
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from the question"""
        words = text.split()
        keywords = []
        
        for word in words:
            # Remove stop words and short words
            if (word not in self.arabic_stop_words and 
                len(word) > 2 and 
                not word.isdigit()):
                keywords.append(word)
        
        return keywords
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from the question"""
        # Simple entity extraction - in practice, you'd use a proper NER system
        entities = []
        
        # Look for common Islamic terms and names
        islamic_terms = [
            'الله', 'محمد', 'عيسى', 'موسى', 'إبراهيم', 'نوح', 'داود',
            'القرآن', 'الحديث', 'الصلاة', 'الزكاة', 'الصوم', 'الحج',
            'الجنة', 'النار', 'الملائكة', 'الأنبياء', 'الرسل'
        ]
        
        for term in islamic_terms:
            if term in text:
                entities.append(term)
        
        return entities

class RetrievalEngine:
    """Handles retrieval of relevant sources"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            min_df=2
        )
        self.quran_vectors = None
        self.hadith_vectors = None
    
    def retrieve_quran(self, question: Question, quran_sources: List[Source]) -> List[Source]:
        """Retrieve relevant Qur'anic passages"""
        if not quran_sources:
            return []
        
        # Create TF-IDF vectors for Qur'an sources
        quran_texts = [source.text for source in quran_sources]
        quran_vectors = self.vectorizer.fit_transform(quran_texts)
        
        # Create vector for question
        question_vector = self.vectorizer.transform([' '.join(question.keywords)])
        
        # Calculate similarities
        similarities = cosine_similarity(question_vector, quran_vectors).flatten()
        
        # Create candidates with scores
        candidates = []
        for i, source in enumerate(quran_sources):
            source.score = similarities[i]
            if similarities[i] > 0.1:  # Threshold for relevance
                candidates.append(source)
        
        return sorted(candidates, key=lambda x: x.score, reverse=True)
    
    def retrieve_hadith(self, question: Question, hadith_sources: List[Source]) -> List[Source]:
        """Retrieve relevant Hadiths"""
        if not hadith_sources:
            return []
        
        # Similar process as Qur'an retrieval
        hadith_texts = [source.text for source in hadith_sources]
        hadith_vectors = self.vectorizer.fit_transform(hadith_texts)
        
        question_vector = self.vectorizer.transform([' '.join(question.keywords)])
        similarities = cosine_similarity(question_vector, hadith_vectors).flatten()
        
        candidates = []
        for i, source in enumerate(hadith_sources):
            source.score = similarities[i]
            if similarities[i] > 0.1:
                candidates.append(source)
        
        return sorted(candidates, key=lambda x: x.score, reverse=True)

class RankingEngine:
    """Ranks retrieved sources by relevance"""
    
    def __init__(self):
        self.ranking_weights = {
            'tfidf_score': 0.4,
            'keyword_overlap': 0.3,
            'entity_match': 0.2,
            'source_type_preference': 0.1
        }
    
    def rank_sources(self, question: Question, candidates: List[Source]) -> List[Source]:
        """
        Rank sources by multiple criteria
        
        Args:
            question: Processed question
            candidates: List of candidate sources
            
        Returns:
            Ranked list of sources
        """
        for candidate in candidates:
            # Calculate composite score
            score = self._calculate_composite_score(question, candidate)
            candidate.final_score = score
        
        # Sort by final score
        ranked_sources = sorted(candidates, key=lambda x: x.final_score, reverse=True)
        
        return ranked_sources
    
    def _calculate_composite_score(self, question: Question, source: Source) -> float:
        """Calculate composite relevance score"""
        scores = {}
        
        # TF-IDF score (already calculated in retrieval)
        scores['tfidf_score'] = getattr(source, 'score', 0.0)
        
        # Keyword overlap
        scores['keyword_overlap'] = self._calculate_keyword_overlap(question, source)
        
        # Entity match
        scores['entity_match'] = self._calculate_entity_match(question, source)
        
        # Source type preference (Qur'an vs Hadith)
        scores['source_type_preference'] = self._calculate_source_preference(question, source)
        
        # Calculate weighted sum
        final_score = sum(
            scores[key] * self.ranking_weights[key] 
            for key in scores
        )
        
        return final_score
    
    def _calculate_keyword_overlap(self, question: Question, source: Source) -> float:
        """Calculate keyword overlap between question and source"""
        question_keywords = set(question.keywords)
        source_words = set(source.text.split())
        
        if not question_keywords:
            return 0.0
        
        overlap = len(question_keywords.intersection(source_words))
        return overlap / len(question_keywords)
    
    def _calculate_entity_match(self, question: Question, source: Source) -> float:
        """Calculate entity match score"""
        question_entities = set(question.entities)
        source_words = set(source.text.split())
        
        if not question_entities:
            return 0.0
        
        matches = len(question_entities.intersection(source_words))
        return matches / len(question_entities) if question_entities else 0.0
    
    def _calculate_source_preference(self, question: Question, source: Source) -> float:
        """Calculate source type preference score"""
        # Simple heuristic: prefer Qur'an for theological questions
        theological_keywords = ['الله', 'القرآن', 'الإيمان', 'الإسلام', 'العبادة']
        
        if question.question_type == 'factoid':
            # For factoid questions, prefer the source type that has more matches
            return 0.5  # Neutral preference
        else:
            # For non-factoid questions, prefer Qur'an for theological content
            if any(keyword in question.text for keyword in theological_keywords):
                return 1.0 if source.source_type == 'quran' else 0.3
            else:
                return 0.5  # Neutral preference

def main():
    """Main function to demonstrate the system"""
    
    # Initialize the system
    system = QuranQASystem()
    
    # Example questions
    test_questions = [
        "ما هي أركان الإسلام الخمسة؟",
        "من هو النبي محمد؟",
        "كيف نصلي؟",
        "ما هي شروط الحج؟",
        "هل يوجد إجابة لهذا السؤال في القرآن؟"  # Question that might not have an answer
    ]
    
    print("=== Qur'an QA System Demo ===\n")
    
    for question in test_questions:
        print(f"Question: {question}")
        print("-" * 50)
        
        result = system.answer_question(question)
        
        print(f"Question Type: {result['question_type']}")
        print(f"Keywords: {', '.join(result['keywords'])}")
        print(f"Entities: {', '.join(result['entities'])}")
        print(f"Has Answer: {result['has_answer']}")
        
        if result['sources']:
            print(f"\nTop Sources:")
            for i, source in enumerate(result['sources'][:5]):  # Show top 5
                print(f"{i+1}. [{source['source_type'].upper()}] {source['text'][:100]}...")
                if source['source_type'] == 'quran':
                    print(f"   Surah: {source.get('surah', 'N/A')}, Ayah: {source.get('ayah', 'N/A')}")
                else:
                    print(f"   Book: {source.get('book', 'N/A')}, Chapter: {source.get('chapter', 'N/A')}")
        else:
            print("No relevant sources found.")
        
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()