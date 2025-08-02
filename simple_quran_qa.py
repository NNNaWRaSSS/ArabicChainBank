#!/usr/bin/env python3
"""
Simple Qur'an QA System for Shared Task 2024
============================================

A simplified version that demonstrates the core concepts without external dependencies.
This version focuses on the key components needed to solve the shared task.
"""

import re
import json
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from collections import Counter

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

class SimpleQuranQASystem:
    """
    Simple Qur'an QA system using basic text processing
    """
    
    def __init__(self):
        self.quran_sources = []
        self.hadith_sources = []
        self.question_analyzer = SimpleQuestionAnalyzer()
        self.retriever = SimpleRetrievalEngine()
        self.ranker = SimpleRankingEngine()
    
    def load_sample_data(self):
        """Load sample data for demonstration"""
        # Sample Qur'anic passages
        self.quran_sources = [
            Source(
                id="quran_1_1",
                text="بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
                source_type='quran',
                surah="1",
                ayah="1"
            ),
            Source(
                id="quran_2_255",
                text="اللَّهُ لَا إِلَٰهَ إِلَّا هُوَ الْحَيُّ الْقَيُّومُ ۚ لَا تَأْخُذُهُ سِنَةٌ وَلَا نَوْمٌ",
                source_type='quran',
                surah="2",
                ayah="255"
            ),
            Source(
                id="quran_112_1",
                text="قُلْ هُوَ اللَّهُ أَحَدٌ",
                source_type='quran',
                surah="112",
                ayah="1"
            ),
            Source(
                id="quran_2_286",
                text="لَا يُكَلِّفُ اللَّهُ نَفْسًا إِلَّا وُسْعَهَا",
                source_type='quran',
                surah="2",
                ayah="286"
            ),
            Source(
                id="quran_49_13",
                text="إِنَّ أَكْرَمَكُمْ عِنْدَ اللَّهِ أَتْقَاكُمْ",
                source_type='quran',
                surah="49",
                ayah="13"
            )
        ]
        
        # Sample Hadiths
        self.hadith_sources = [
            Source(
                id="hadith_1",
                text="إنما الأعمال بالنيات وإنما لكل امرئ ما نوى",
                source_type='hadith',
                hadith_number="1",
                book="كتاب بدء الوحي",
                chapter="باب كيف كان بدء الوحي"
            ),
            Source(
                id="hadith_2",
                text="بني الإسلام على خمس شهادة أن لا إله إلا الله وأن محمدا رسول الله وإقام الصلاة وإيتاء الزكاة وحج البيت وصوم رمضان",
                source_type='hadith',
                hadith_number="8",
                book="كتاب الإيمان",
                chapter="باب قول النبي صلى الله عليه وسلم بني الإسلام على خمس"
            ),
            Source(
                id="hadith_3",
                text="من حسن إسلام المرء تركه ما لا يعنيه",
                source_type='hadith',
                hadith_number="11",
                book="كتاب الإيمان",
                chapter="باب من حسن إسلام المرء تركه ما لا يعنيه"
            ),
            Source(
                id="hadith_4",
                text="لا يؤمن أحدكم حتى يحب لأخيه ما يحب لنفسه",
                source_type='hadith',
                hadith_number="13",
                book="كتاب الإيمان",
                chapter="باب من الإيمان أن يحب لأخيه ما يحب لنفسه"
            ),
            Source(
                id="hadith_5",
                text="أمرت أن أقاتل الناس حتى يشهدوا أن لا إله إلا الله وأن محمدا رسول الله",
                source_type='hadith',
                hadith_number="25",
                book="كتاب الإيمان",
                chapter="باب قول النبي صلى الله عليه وسلم أمرت أن أقاتل الناس"
            )
        ]
        
        logger.info(f"Loaded {len(self.quran_sources)} Qur'anic sources")
        logger.info(f"Loaded {len(self.hadith_sources)} Hadith sources")
    
    def process_question(self, question_text: str) -> Question:
        """Process and analyze the input question"""
        return self.question_analyzer.analyze(question_text)
    
    def retrieve_sources(self, question: Question, max_results: int = 20) -> List[Source]:
        """Retrieve relevant sources for the given question"""
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
        """Main method to answer a question"""
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

class SimpleQuestionAnalyzer:
    """Simple question analyzer using basic text processing"""
    
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
        """Analyze the question to extract features"""
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

class SimpleRetrievalEngine:
    """Simple retrieval engine using keyword matching"""
    
    def retrieve_quran(self, question: Question, quran_sources: List[Source]) -> List[Source]:
        """Retrieve relevant Qur'anic passages using keyword matching"""
        if not quran_sources:
            return []
        
        candidates = []
        question_keywords = set(question.keywords)
        
        for source in quran_sources:
            source_words = set(source.text.split())
            
            # Calculate keyword overlap
            overlap = len(question_keywords.intersection(source_words))
            if overlap > 0:
                source.score = overlap / len(question_keywords) if question_keywords else 0
                candidates.append(source)
        
        return sorted(candidates, key=lambda x: x.score, reverse=True)
    
    def retrieve_hadith(self, question: Question, hadith_sources: List[Source]) -> List[Source]:
        """Retrieve relevant Hadiths using keyword matching"""
        if not hadith_sources:
            return []
        
        candidates = []
        question_keywords = set(question.keywords)
        
        for source in hadith_sources:
            source_words = set(source.text.split())
            
            # Calculate keyword overlap
            overlap = len(question_keywords.intersection(source_words))
            if overlap > 0:
                source.score = overlap / len(question_keywords) if question_keywords else 0
                candidates.append(source)
        
        return sorted(candidates, key=lambda x: x.score, reverse=True)

class SimpleRankingEngine:
    """Simple ranking engine with basic scoring"""
    
    def __init__(self):
        self.ranking_weights = {
            'keyword_score': 0.7,
            'entity_match': 0.2,
            'source_type_preference': 0.1
        }
    
    def rank_sources(self, question: Question, candidates: List[Source]) -> List[Source]:
        """Rank sources by multiple criteria"""
        for candidate in candidates:
            score = self._calculate_composite_score(question, candidate)
            candidate.final_score = score
        
        return sorted(candidates, key=lambda x: x.final_score, reverse=True)
    
    def _calculate_composite_score(self, question: Question, source: Source) -> float:
        """Calculate composite relevance score"""
        scores = {}
        
        # Keyword score (already calculated in retrieval)
        scores['keyword_score'] = getattr(source, 'score', 0.0)
        
        # Entity match
        scores['entity_match'] = self._calculate_entity_match(question, source)
        
        # Source type preference
        scores['source_type_preference'] = self._calculate_source_preference(question, source)
        
        # Calculate weighted sum
        final_score = sum(
            scores[key] * self.ranking_weights[key] 
            for key in scores
        )
        
        return final_score
    
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
    system = SimpleQuranQASystem()
    system.load_sample_data()
    
    # Example questions
    test_questions = [
        "ما هي أركان الإسلام الخمسة؟",
        "من هو النبي محمد؟",
        "كيف نصلي؟",
        "ما هي شروط الحج؟",
        "هل يوجد إجابة لهذا السؤال في القرآن؟"
    ]
    
    print("=== Simple Qur'an QA System Demo ===\n")
    
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
            for i, source in enumerate(result['sources'][:3]):  # Show top 3
                print(f"{i+1}. [{source['source_type'].upper()}] {source['text'][:80]}...")
                print(f"   Score: {source.get('relevance_score', 0.0):.3f}")
        else:
            print("No relevant sources found.")
        
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()