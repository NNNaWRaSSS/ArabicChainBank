#!/usr/bin/env python3
"""
Enhanced Qur'an QA System for Shared Task 2024
==============================================

This enhanced system provides a complete solution for the Qur'an QA shared task,
including advanced retrieval methods, better ranking algorithms, and comprehensive
evaluation capabilities.
"""

import json
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd

from quran_qa_system import QuranQASystem, Source, Question
from data_loader import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedQuranQASystem(QuranQASystem):
    """
    Enhanced Qur'an QA system with advanced features
    """
    
    def __init__(self, 
                 quran_data_path: str = None, 
                 hadith_data_path: str = None,
                 use_embeddings: bool = True,
                 model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialize the enhanced system
        
        Args:
            quran_data_path: Path to Qur'anic passages data
            hadith_data_path: Path to Sahih Bukhari Hadiths data
            use_embeddings: Whether to use sentence embeddings for retrieval
            model_name: Name of the sentence transformer model to use
        """
        super().__init__(quran_data_path, hadith_data_path)
        
        self.use_embeddings = use_embeddings
        self.model_name = model_name
        
        # Initialize sentence transformer for semantic similarity
        if self.use_embeddings:
            try:
                self.sentence_model = SentenceTransformer(model_name)
                logger.info(f"Loaded sentence transformer model: {model_name}")
            except Exception as e:
                logger.warning(f"Could not load sentence transformer: {e}")
                self.use_embeddings = False
        
        # Enhanced retrieval engine
        self.enhanced_retriever = EnhancedRetrievalEngine(self.use_embeddings, self.sentence_model)
        
        # Load data using the data loader
        self.data_loader = DataLoader()
        self._load_data()
    
    def _load_data(self):
        """Load Qur'an and Hadith data"""
        if not self.quran_sources:
            # Try to load from sample data first
            try:
                self.quran_sources, self.hadith_sources = self.data_loader.create_sample_data()
                logger.info("Loaded sample data")
            except Exception as e:
                logger.error(f"Error loading sample data: {e}")
        
        logger.info(f"Loaded {len(self.quran_sources)} Qur'anic sources")
        logger.info(f"Loaded {len(self.hadith_sources)} Hadith sources")
    
    def retrieve_sources(self, question: Question, max_results: int = 20) -> List[Source]:
        """
        Enhanced retrieval using multiple methods
        """
        # Use enhanced retriever
        quran_candidates = self.enhanced_retriever.retrieve_quran(question, self.quran_sources)
        hadith_candidates = self.enhanced_retriever.retrieve_hadith(question, self.hadith_sources)
        
        # Combine candidates
        all_candidates = quran_candidates + hadith_candidates
        
        if not all_candidates:
            logger.info("No relevant sources found for the question")
            return []
        
        # Enhanced ranking
        ranked_sources = self.enhanced_ranker.rank_sources(question, all_candidates)
        
        return ranked_sources[:max_results]

class EnhancedRetrievalEngine:
    """Enhanced retrieval engine with multiple methods"""
    
    def __init__(self, use_embeddings: bool = True, sentence_model = None):
        self.use_embeddings = use_embeddings
        self.sentence_model = sentence_model
        
        # TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            min_df=2
        )
        
        # BM25-like scoring
        self.bm25_k1 = 1.2
        self.bm25_b = 0.75
        
        # Pre-computed vectors
        self.quran_vectors = None
        self.hadith_vectors = None
        self.quran_embeddings = None
        self.hadith_embeddings = None
    
    def retrieve_quran(self, question: Question, quran_sources: List[Source]) -> List[Source]:
        """Retrieve relevant Qur'anic passages using multiple methods"""
        if not quran_sources:
            return []
        
        candidates = []
        
        # Method 1: TF-IDF retrieval
        tfidf_candidates = self._tfidf_retrieval(question, quran_sources, 'quran')
        candidates.extend(tfidf_candidates)
        
        # Method 2: Semantic similarity (if embeddings available)
        if self.use_embeddings and self.sentence_model:
            semantic_candidates = self._semantic_retrieval(question, quran_sources, 'quran')
            candidates.extend(semantic_candidates)
        
        # Method 3: Keyword-based retrieval
        keyword_candidates = self._keyword_retrieval(question, quran_sources)
        candidates.extend(keyword_candidates)
        
        # Remove duplicates and sort by score
        unique_candidates = self._remove_duplicates(candidates)
        return sorted(unique_candidates, key=lambda x: x.score, reverse=True)
    
    def retrieve_hadith(self, question: Question, hadith_sources: List[Source]) -> List[Source]:
        """Retrieve relevant Hadiths using multiple methods"""
        if not hadith_sources:
            return []
        
        candidates = []
        
        # Method 1: TF-IDF retrieval
        tfidf_candidates = self._tfidf_retrieval(question, hadith_sources, 'hadith')
        candidates.extend(tfidf_candidates)
        
        # Method 2: Semantic similarity
        if self.use_embeddings and self.sentence_model:
            semantic_candidates = self._semantic_retrieval(question, hadith_sources, 'hadith')
            candidates.extend(semantic_candidates)
        
        # Method 3: Keyword-based retrieval
        keyword_candidates = self._keyword_retrieval(question, hadith_sources)
        candidates.extend(keyword_candidates)
        
        # Remove duplicates and sort by score
        unique_candidates = self._remove_duplicates(candidates)
        return sorted(unique_candidates, key=lambda x: x.score, reverse=True)
    
    def _tfidf_retrieval(self, question: Question, sources: List[Source], source_type: str) -> List[Source]:
        """TF-IDF based retrieval"""
        if not sources:
            return []
        
        # Create TF-IDF vectors
        texts = [source.text for source in sources]
        vectors = self.tfidf_vectorizer.fit_transform(texts)
        
        # Create question vector
        question_text = ' '.join(question.keywords)
        question_vector = self.tfidf_vectorizer.transform([question_text])
        
        # Calculate similarities
        similarities = cosine_similarity(question_vector, vectors).flatten()
        
        # Create candidates
        candidates = []
        for i, source in enumerate(sources):
            if similarities[i] > 0.1:  # Threshold
                source.score = similarities[i]
                candidates.append(source)
        
        return candidates
    
    def _semantic_retrieval(self, question: Question, sources: List[Source], source_type: str) -> List[Source]:
        """Semantic similarity based retrieval"""
        if not self.sentence_model or not sources:
            return []
        
        try:
            # Get embeddings for sources
            source_texts = [source.text for source in sources]
            source_embeddings = self.sentence_model.encode(source_texts)
            
            # Get embedding for question
            question_text = ' '.join(question.keywords)
            question_embedding = self.sentence_model.encode([question_text])
            
            # Calculate similarities
            similarities = cosine_similarity(question_embedding, source_embeddings).flatten()
            
            # Create candidates
            candidates = []
            for i, source in enumerate(sources):
                if similarities[i] > 0.3:  # Higher threshold for semantic similarity
                    source.semantic_score = similarities[i]
                    candidates.append(source)
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error in semantic retrieval: {e}")
            return []
    
    def _keyword_retrieval(self, question: Question, sources: List[Source]) -> List[Source]:
        """Keyword-based retrieval"""
        candidates = []
        question_keywords = set(question.keywords)
        
        for source in sources:
            source_words = set(source.text.split())
            
            # Calculate keyword overlap
            overlap = len(question_keywords.intersection(source_words))
            if overlap > 0:
                source.keyword_score = overlap / len(question_keywords) if question_keywords else 0
                candidates.append(source)
        
        return candidates
    
    def _remove_duplicates(self, candidates: List[Source]) -> List[Source]:
        """Remove duplicate sources and combine scores"""
        unique_sources = {}
        
        for candidate in candidates:
            if candidate.id not in unique_sources:
                unique_sources[candidate.id] = candidate
            else:
                # Combine scores from different methods
                existing = unique_sources[candidate.id]
                existing.score = max(getattr(existing, 'score', 0), getattr(candidate, 'score', 0))
                existing.semantic_score = max(getattr(existing, 'semantic_score', 0), 
                                           getattr(candidate, 'semantic_score', 0))
                existing.keyword_score = max(getattr(existing, 'keyword_score', 0), 
                                          getattr(candidate, 'keyword_score', 0))
        
        return list(unique_sources.values())

class EnhancedRankingEngine:
    """Enhanced ranking engine with multiple criteria"""
    
    def __init__(self):
        self.ranking_weights = {
            'tfidf_score': 0.25,
            'semantic_score': 0.35,
            'keyword_score': 0.20,
            'entity_match': 0.10,
            'source_type_preference': 0.10
        }
    
    def rank_sources(self, question: Question, candidates: List[Source]) -> List[Source]:
        """Enhanced ranking with multiple criteria"""
        for candidate in candidates:
            score = self._calculate_enhanced_score(question, candidate)
            candidate.final_score = score
        
        return sorted(candidates, key=lambda x: x.final_score, reverse=True)
    
    def _calculate_enhanced_score(self, question: Question, source: Source) -> float:
        """Calculate enhanced composite score"""
        scores = {}
        
        # TF-IDF score
        scores['tfidf_score'] = getattr(source, 'score', 0.0)
        
        # Semantic similarity score
        scores['semantic_score'] = getattr(source, 'semantic_score', 0.0)
        
        # Keyword overlap score
        scores['keyword_score'] = getattr(source, 'keyword_score', 0.0)
        
        # Entity match score
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
        # Enhanced preference logic
        theological_keywords = ['الله', 'القرآن', 'الإيمان', 'الإسلام', 'العبادة', 'الصلاة', 'الزكاة']
        practical_keywords = ['كيف', 'متى', 'أين', 'أي', 'كم']
        
        question_text = question.text.lower()
        
        # Prefer Qur'an for theological questions
        if any(keyword in question_text for keyword in theological_keywords):
            return 1.0 if source.source_type == 'quran' else 0.3
        
        # Prefer Hadith for practical questions
        elif any(keyword in question_text for keyword in practical_keywords):
            return 0.8 if source.source_type == 'hadith' else 0.4
        
        # Neutral preference for other questions
        return 0.5

class EvaluationEngine:
    """Evaluation engine for the Qur'an QA system"""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate_system(self, system: EnhancedQuranQASystem, test_questions: List[Dict]) -> Dict:
        """
        Evaluate the system performance
        
        Args:
            system: The QA system to evaluate
            test_questions: List of test questions with ground truth
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = {
            'precision': [],
            'recall': [],
            'f1_score': [],
            'mrr': [],
            'has_answer_accuracy': []
        }
        
        for test_case in test_questions:
            question = test_case['question']
            ground_truth = test_case.get('ground_truth', [])
            has_answer = test_case.get('has_answer', True)
            
            # Get system response
            response = system.answer_question(question)
            
            # Calculate metrics
            precision, recall, f1 = self._calculate_retrieval_metrics(response, ground_truth)
            mrr = self._calculate_mrr(response, ground_truth)
            has_answer_acc = self._calculate_has_answer_accuracy(response, has_answer)
            
            results['precision'].append(precision)
            results['recall'].append(recall)
            results['f1_score'].append(f1)
            results['mrr'].append(mrr)
            results['has_answer_accuracy'].append(has_answer_acc)
        
        # Calculate averages
        avg_results = {}
        for metric in results:
            avg_results[f'avg_{metric}'] = np.mean(results[metric])
        
        return avg_results
    
    def _calculate_retrieval_metrics(self, response: Dict, ground_truth: List[str]) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score"""
        predicted_ids = [source['id'] for source in response['sources']]
        ground_truth_ids = ground_truth
        
        if not ground_truth_ids:
            return 0.0, 0.0, 0.0
        
        # Calculate precision and recall
        relevant_retrieved = len(set(predicted_ids) & set(ground_truth_ids))
        
        precision = relevant_retrieved / len(predicted_ids) if predicted_ids else 0.0
        recall = relevant_retrieved / len(ground_truth_ids) if ground_truth_ids else 0.0
        
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    def _calculate_mrr(self, response: Dict, ground_truth: List[str]) -> float:
        """Calculate Mean Reciprocal Rank"""
        if not ground_truth:
            return 0.0
        
        for i, source in enumerate(response['sources']):
            if source['id'] in ground_truth:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def _calculate_has_answer_accuracy(self, response: Dict, ground_truth_has_answer: bool) -> float:
        """Calculate accuracy for has_answer prediction"""
        return 1.0 if response['has_answer'] == ground_truth_has_answer else 0.0

def main():
    """Main function to demonstrate the enhanced system"""
    
    # Initialize the enhanced system
    system = EnhancedQuranQASystem(use_embeddings=True)
    
    # Test questions
    test_questions = [
        {
            "question": "ما هي أركان الإسلام الخمسة؟",
            "expected_type": "factoid",
            "expected_keywords": ["أركان", "الإسلام", "خمسة"]
        },
        {
            "question": "من هو النبي محمد؟",
            "expected_type": "factoid",
            "expected_keywords": ["النبي", "محمد"]
        },
        {
            "question": "كيف نصلي؟",
            "expected_type": "non-factoid",
            "expected_keywords": ["كيف", "نصلي", "الصلاة"]
        },
        {
            "question": "ما هي شروط الحج؟",
            "expected_type": "non-factoid",
            "expected_keywords": ["شروط", "الحج"]
        },
        {
            "question": "هل يوجد إجابة لهذا السؤال في القرآن؟",
            "expected_type": "non-factoid",
            "expected_keywords": ["إجابة", "السؤال", "القرآن"]
        }
    ]
    
    print("=== Enhanced Qur'an QA System Demo ===\n")
    
    for i, test_case in enumerate(test_questions, 1):
        question = test_case["question"]
        print(f"Test Case {i}: {question}")
        print("-" * 60)
        
        # Process question
        processed_question = system.process_question(question)
        print(f"Question Type: {processed_question.question_type}")
        print(f"Keywords: {', '.join(processed_question.keywords)}")
        print(f"Entities: {', '.join(processed_question.entities)}")
        
        # Get answer
        result = system.answer_question(question)
        print(f"Has Answer: {result['has_answer']}")
        
        if result['sources']:
            print(f"\nTop Sources:")
            for j, source in enumerate(result['sources'][:3]):  # Show top 3
                print(f"{j+1}. [{source['source_type'].upper()}] {source['text'][:80]}...")
                print(f"   Score: {source.get('relevance_score', 0.0):.3f}")
        else:
            print("No relevant sources found.")
        
        print("\n" + "="*80 + "\n")
    
    # Demonstrate evaluation
    print("=== System Evaluation ===")
    evaluator = EvaluationEngine()
    
    # Sample evaluation data
    eval_data = [
        {
            "question": "ما هي أركان الإسلام الخمسة؟",
            "ground_truth": ["hadith_2"],  # Hadith about pillars of Islam
            "has_answer": True
        },
        {
            "question": "كيف نصلي؟",
            "ground_truth": ["hadith_1"],  # Hadith about prayer
            "has_answer": True
        }
    ]
    
    # Run evaluation
    eval_results = evaluator.evaluate_system(system, eval_data)
    print("Evaluation Results:")
    for metric, value in eval_results.items():
        print(f"{metric}: {value:.3f}")

if __name__ == "__main__":
    main()