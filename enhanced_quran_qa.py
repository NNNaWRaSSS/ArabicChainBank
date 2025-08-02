#!/usr/bin/env python3
"""
Enhanced Qur'an QA System with Matryoshka Semantic Layers
========================================================

This enhanced system implements a multi-layered semantic representation approach
using Matryoshka layers to capture different levels of meaning:
- Literal layer: Direct word meanings
- Cultural layer: Cultural and historical context
- Theological layer: Religious and spiritual meanings
- Interpretive layer: Scholarly interpretations
- Contextual layer: Situational and contextual meanings
"""

import json
import logging
from typing import List, Dict, Optional, Tuple, NamedTuple
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from dataclasses import dataclass
from enum import Enum

from quran_qa_system import QuranQASystem, Source, Question
from data_loader import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticLayer(Enum):
    """Enumeration of semantic layers in the Matryoshka approach"""
    LITERAL = "literal"
    CULTURAL = "cultural"
    THEOLOGICAL = "theological"
    INTERPRETIVE = "interpretive"
    CONTEXTUAL = "contextual"

@dataclass
class MatryoshkaEmbedding:
    """Represents multi-layered embeddings for semantic analysis"""
    literal_embedding: np.ndarray
    cultural_embedding: np.ndarray
    theological_embedding: np.ndarray
    interpretive_embedding: np.ndarray
    contextual_embedding: np.ndarray
    
    def get_layer_embedding(self, layer: SemanticLayer) -> np.ndarray:
        """Get embedding for a specific semantic layer"""
        layer_map = {
            SemanticLayer.LITERAL: self.literal_embedding,
            SemanticLayer.CULTURAL: self.cultural_embedding,
            SemanticLayer.THEOLOGICAL: self.theological_embedding,
            SemanticLayer.INTERPRETIVE: self.interpretive_embedding,
            SemanticLayer.CONTEXTUAL: self.contextual_embedding
        }
        return layer_map[layer]

class MatryoshkaSemanticAnalyzer:
    """
    Analyzes text using multiple semantic layers in a Matryoshka approach
    """
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.sentence_model = SentenceTransformer(model_name)
        
        # Layer-specific prompts for semantic transformation
        self.layer_prompts = {
            SemanticLayer.LITERAL: "Literal meaning: ",
            SemanticLayer.CULTURAL: "Cultural and historical context: ",
            SemanticLayer.THEOLOGICAL: "Religious and spiritual meaning: ",
            SemanticLayer.INTERPRETIVE: "Scholarly interpretation: ",
            SemanticLayer.CONTEXTUAL: "Situational context: "
        }
        
        # Islamic-specific semantic mappings
        self.islamic_semantic_mappings = {
            'الله': {
                SemanticLayer.LITERAL: 'God',
                SemanticLayer.CULTURAL: 'The divine being in Islamic tradition',
                SemanticLayer.THEOLOGICAL: 'The one and only God, creator of all',
                SemanticLayer.INTERPRETIVE: 'The absolute, transcendent deity',
                SemanticLayer.CONTEXTUAL: 'The object of worship and devotion'
            },
            'الصلاة': {
                SemanticLayer.LITERAL: 'prayer',
                SemanticLayer.CULTURAL: 'Islamic ritual prayer',
                SemanticLayer.THEOLOGICAL: 'Direct communication with Allah',
                SemanticLayer.INTERPRETIVE: 'Spiritual discipline and worship',
                SemanticLayer.CONTEXTUAL: 'Prescribed daily worship'
            },
            'الزكاة': {
                SemanticLayer.LITERAL: 'charity',
                SemanticLayer.CULTURAL: 'Islamic almsgiving',
                SemanticLayer.THEOLOGICAL: 'Purification of wealth',
                SemanticLayer.INTERPRETIVE: 'Social responsibility and justice',
                SemanticLayer.CONTEXTUAL: 'Obligatory giving to the poor'
            },
            'الحج': {
                SemanticLayer.LITERAL: 'pilgrimage',
                SemanticLayer.CULTURAL: 'Annual Islamic pilgrimage to Mecca',
                SemanticLayer.THEOLOGICAL: 'Spiritual journey and submission',
                SemanticLayer.INTERPRETIVE: 'Unity of the Muslim community',
                SemanticLayer.CONTEXTUAL: 'Once-in-lifetime obligation'
            }
        }
    
    def create_matryoshka_embedding(self, text: str) -> MatryoshkaEmbedding:
        """
        Create multi-layered embeddings for the given text
        
        Args:
            text: Input text to analyze
            
        Returns:
            MatryoshkaEmbedding with embeddings for each semantic layer
        """
        # Generate layer-specific versions of the text
        layer_texts = self._generate_layer_texts(text)
        
        # Create embeddings for each layer
        embeddings = {}
        for layer, layer_text in layer_texts.items():
            embeddings[layer] = self.sentence_model.encode([layer_text])[0]
        
        return MatryoshkaEmbedding(
            literal_embedding=embeddings[SemanticLayer.LITERAL],
            cultural_embedding=embeddings[SemanticLayer.CULTURAL],
            theological_embedding=embeddings[SemanticLayer.THEOLOGICAL],
            interpretive_embedding=embeddings[SemanticLayer.INTERPRETIVE],
            contextual_embedding=embeddings[SemanticLayer.CONTEXTUAL]
        )
    
    def _generate_layer_texts(self, text: str) -> Dict[SemanticLayer, str]:
        """
        Generate layer-specific versions of the input text
        
        Args:
            text: Original text
            
        Returns:
            Dictionary mapping each semantic layer to its specific text version
        """
        layer_texts = {}
        
        # Literal layer - direct translation/meaning
        layer_texts[SemanticLayer.LITERAL] = f"{self.layer_prompts[SemanticLayer.LITERAL]}{text}"
        
        # Cultural layer - add cultural context
        cultural_context = self._add_cultural_context(text)
        layer_texts[SemanticLayer.CULTURAL] = f"{self.layer_prompts[SemanticLayer.CULTURAL]}{cultural_context}"
        
        # Theological layer - add religious meaning
        theological_context = self._add_theological_context(text)
        layer_texts[SemanticLayer.THEOLOGICAL] = f"{self.layer_prompts[SemanticLayer.THEOLOGICAL]}{theological_context}"
        
        # Interpretive layer - add scholarly interpretation
        interpretive_context = self._add_interpretive_context(text)
        layer_texts[SemanticLayer.INTERPRETIVE] = f"{self.layer_prompts[SemanticLayer.INTERPRETIVE]}{interpretive_context}"
        
        # Contextual layer - add situational context
        contextual_context = self._add_contextual_context(text)
        layer_texts[SemanticLayer.CONTEXTUAL] = f"{self.layer_prompts[SemanticLayer.CONTEXTUAL]}{contextual_context}"
        
        return layer_texts
    
    def _add_cultural_context(self, text: str) -> str:
        """Add cultural and historical context to the text"""
        # Add cultural context based on Islamic terms
        cultural_additions = []
        
        islamic_terms = ['الله', 'محمد', 'القرآن', 'الصلاة', 'الزكاة', 'الحج', 'رمضان']
        for term in islamic_terms:
            if term in text:
                cultural_additions.append(f"in Islamic tradition and culture")
                break
        
        if cultural_additions:
            return f"{text} {' '.join(cultural_additions)}"
        return text
    
    def _add_theological_context(self, text: str) -> str:
        """Add religious and spiritual context to the text"""
        theological_additions = []
        
        # Check for theological terms
        theological_terms = ['الله', 'الإيمان', 'الإسلام', 'العبادة', 'الجنة', 'النار']
        for term in theological_terms:
            if term in text:
                theological_additions.append("in religious and spiritual context")
                break
        
        if theological_additions:
            return f"{text} {' '.join(theological_additions)}"
        return text
    
    def _add_interpretive_context(self, text: str) -> str:
        """Add scholarly interpretation context to the text"""
        interpretive_additions = []
        
        # Check for terms that have scholarly interpretations
        interpretive_terms = ['القرآن', 'الحديث', 'الشريعة', 'الفقه']
        for term in interpretive_terms:
            if term in text:
                interpretive_additions.append("as interpreted by Islamic scholars")
                break
        
        if interpretive_additions:
            return f"{text} {' '.join(interpretive_additions)}"
        return text
    
    def _add_contextual_context(self, text: str) -> str:
        """Add situational and contextual meaning to the text"""
        contextual_additions = []
        
        # Check for contextual terms
        contextual_terms = ['كيف', 'متى', 'أين', 'لماذا']
        for term in contextual_terms:
            if term in text:
                contextual_additions.append("in specific situations and contexts")
                break
        
        if contextual_additions:
            return f"{text} {' '.join(contextual_additions)}"
        return text
    
    def calculate_layer_similarity(self, 
                                 embedding1: MatryoshkaEmbedding, 
                                 embedding2: MatryoshkaEmbedding,
                                 layer_weights: Optional[Dict[SemanticLayer, float]] = None) -> float:
        """
        Calculate weighted similarity across all semantic layers
        
        Args:
            embedding1: First Matryoshka embedding
            embedding2: Second Matryoshka embedding
            layer_weights: Optional weights for each layer
            
        Returns:
            Weighted similarity score
        """
        if layer_weights is None:
            # Default weights - emphasize theological and interpretive layers for Islamic texts
            layer_weights = {
                SemanticLayer.LITERAL: 0.15,
                SemanticLayer.CULTURAL: 0.20,
                SemanticLayer.THEOLOGICAL: 0.25,
                SemanticLayer.INTERPRETIVE: 0.25,
                SemanticLayer.CONTEXTUAL: 0.15
            }
        
        similarities = {}
        for layer in SemanticLayer:
            emb1 = embedding1.get_layer_embedding(layer)
            emb2 = embedding2.get_layer_embedding(layer)
            similarities[layer] = cosine_similarity([emb1], [emb2])[0][0]
        
        # Calculate weighted average
        weighted_similarity = sum(
            similarities[layer] * layer_weights[layer] 
            for layer in SemanticLayer
        )
        
        return weighted_similarity

class MatryoshkaRetrievalEngine:
    """Enhanced retrieval engine using Matryoshka semantic layers"""
    
    def __init__(self, semantic_analyzer: MatryoshkaSemanticAnalyzer):
        self.semantic_analyzer = semantic_analyzer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            min_df=2
        )
        
        # Store Matryoshka embeddings for sources
        self.quran_matryoshka_embeddings = {}
        self.hadith_matryoshka_embeddings = {}
    
    def precompute_source_embeddings(self, quran_sources: List[Source], hadith_sources: List[Source]):
        """Precompute Matryoshka embeddings for all sources"""
        logger.info("Precomputing Matryoshka embeddings for sources...")
        
        # Precompute Qur'an embeddings
        for source in quran_sources:
            self.quran_matryoshka_embeddings[source.id] = self.semantic_analyzer.create_matryoshka_embedding(source.text)
        
        # Precompute Hadith embeddings
        for source in hadith_sources:
            self.hadith_matryoshka_embeddings[source.id] = self.semantic_analyzer.create_matryoshka_embedding(source.text)
        
        logger.info(f"Precomputed embeddings for {len(self.quran_matryoshka_embeddings)} Qur'an sources and {len(self.hadith_matryoshka_embeddings)} Hadith sources")
    
    def retrieve_quran(self, question: Question, quran_sources: List[Source]) -> List[Source]:
        """Retrieve relevant Qur'anic passages using Matryoshka semantic layers"""
        if not quran_sources:
            return []
        
        candidates = []
        
        # Create Matryoshka embedding for the question
        question_embedding = self.semantic_analyzer.create_matryoshka_embedding(' '.join(question.keywords))
        
        # Calculate similarities using Matryoshka approach
        for source in quran_sources:
            if source.id in self.quran_matryoshka_embeddings:
                source_embedding = self.quran_matryoshka_embeddings[source.id]
                
                # Calculate weighted similarity across all layers
                similarity = self.semantic_analyzer.calculate_layer_similarity(
                    question_embedding, 
                    source_embedding,
                    self._get_quran_layer_weights(question)
                )
                
                if similarity > 0.1:  # Threshold for relevance
                    source.matryoshka_score = similarity
                    candidates.append(source)
        
        return sorted(candidates, key=lambda x: getattr(x, 'matryoshka_score', 0), reverse=True)
    
    def retrieve_hadith(self, question: Question, hadith_sources: List[Source]) -> List[Source]:
        """Retrieve relevant Hadiths using Matryoshka semantic layers"""
        if not hadith_sources:
            return []
        
        candidates = []
        
        # Create Matryoshka embedding for the question
        question_embedding = self.semantic_analyzer.create_matryoshka_embedding(' '.join(question.keywords))
        
        # Calculate similarities using Matryoshka approach
        for source in hadith_sources:
            if source.id in self.hadith_matryoshka_embeddings:
                source_embedding = self.hadith_matryoshka_embeddings[source.id]
                
                # Calculate weighted similarity across all layers
                similarity = self.semantic_analyzer.calculate_layer_similarity(
                    question_embedding, 
                    source_embedding,
                    self._get_hadith_layer_weights(question)
                )
                
                if similarity > 0.1:  # Threshold for relevance
                    source.matryoshka_score = similarity
                    candidates.append(source)
        
        return sorted(candidates, key=lambda x: getattr(x, 'matryoshka_score', 0), reverse=True)
    
    def _get_quran_layer_weights(self, question: Question) -> Dict[SemanticLayer, float]:
        """Get layer weights for Qur'an retrieval based on question type"""
        if question.question_type == 'factoid':
            # For factoid questions, emphasize literal and cultural layers
            return {
                SemanticLayer.LITERAL: 0.25,
                SemanticLayer.CULTURAL: 0.25,
                SemanticLayer.THEOLOGICAL: 0.20,
                SemanticLayer.INTERPRETIVE: 0.20,
                SemanticLayer.CONTEXTUAL: 0.10
            }
        else:
            # For non-factoid questions, emphasize theological and interpretive layers
            return {
                SemanticLayer.LITERAL: 0.15,
                SemanticLayer.CULTURAL: 0.20,
                SemanticLayer.THEOLOGICAL: 0.25,
                SemanticLayer.INTERPRETIVE: 0.25,
                SemanticLayer.CONTEXTUAL: 0.15
            }
    
    def _get_hadith_layer_weights(self, question: Question) -> Dict[SemanticLayer, float]:
        """Get layer weights for Hadith retrieval based on question type"""
        if question.question_type == 'factoid':
            # For factoid questions, emphasize literal and cultural layers
            return {
                SemanticLayer.LITERAL: 0.20,
                SemanticLayer.CULTURAL: 0.30,
                SemanticLayer.THEOLOGICAL: 0.20,
                SemanticLayer.INTERPRETIVE: 0.20,
                SemanticLayer.CONTEXTUAL: 0.10
            }
        else:
            # For non-factoid questions, emphasize interpretive and contextual layers
            return {
                SemanticLayer.LITERAL: 0.15,
                SemanticLayer.CULTURAL: 0.20,
                SemanticLayer.THEOLOGICAL: 0.20,
                SemanticLayer.INTERPRETIVE: 0.25,
                SemanticLayer.CONTEXTUAL: 0.20
            }

class MatryoshkaRankingEngine:
    """Enhanced ranking engine using Matryoshka semantic layers"""
    
    def __init__(self, semantic_analyzer: MatryoshkaSemanticAnalyzer):
        self.semantic_analyzer = semantic_analyzer
        self.ranking_weights = {
            'matryoshka_score': 0.5,
            'keyword_overlap': 0.2,
            'entity_match': 0.2,
            'source_type_preference': 0.1
        }
    
    def rank_sources(self, question: Question, candidates: List[Source]) -> List[Source]:
        """Rank sources using Matryoshka semantic analysis"""
        for candidate in candidates:
            score = self._calculate_matryoshka_score(question, candidate)
            candidate.final_score = score
        
        return sorted(candidates, key=lambda x: x.final_score, reverse=True)
    
    def _calculate_matryoshka_score(self, question: Question, source: Source) -> float:
        """Calculate composite score using Matryoshka semantic analysis"""
        scores = {}
        
        # Matryoshka semantic score
        scores['matryoshka_score'] = getattr(source, 'matryoshka_score', 0.0)
        
        # Keyword overlap
        scores['keyword_overlap'] = self._calculate_keyword_overlap(question, source)
        
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
        theological_keywords = ['الله', 'القرآن', 'الإيمان', 'الإسلام', 'العبادة']
        
        if question.question_type == 'factoid':
            return 0.5  # Neutral preference
        else:
            if any(keyword in question.text for keyword in theological_keywords):
                return 1.0 if source.source_type == 'quran' else 0.3
            else:
                return 0.5

class MatryoshkaQuranQASystem(QuranQASystem):
    """
    Enhanced Qur'an QA system with Matryoshka semantic layers
    """
    
    def __init__(self, 
                 quran_data_path: str = None, 
                 hadith_data_path: str = None,
                 model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialize the Matryoshka-enhanced system
        
        Args:
            quran_data_path: Path to Qur'anic passages data
            hadith_data_path: Path to Sahih Bukhari Hadiths data
            model_name: Name of the sentence transformer model to use
        """
        super().__init__(quran_data_path, hadith_data_path)
        
        # Initialize Matryoshka semantic analyzer
        self.semantic_analyzer = MatryoshkaSemanticAnalyzer(model_name)
        
        # Enhanced retrieval and ranking engines
        self.matryoshka_retriever = MatryoshkaRetrievalEngine(self.semantic_analyzer)
        self.matryoshka_ranker = MatryoshkaRankingEngine(self.semantic_analyzer)
        
        # Load data using the data loader
        self.data_loader = DataLoader()
        self._load_data()
        
        # Precompute embeddings if data is available
        if self.quran_sources or self.hadith_sources:
            self.matryoshka_retriever.precompute_source_embeddings(self.quran_sources, self.hadith_sources)
    
    def _load_data(self):
        """Load Qur'an and Hadith data"""
        if not self.quran_sources:
            try:
                self.quran_sources, self.hadith_sources = self.data_loader.create_sample_data()
                logger.info("Loaded sample data")
            except Exception as e:
                logger.error(f"Error loading sample data: {e}")
        
        logger.info(f"Loaded {len(self.quran_sources)} Qur'anic sources")
        logger.info(f"Loaded {len(self.hadith_sources)} Hadith sources")
    
    def retrieve_sources(self, question: Question, max_results: int = 20) -> List[Source]:
        """
        Enhanced retrieval using Matryoshka semantic layers
        """
        # Use Matryoshka retriever
        quran_candidates = self.matryoshka_retriever.retrieve_quran(question, self.quran_sources)
        hadith_candidates = self.matryoshka_retriever.retrieve_hadith(question, self.hadith_sources)
        
        # Combine candidates
        all_candidates = quran_candidates + hadith_candidates
        
        if not all_candidates:
            logger.info("No relevant sources found for the question")
            return []
        
        # Rank using Matryoshka approach
        ranked_sources = self.matryoshka_ranker.rank_sources(question, all_candidates)
        
        return ranked_sources[:max_results]
    
    def analyze_semantic_layers(self, text: str) -> Dict[SemanticLayer, float]:
        """
        Analyze the semantic layer distribution of a text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with confidence scores for each semantic layer
        """
        embedding = self.semantic_analyzer.create_matryoshka_embedding(text)
        
        # Calculate layer-specific characteristics
        layer_analysis = {}
        for layer in SemanticLayer:
            layer_embedding = embedding.get_layer_embedding(layer)
            # Use embedding magnitude as a proxy for layer strength
            layer_analysis[layer] = float(np.linalg.norm(layer_embedding))
        
        return layer_analysis
    
    def get_layer_explanation(self, question: Question, source: Source) -> Dict[SemanticLayer, str]:
        """
        Get explanations for why a source was selected based on each semantic layer
        
        Args:
            question: The original question
            source: The selected source
            
        Returns:
            Dictionary with explanations for each semantic layer
        """
        question_embedding = self.semantic_analyzer.create_matryoshka_embedding(' '.join(question.keywords))
        source_embedding = self.semantic_analyzer.create_matryoshka_embedding(source.text)
        
        explanations = {}
        for layer in SemanticLayer:
            emb1 = question_embedding.get_layer_embedding(layer)
            emb2 = source_embedding.get_layer_embedding(layer)
            similarity = cosine_similarity([emb1], [emb2])[0][0]
            
            explanations[layer] = f"Layer similarity: {similarity:.3f}"
        
        return explanations

def main():
    """Main function to demonstrate the Matryoshka-enhanced system"""
    
    # Initialize the Matryoshka-enhanced system
    system = MatryoshkaQuranQASystem()
    
    # Test questions
    test_questions = [
        "ما هي أركان الإسلام الخمسة؟",
        "من هو النبي محمد؟",
        "كيف نصلي؟",
        "ما هي شروط الحج؟",
        "هل يوجد إجابة لهذا السؤال في القرآن؟"
    ]
    
    print("=== Matryoshka-Enhanced Qur'an QA System Demo ===\n")
    
    for i, question in enumerate(test_questions, 1):
        print(f"Test Case {i}: {question}")
        print("-" * 60)
        
        # Process question
        processed_question = system.process_question(question)
        print(f"Question Type: {processed_question.question_type}")
        print(f"Keywords: {', '.join(processed_question.keywords)}")
        print(f"Entities: {', '.join(processed_question.entities)}")
        
        # Analyze semantic layers
        layer_analysis = system.analyze_semantic_layers(question)
        print(f"\nSemantic Layer Analysis:")
        for layer, score in layer_analysis.items():
            print(f"  {layer.value}: {score:.3f}")
        
        # Get answer
        result = system.answer_question(question)
        print(f"Has Answer: {result['has_answer']}")
        
        if result['sources']:
            print(f"\nTop Sources:")
            for j, source in enumerate(result['sources'][:3]):  # Show top 3
                print(f"{j+1}. [{source['source_type'].upper()}] {source['text'][:80]}...")
                print(f"   Score: {source.get('relevance_score', 0.0):.3f}")
                
                # Get layer explanations for the top source
                if j == 0:
                    explanations = system.get_layer_explanation(processed_question, Source(
                        id=source['id'],
                        text=source['text'],
                        source_type=source['source_type']
                    ))
                    print(f"   Layer Explanations:")
                    for layer, explanation in explanations.items():
                        print(f"     {layer.value}: {explanation}")
        else:
            print("No relevant sources found.")
        
        print("\n" + "="*80 + "\n")
    
    # Demonstrate semantic layer analysis
    print("=== Semantic Layer Analysis Demo ===")
    sample_texts = [
        "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
        "إنما الأعمال بالنيات",
        "بني الإسلام على خمس"
    ]
    
    for text in sample_texts:
        print(f"\nText: {text}")
        layer_analysis = system.analyze_semantic_layers(text)
        for layer, score in layer_analysis.items():
            print(f"  {layer.value}: {score:.3f}")

if __name__ == "__main__":
    main()