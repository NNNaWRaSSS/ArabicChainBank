#!/usr/bin/env python3
"""
Matryoshka Layers Demo for Qur'an QA System
===========================================

This script demonstrates the multi-layered semantic approach using Matryoshka layers
to capture different levels of meaning in Islamic texts.
"""

import sys
import numpy as np
from simple_quran_qa import SimpleQuranQASystem, Source, Question
from enhanced_quran_qa import (
    MatryoshkaSemanticAnalyzer, 
    MatryoshkaEmbedding, 
    SemanticLayer,
    MatryoshkaQuranQASystem
)

def print_matryoshka_explanation():
    """Print explanation of the Matryoshka layers approach"""
    print("=" * 80)
    print("🎭 MATRYOSHKA LAYERS APPROACH")
    print("=" * 80)
    print()
    print("The Matryoshka layers approach captures multiple semantic levels:")
    print()
    print("1. LITERAL LAYER:")
    print("   - Direct word meanings and translations")
    print("   - Surface-level understanding")
    print("   - Example: 'الصلاة' = 'prayer'")
    print()
    print("2. CULTURAL LAYER:")
    print("   - Cultural and historical context")
    print("   - Traditional practices and customs")
    print("   - Example: 'الصلاة' = 'Islamic ritual prayer'")
    print()
    print("3. THEOLOGICAL LAYER:")
    print("   - Religious and spiritual meanings")
    print("   - Divine significance and purpose")
    print("   - Example: 'الصلاة' = 'Direct communication with Allah'")
    print()
    print("4. INTERPRETIVE LAYER:")
    print("   - Scholarly interpretations and jurisprudence")
    print("   - Legal and doctrinal understandings")
    print("   - Example: 'الصلاة' = 'Spiritual discipline and worship'")
    print()
    print("5. CONTEXTUAL LAYER:")
    print("   - Situational and contextual meanings")
    print("   - Practical applications and circumstances")
    print("   - Example: 'الصلاة' = 'Prescribed daily worship'")
    print()

def demonstrate_layer_generation():
    """Demonstrate how different semantic layers are generated"""
    print("=" * 80)
    print("🔍 LAYER GENERATION DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Initialize semantic analyzer
    analyzer = MatryoshkaSemanticAnalyzer()
    
    # Sample Islamic terms
    islamic_terms = [
        "الله",
        "الصلاة", 
        "الزكاة",
        "الحج",
        "القرآن"
    ]
    
    for term in islamic_terms:
        print(f"Term: {term}")
        print("-" * 40)
        
        # Generate layer-specific texts
        layer_texts = analyzer._generate_layer_texts(term)
        
        for layer in SemanticLayer:
            print(f"{layer.value.upper()}: {layer_texts[layer]}")
        
        print()
        print("=" * 60)
        print()

def demonstrate_semantic_mappings():
    """Demonstrate Islamic-specific semantic mappings"""
    print("=" * 80)
    print("📚 ISLAMIC SEMANTIC MAPPINGS")
    print("=" * 80)
    print()
    
    analyzer = MatryoshkaSemanticAnalyzer()
    
    for term, mappings in analyzer.islamic_semantic_mappings.items():
        print(f"Term: {term}")
        print("-" * 30)
        
        for layer, meaning in mappings.items():
            print(f"{layer.value}: {meaning}")
        
        print()
        print("=" * 50)
        print()

def demonstrate_layer_similarity():
    """Demonstrate layer-based similarity calculation"""
    print("=" * 80)
    print("🔗 LAYER SIMILARITY CALCULATION")
    print("=" * 80)
    print()
    
    analyzer = MatryoshkaSemanticAnalyzer()
    
    # Sample question-source pairs
    test_pairs = [
        ("ما هي الصلاة؟", "الصلاة هي العبادة المفروضة"),
        ("كيف نصلي؟", "إنما الأعمال بالنيات"),
        ("ما هو الحج؟", "الحج إلى البيت الحرام")
    ]
    
    for question, source in test_pairs:
        print(f"Question: {question}")
        print(f"Source: {source}")
        print("-" * 50)
        
        # Create embeddings
        question_embedding = analyzer.create_matryoshka_embedding(question)
        source_embedding = analyzer.create_matryoshka_embedding(source)
        
        # Calculate similarities for each layer
        for layer in SemanticLayer:
            emb1 = question_embedding.get_layer_embedding(layer)
            emb2 = source_embedding.get_layer_embedding(layer)
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            print(f"{layer.value}: {similarity:.3f}")
        
        # Calculate weighted similarity
        weighted_sim = analyzer.calculate_layer_similarity(question_embedding, source_embedding)
        print(f"Weighted similarity: {weighted_sim:.3f}")
        print()
        print("=" * 60)
        print()

def demonstrate_question_analysis():
    """Demonstrate semantic layer analysis for questions"""
    print("=" * 80)
    print("❓ QUESTION SEMANTIC ANALYSIS")
    print("=" * 80)
    print()
    
    # Initialize system
    system = MatryoshkaQuranQASystem()
    
    # Sample questions
    questions = [
        "ما هي أركان الإسلام الخمسة؟",
        "كيف نصلي؟",
        "ما هو الحج؟",
        "من هو النبي محمد؟",
        "ما هي شروط الزكاة؟"
    ]
    
    for question in questions:
        print(f"Question: {question}")
        print("-" * 40)
        
        # Analyze semantic layers
        layer_analysis = system.analyze_semantic_layers(question)
        
        # Sort layers by strength
        sorted_layers = sorted(layer_analysis.items(), key=lambda x: x[1], reverse=True)
        
        print("Semantic Layer Strengths:")
        for layer, strength in sorted_layers:
            print(f"  {layer.value}: {strength:.3f}")
        
        print()
        print("=" * 60)
        print()

def demonstrate_retrieval_with_layers():
    """Demonstrate retrieval using Matryoshka layers"""
    print("=" * 80)
    print("🎯 RETRIEVAL WITH MATRYOSHKA LAYERS")
    print("=" * 80)
    print()
    
    # Initialize system
    system = MatryoshkaQuranQASystem()
    
    # Test question
    question = "ما هي أركان الإسلام الخمسة؟"
    print(f"Question: {question}")
    print("-" * 50)
    
    # Process question
    processed_question = system.process_question(question)
    print(f"Question Type: {processed_question.question_type}")
    print(f"Keywords: {', '.join(processed_question.keywords)}")
    print(f"Entities: {', '.join(processed_question.entities)}")
    
    # Get answer
    result = system.answer_question(question)
    print(f"Has Answer: {result['has_answer']}")
    
    if result['sources']:
        print(f"\nTop Sources with Layer Analysis:")
        for i, source in enumerate(result['sources'][:2]):  # Show top 2
            print(f"\n{i+1}. [{source['source_type'].upper()}] {source['text']}")
            print(f"   Score: {source.get('relevance_score', 0.0):.3f}")
            
            # Get layer explanations
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
    
    print()

def demonstrate_layer_weighting():
    """Demonstrate different layer weighting strategies"""
    print("=" * 80)
    print("⚖️ LAYER WEIGHTING STRATEGIES")
    print("=" * 80)
    print()
    
    analyzer = MatryoshkaSemanticAnalyzer()
    
    # Different weighting strategies
    strategies = {
        "Theological Focus": {
            SemanticLayer.LITERAL: 0.10,
            SemanticLayer.CULTURAL: 0.15,
            SemanticLayer.THEOLOGICAL: 0.35,
            SemanticLayer.INTERPRETIVE: 0.25,
            SemanticLayer.CONTEXTUAL: 0.15
        },
        "Cultural Focus": {
            SemanticLayer.LITERAL: 0.15,
            SemanticLayer.CULTURAL: 0.35,
            SemanticLayer.THEOLOGICAL: 0.20,
            SemanticLayer.INTERPRETIVE: 0.20,
            SemanticLayer.CONTEXTUAL: 0.10
        },
        "Balanced": {
            SemanticLayer.LITERAL: 0.20,
            SemanticLayer.CULTURAL: 0.20,
            SemanticLayer.THEOLOGICAL: 0.20,
            SemanticLayer.INTERPRETIVE: 0.20,
            SemanticLayer.CONTEXTUAL: 0.20
        }
    }
    
    # Test question and source
    question = "ما هي الصلاة؟"
    source = "الصلاة هي العبادة المفروضة"
    
    question_embedding = analyzer.create_matryoshka_embedding(question)
    source_embedding = analyzer.create_matryoshka_embedding(source)
    
    print(f"Question: {question}")
    print(f"Source: {source}")
    print()
    
    for strategy_name, weights in strategies.items():
        similarity = analyzer.calculate_layer_similarity(
            question_embedding, 
            source_embedding, 
            weights
        )
        print(f"{strategy_name}: {similarity:.3f}")
    
    print()

def main():
    """Main demonstration function"""
    print("🎭 MATRYOSHKA LAYERS DEMONSTRATION")
    print("=" * 80)
    print()
    
    try:
        # 1. Explain the approach
        print_matryoshka_explanation()
        
        # 2. Demonstrate layer generation
        demonstrate_layer_generation()
        
        # 3. Show semantic mappings
        demonstrate_semantic_mappings()
        
        # 4. Demonstrate similarity calculation
        demonstrate_layer_similarity()
        
        # 5. Show question analysis
        demonstrate_question_analysis()
        
        # 6. Demonstrate retrieval
        demonstrate_retrieval_with_layers()
        
        # 7. Show weighting strategies
        demonstrate_layer_weighting()
        
        print("✅ Matryoshka layers demonstration completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()