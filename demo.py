#!/usr/bin/env python3
"""
Demo Script for Qur'an QA System
================================

This script provides an interactive demo of the Qur'an QA system.
Users can ask questions in Arabic and see the system's responses.
"""

import sys
from quran_qa_system import QuranQASystem
from data_loader import DataLoader

def print_banner():
    """Print a welcome banner"""
    print("=" * 80)
    print("🎯 QUR'AN QA SYSTEM DEMO")
    print("=" * 80)
    print()
    print("This demo allows you to ask questions in Arabic and see")
    print("how the system retrieves relevant Qur'anic passages and Hadiths.")
    print()
    print("Example questions you can try:")
    print("• ما هي أركان الإسلام الخمسة؟")
    print("• من هو النبي محمد؟")
    print("• كيف نصلي؟")
    print("• ما هي شروط الحج؟")
    print("• هل يوجد إجابة لهذا السؤال في القرآن؟")
    print()
    print("Type 'quit' or 'exit' to end the demo.")
    print("Type 'help' for more information.")
    print()

def print_help():
    """Print help information"""
    print("\n📖 HELP")
    print("-" * 40)
    print("Commands:")
    print("  help     - Show this help message")
    print("  quit     - Exit the demo")
    print("  exit     - Exit the demo")
    print("  clear    - Clear the screen")
    print()
    print("Question Types:")
    print("  • Factoid questions (ما، من، أين، متى)")
    print("  • Non-factoid questions (كيف، لماذا)")
    print()
    print("The system will:")
    print("  1. Analyze your question")
    print("  2. Retrieve relevant sources")
    print("  3. Rank them by relevance")
    print("  4. Show up to 20 most relevant sources")
    print()

def display_result(result):
    """Display the system's response in a formatted way"""
    print("\n" + "=" * 80)
    print("🔍 ANALYSIS RESULTS")
    print("=" * 80)
    
    print(f"Question: {result['question']}")
    print(f"Type: {result['question_type']}")
    print(f"Keywords: {', '.join(result['keywords'])}")
    print(f"Entities: {', '.join(result['entities'])}")
    print(f"Has Answer: {'Yes' if result['has_answer'] else 'No'}")
    
    if result['sources']:
        print(f"\n📚 TOP SOURCES ({len(result['sources'])} found)")
        print("-" * 60)
        
        for i, source in enumerate(result['sources'][:10], 1):  # Show top 10
            print(f"\n{i}. [{source['source_type'].upper()}]")
            print(f"   ID: {source['id']}")
            print(f"   Text: {source['text']}")
            print(f"   Score: {source.get('relevance_score', 0.0):.3f}")
            
            if source['source_type'] == 'quran':
                print(f"   Surah: {source.get('surah', 'N/A')}, Ayah: {source.get('ayah', 'N/A')}")
            elif source['source_type'] == 'hadith':
                print(f"   Book: {source.get('book', 'N/A')}")
                print(f"   Chapter: {source.get('chapter', 'N/A')}")
    else:
        print("\n❌ No relevant sources found.")
        print("This might mean:")
        print("  • The question doesn't have an answer in the provided sources")
        print("  • The question is outside the scope of Islamic texts")
        print("  • The system needs more data to answer this question")
    
    print("\n" + "=" * 80)

def interactive_demo():
    """Run the interactive demo"""
    print_banner()
    
    # Initialize the system
    print("🔄 Initializing Qur'an QA System...")
    system = QuranQASystem()
    
    # Load sample data
    data_loader = DataLoader()
    system.quran_sources, system.hadith_sources = data_loader.create_sample_data()
    
    print(f"✅ Loaded {len(system.quran_sources)} Qur'anic sources")
    print(f"✅ Loaded {len(system.hadith_sources)} Hadith sources")
    print("✅ System ready!\n")
    
    # Main interaction loop
    while True:
        try:
            # Get user input
            question = input("\n❓ Enter your question in Arabic: ").strip()
            
            # Handle commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using the Qur'an QA System Demo!")
                break
            elif question.lower() in ['help', 'h']:
                print_help()
                continue
            elif question.lower() in ['clear', 'cls']:
                import os
                os.system('clear' if os.name == 'posix' else 'cls')
                print_banner()
                continue
            elif not question:
                print("⚠️  Please enter a question.")
                continue
            
            # Process the question
            print(f"\n🔄 Processing: {question}")
            result = system.answer_question(question)
            
            # Display results
            display_result(result)
            
        except KeyboardInterrupt:
            print("\n\n👋 Demo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error processing question: {e}")
            print("Please try again with a different question.")

def batch_demo():
    """Run a batch demo with predefined questions"""
    print("🔄 Running Batch Demo...\n")
    
    # Initialize system
    system = QuranQASystem()
    data_loader = DataLoader()
    system.quran_sources, system.hadith_sources = data_loader.create_sample_data()
    
    # Predefined test questions
    test_questions = [
        "ما هي أركان الإسلام الخمسة؟",
        "من هو النبي محمد؟",
        "كيف نصلي؟",
        "ما هي شروط الحج؟",
        "هل يوجد إجابة لهذا السؤال في القرآن؟",
        "ما هو أفضل مطعم في الرياض؟"  # Should return no answer
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*20} Question {i} {'='*20}")
        print(f"Question: {question}")
        
        result = system.answer_question(question)
        
        print(f"Type: {result['question_type']}")
        print(f"Has Answer: {result['has_answer']}")
        print(f"Sources Found: {len(result['sources'])}")
        
        if result['sources']:
            print("Top Source:")
            top_source = result['sources'][0]
            print(f"  [{top_source['source_type'].upper()}] {top_source['text'][:100]}...")
        
        print()

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == 'batch':
        batch_demo()
    else:
        interactive_demo()

if __name__ == "__main__":
    main()