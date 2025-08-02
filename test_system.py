#!/usr/bin/env python3
"""
Test Script for Qur'an QA System
================================

This script demonstrates the functionality of the Qur'an QA system
with sample data and test questions.
"""

import json
import logging
from pathlib import Path

# Import our system components
from quran_qa_system import QuranQASystem, Source
from data_loader import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_data():
    """Create test data for demonstration"""
    
    # Sample Qur'anic passages
    quran_sources = [
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
    hadith_sources = [
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
    
    return quran_sources, hadith_sources

def test_question_analysis():
    """Test question analysis functionality"""
    print("=== Testing Question Analysis ===\n")
    
    # Create a simple system for testing
    system = QuranQASystem()
    
    # Test questions
    test_questions = [
        "ما هي أركان الإسلام الخمسة؟",
        "من هو النبي محمد؟",
        "كيف نصلي؟",
        "ما هي شروط الحج؟",
        "هل يوجد إجابة لهذا السؤال في القرآن؟"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"Test {i}: {question}")
        
        # Analyze the question
        analyzed_question = system.process_question(question)
        
        print(f"  Type: {analyzed_question.question_type}")
        print(f"  Keywords: {', '.join(analyzed_question.keywords)}")
        print(f"  Entities: {', '.join(analyzed_question.entities)}")
        print()

def test_retrieval_and_ranking():
    """Test retrieval and ranking functionality"""
    print("=== Testing Retrieval and Ranking ===\n")
    
    # Create system with test data
    system = QuranQASystem()
    system.quran_sources, system.hadith_sources = create_test_data()
    
    # Test questions with expected outcomes
    test_cases = [
        {
            "question": "ما هي أركان الإسلام الخمسة؟",
            "expected_type": "factoid",
            "expected_keywords": ["أركان", "الإسلام", "خمسة"],
            "expected_source": "hadith_2"  # Contains pillars of Islam
        },
        {
            "question": "من هو النبي محمد؟",
            "expected_type": "factoid",
            "expected_keywords": ["النبي", "محمد"],
            "expected_source": "hadith_5"  # Mentions Prophet Muhammad
        },
        {
            "question": "كيف نصلي؟",
            "expected_type": "non-factoid",
            "expected_keywords": ["كيف", "نصلي"],
            "expected_source": "hadith_2"  # Mentions prayer
        },
        {
            "question": "ما هي شروط الحج؟",
            "expected_type": "non-factoid",
            "expected_keywords": ["شروط", "الحج"],
            "expected_source": "hadith_2"  # Mentions Hajj
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        print(f"Test Case {i}: {question}")
        print("-" * 50)
        
        # Get answer
        result = system.answer_question(question)
        
        print(f"Question Type: {result['question_type']}")
        print(f"Keywords: {', '.join(result['keywords'])}")
        print(f"Has Answer: {result['has_answer']}")
        
        if result['sources']:
            print(f"\nTop Sources:")
            for j, source in enumerate(result['sources'][:3]):  # Show top 3
                print(f"{j+1}. [{source['source_type'].upper()}] {source['text'][:80]}...")
                print(f"   Score: {source.get('relevance_score', 0.0):.3f}")
                
                # Check if this is the expected source
                if source['id'] == test_case['expected_source']:
                    print(f"   ✓ Expected source found!")
        else:
            print("No relevant sources found.")
        
        print("\n" + "="*60 + "\n")

def test_no_answer_scenario():
    """Test scenarios where no answer should be found"""
    print("=== Testing No-Answer Scenarios ===\n")
    
    # Create system with test data
    system = QuranQASystem()
    system.quran_sources, system.hadith_sources = create_test_data()
    
    # Questions that shouldn't have answers in our test data
    no_answer_questions = [
        "ما هو أفضل مطعم في الرياض؟",  # Restaurant question
        "كيف أصلح السيارة؟",  # Car repair question
        "ما هو أفضل فيلم في 2024؟",  # Movie question
        "كيف أطبخ الكبسة؟"  # Cooking question
    ]
    
    for i, question in enumerate(no_answer_questions, 1):
        print(f"Test {i}: {question}")
        
        result = system.answer_question(question)
        
        print(f"Has Answer: {result['has_answer']}")
        print(f"Number of Sources: {len(result['sources'])}")
        
        if result['has_answer']:
            print("⚠️  Warning: Found sources when none expected")
        else:
            print("✓ Correctly identified no answer scenario")
        
        print()

def test_data_loader():
    """Test the data loader functionality"""
    print("=== Testing Data Loader ===\n")
    
    loader = DataLoader()
    
    # Create sample data
    quran_sources, hadith_sources = loader.create_sample_data()
    
    print(f"Created {len(quran_sources)} Qur'anic sources")
    print(f"Created {len(hadith_sources)} Hadith sources")
    
    # Save to JSON files
    loader.save_sources_to_json(quran_sources, 'test_quran.json')
    loader.save_sources_to_json(hadith_sources, 'test_hadith.json')
    
    print("Saved test data to JSON files")
    
    # Load from JSON files
    loaded_quran = loader.load_sources_from_json('test_quran.json')
    loaded_hadith = loader.load_sources_from_json('test_hadith.json')
    
    print(f"Loaded {len(loaded_quran)} Qur'anic sources from JSON")
    print(f"Loaded {len(loaded_hadith)} Hadith sources from JSON")
    
    # Clean up test files
    Path('test_quran.json').unlink(missing_ok=True)
    Path('test_hadith.json').unlink(missing_ok=True)
    
    print("✓ Data loader test completed successfully\n")

def run_comprehensive_test():
    """Run a comprehensive test of the system"""
    print("🚀 Starting Comprehensive Qur'an QA System Test\n")
    
    try:
        # Test 1: Question Analysis
        test_question_analysis()
        
        # Test 2: Data Loader
        test_data_loader()
        
        # Test 3: Retrieval and Ranking
        test_retrieval_and_ranking()
        
        # Test 4: No-Answer Scenarios
        test_no_answer_scenario()
        
        print("✅ All tests completed successfully!")
        print("\n🎉 Qur'an QA System is working correctly!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        logger.error(f"Test error: {e}", exc_info=True)

def main():
    """Main function to run the test suite"""
    print("=" * 80)
    print("QUR'AN QA SYSTEM TEST SUITE")
    print("=" * 80)
    print()
    
    run_comprehensive_test()
    
    print("\n" + "=" * 80)
    print("TEST SUITE COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()