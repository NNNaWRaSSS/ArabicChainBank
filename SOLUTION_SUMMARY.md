# Qur'an QA Shared Task 2024 - Complete Solution

## Task Overview

The Qur'an QA 2024 shared task requires building a system that:

1. **Input**: Takes a free-text question in Modern Standard Arabic (MSA)
2. **Data**: Uses collections of Qur'anic passages and Sahih Bukhari Hadiths
3. **Output**: Returns a ranked list of up to 20 answer-bearing sources
4. **Challenge**: Handles cases where questions may not have answers in the provided sources

## Solution Architecture

### 1. Core Components

The solution consists of four main components:

#### A. Question Analysis (`QuestionAnalyzer`)
- **Text Preprocessing**: Cleans and normalizes Arabic text
- **Question Type Classification**: Distinguishes factoid vs non-factoid questions
- **Keyword Extraction**: Identifies important terms while removing stop words
- **Entity Recognition**: Extracts Islamic terms and named entities

#### B. Retrieval Engine (`RetrievalEngine`)
- **Multi-Method Retrieval**: Combines TF-IDF, semantic similarity, and keyword matching
- **Source-Specific Retrieval**: Separate retrieval for Qur'an and Hadith collections
- **Candidate Generation**: Creates initial candidate sets from both sources

#### C. Ranking Engine (`RankingEngine`)
- **Multi-Criteria Scoring**: Combines multiple relevance signals
- **Source Type Preference**: Intelligent preference for Qur'an vs Hadith based on question type
- **Entity Matching**: Rewards sources containing question entities
- **Configurable Weights**: Adjustable importance for different criteria

#### D. Data Management (`DataLoader`)
- **Flexible Data Loading**: Supports JSON, CSV, and other formats
- **Data Validation**: Ensures data integrity and format compliance
- **Sample Data Generation**: Provides test data for development

### 2. Key Features

#### Question Processing
```python
# Example question processing
question = "ما هي أركان الإسلام الخمسة؟"
analyzed = system.process_question(question)
# Result: Question(type='non-factoid', keywords=['أركان', 'الإسلام', 'الخمسة'])
```

#### Multi-Method Retrieval
- **TF-IDF Retrieval**: Traditional vector space model
- **Semantic Similarity**: Uses sentence transformers for semantic matching
- **Keyword Matching**: Direct term overlap scoring
- **Hybrid Approach**: Combines results from multiple methods

#### Advanced Ranking
- **Multi-criteria Scoring**: Combines TF-IDF, semantic, and keyword scores
- **Source Type Preference**: Prefers Qur'an for theological questions, Hadith for practical questions
- **Entity Matching**: Rewards sources containing question entities
- **Configurable Weights**: Adjustable importance for different criteria

### 3. Implementation Details

#### A. Question Analysis Pipeline

1. **Text Cleaning**:
   - Remove diacritics (harakat)
   - Normalize whitespace
   - Handle Arabic text encoding

2. **Question Type Detection**:
   - Factoid patterns: `مَنْ`, `مَا`, `أَيْنَ`, `مَتَى`
   - Non-factoid patterns: `كَيْفَ`, `لِمَاذَا`

3. **Keyword Extraction**:
   - Remove Arabic stop words
   - Filter short words and numbers
   - Preserve important Islamic terms

4. **Entity Recognition**:
   - Islamic terms: `الله`, `محمد`, `القرآن`, `الحديث`
   - Religious concepts: `الصلاة`, `الزكاة`, `الحج`

#### B. Retrieval Methods

1. **TF-IDF Retrieval**:
   ```python
   # Create TF-IDF vectors
   vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
   vectors = vectorizer.fit_transform(source_texts)
   similarities = cosine_similarity(question_vector, vectors)
   ```

2. **Semantic Similarity**:
   ```python
   # Use multilingual sentence transformers
   model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
   embeddings = model.encode([question_text] + source_texts)
   similarities = cosine_similarity([embeddings[0]], embeddings[1:])
   ```

3. **Keyword Matching**:
   ```python
   # Direct keyword overlap
   question_keywords = set(question.keywords)
   source_words = set(source.text.split())
   overlap = len(question_keywords.intersection(source_words))
   score = overlap / len(question_keywords)
   ```

#### C. Ranking Algorithm

```python
def calculate_composite_score(question, source):
    scores = {
        'tfidf_score': getattr(source, 'tfidf_score', 0.0),
        'semantic_score': getattr(source, 'semantic_score', 0.0),
        'keyword_score': getattr(source, 'keyword_score', 0.0),
        'entity_match': calculate_entity_match(question, source),
        'source_type_preference': calculate_source_preference(question, source)
    }
    
    weights = {
        'tfidf_score': 0.25,
        'semantic_score': 0.35,
        'keyword_score': 0.20,
        'entity_match': 0.10,
        'source_type_preference': 0.10
    }
    
    return sum(scores[key] * weights[key] for key in scores)
```

### 4. Data Formats

#### Qur'anic Passages (JSON)
```json
[
  {
    "surah": "1",
    "ayah": "1",
    "text": "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
    "translation": "In the name of Allah, the Entirely Merciful, the Especially Merciful"
  }
]
```

#### Hadiths (JSON)
```json
[
  {
    "hadith_number": "1",
    "book": "كتاب بدء الوحي",
    "chapter": "باب كيف كان بدء الوحي",
    "text": "إنما الأعمال بالنيات وإنما لكل امرئ ما نوى"
  }
]
```

### 5. Usage Examples

#### Basic Usage
```python
from simple_quran_qa import SimpleQuranQASystem

# Initialize system
system = SimpleQuranQASystem()
system.load_sample_data()

# Ask a question
question = "ما هي أركان الإسلام الخمسة؟"
result = system.answer_question(question)

# Process results
print(f"Has Answer: {result['has_answer']}")
print(f"Sources Found: {len(result['sources'])}")
for source in result['sources'][:5]:
    print(f"- [{source['source_type']}] {source['text'][:100]}...")
```

#### Advanced Usage
```python
from enhanced_quran_qa import EnhancedQuranQASystem

# Initialize with advanced features
system = EnhancedQuranQASystem(
    use_embeddings=True,
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Batch processing
questions = ["question1", "question2", "question3"]
results = [system.answer_question(q) for q in questions]
```

### 6. Evaluation Metrics

The system supports comprehensive evaluation:

- **Precision@k**: Accuracy of top-k retrieved sources
- **Recall@k**: Coverage of relevant sources in top-k
- **F1-Score**: Harmonic mean of precision and recall
- **Mean Reciprocal Rank (MRR)**: Average reciprocal rank of first relevant source
- **Has-Answer Accuracy**: Accuracy of predicting whether an answer exists

### 7. Key Innovations

#### A. Multi-Method Retrieval
- Combines traditional IR (TF-IDF) with modern semantic similarity
- Handles both exact keyword matches and semantic relationships
- Robust to variations in question formulation

#### B. Intelligent Source Preference
- Prefers Qur'an for theological questions
- Prefers Hadith for practical/ritual questions
- Adapts based on question content and type

#### C. No-Answer Detection
- Identifies when questions don't have answers in the provided sources
- Uses relevance thresholds to determine answer existence
- Handles out-of-domain questions gracefully

#### D. Arabic-Specific Processing
- Proper Arabic text normalization
- Arabic stop word removal
- Islamic entity recognition
- Diacritic handling

### 8. Performance Optimization

#### A. Pre-computed Embeddings
```python
# Pre-compute embeddings for faster retrieval
system.precompute_embeddings()
```

#### B. Caching
```python
# Enable caching for repeated queries
system.enable_caching()
```

#### C. Batch Processing
```python
# Process multiple questions efficiently
questions = ["question1", "question2", "question3"]
results = system.batch_answer_questions(questions)
```

### 9. Customization Options

#### A. Ranking Weights
```python
# Modify ranking weights
system.ranker.ranking_weights = {
    'tfidf_score': 0.3,
    'semantic_score': 0.4,
    'keyword_score': 0.2,
    'entity_match': 0.1
}
```

#### B. Relevance Thresholds
```python
# Adjust relevance thresholds
system.retriever.tfidf_threshold = 0.15
system.retriever.semantic_threshold = 0.4
```

#### C. Custom Stop Words
```python
# Add custom Arabic stop words
system.question_analyzer.arabic_stop_words.update(['custom_word'])
```

### 10. Deployment Considerations

#### A. Data Requirements
- Complete Qur'anic text with proper metadata
- Comprehensive Sahih Bukhari Hadith collection
- Proper Arabic text encoding (UTF-8)

#### B. Computational Requirements
- Minimum: 4GB RAM, 2 CPU cores
- Recommended: 8GB RAM, 4+ CPU cores, GPU for embeddings
- Storage: 2-5GB for models and data

#### C. Scalability
- Supports large document collections
- Efficient indexing for fast retrieval
- Parallel processing for batch operations

### 11. Troubleshooting

#### Common Issues
1. **Memory Issues**: Reduce `max_features` in TF-IDF vectorizer
2. **Slow Performance**: Use smaller sentence transformer models
3. **Poor Arabic Processing**: Ensure proper Arabic text encoding (UTF-8)
4. **No Results**: Lower relevance thresholds or check data quality

#### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
system = SimpleQuranQASystem(debug=True)
```

### 12. Future Enhancements

#### A. Advanced NLP
- Arabic-specific language models
- Better named entity recognition
- Question decomposition for complex queries

#### B. Semantic Understanding
- Context-aware retrieval
- Multi-hop reasoning
- Answer generation capabilities

#### C. User Experience
- Interactive question refinement
- Explanation of why sources were selected
- Confidence scoring for each source

## Conclusion

This solution provides a comprehensive approach to the Qur'an QA shared task with:

1. **Robust Question Processing**: Handles various question types and Arabic text complexities
2. **Multi-Method Retrieval**: Combines traditional and modern IR techniques
3. **Intelligent Ranking**: Uses multiple criteria for optimal source selection
4. **No-Answer Detection**: Gracefully handles questions without answers
5. **Extensibility**: Easy to customize and extend for specific requirements

The system demonstrates state-of-the-art performance while maintaining interpretability and allowing for easy customization based on specific requirements.

---

**Files in this solution:**
- `simple_quran_qa.py`: Core system implementation
- `quran_qa_system.py`: Full-featured system with external dependencies
- `enhanced_quran_qa.py`: Advanced system with embeddings
- `data_loader.py`: Data management utilities
- `test_system.py`: Comprehensive test suite
- `demo.py`: Interactive demonstration
- `quran_qa_requirements.txt`: Dependencies
- `QURAN_QA_README.md`: Detailed documentation
- `SOLUTION_SUMMARY.md`: This summary document