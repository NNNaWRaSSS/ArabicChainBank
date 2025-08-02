# Qur'an QA System for Shared Task 2024

## Task Definition

This system addresses the Qur'an QA 2024 shared task, which is defined as follows:

**Given**: A free-text question posed in Modern Standard Arabic (MSA), a collection of Qur'anic passages (covering the entire Holy Qur'an), and a collection of Hadiths from Sahih Bukhari.

**Required**: Retrieve a ranked list of up to 20 answer-bearing Qur'anic passages or Hadiths (Islamic sources that potentially contain answers to the given question).

**Key Features**:
- Supports both factoid and non-factoid questions
- Handles cases where questions may not have answers in the provided sources
- Returns no answers when no relevant sources are found
- Otherwise returns a ranked list of up to 20 answer-bearing sources

## System Architecture

The solution consists of several key components:

### 1. Question Analysis (`QuestionAnalyzer`)
- **Text Preprocessing**: Cleans and normalizes Arabic text
- **Question Type Classification**: Distinguishes between factoid and non-factoid questions
- **Keyword Extraction**: Identifies important terms from the question
- **Entity Recognition**: Extracts Islamic terms and named entities

### 2. Retrieval Engine (`EnhancedRetrievalEngine`)
- **TF-IDF Retrieval**: Traditional vector space model
- **Semantic Similarity**: Uses sentence transformers for semantic matching
- **Keyword-based Retrieval**: Direct keyword overlap matching
- **Multi-method Fusion**: Combines results from different retrieval methods

### 3. Ranking Engine (`EnhancedRankingEngine`)
- **Multi-criteria Scoring**: Combines multiple relevance signals
- **Source Type Preference**: Prefers Qur'an for theological questions, Hadith for practical questions
- **Entity Matching**: Rewards sources containing question entities
- **Weighted Ranking**: Uses configurable weights for different criteria

### 4. Data Management (`DataLoader`)
- **Flexible Data Loading**: Supports JSON, CSV, and other formats
- **Sample Data Generation**: Provides test data for development
- **Data Validation**: Ensures data integrity and format compliance

## Installation and Setup

### 1. Install Dependencies

```bash
pip install -r quran_qa_requirements.txt
```

### 2. Download Arabic NLP Resources

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### 3. Prepare Your Data

The system expects data in the following formats:

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

## Usage Examples

### Basic Usage

```python
from enhanced_quran_qa import EnhancedQuranQASystem

# Initialize the system
system = EnhancedQuranQASystem(
    quran_data_path='path/to/quran.json',
    hadith_data_path='path/to/hadith.json',
    use_embeddings=True
)

# Ask a question
question = "ما هي أركان الإسلام الخمسة؟"
result = system.answer_question(question)

# Process results
print(f"Question: {result['question']}")
print(f"Has Answer: {result['has_answer']}")
print(f"Number of Sources: {len(result['sources'])}")

for i, source in enumerate(result['sources'][:5]):  # Top 5
    print(f"{i+1}. [{source['source_type']}] {source['text'][:100]}...")
```

### Advanced Usage with Custom Configuration

```python
from enhanced_quran_qa import EnhancedQuranQASystem, EvaluationEngine

# Initialize with custom settings
system = EnhancedQuranQASystem(
    use_embeddings=True,
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Test multiple questions
questions = [
    "ما هي أركان الإسلام الخمسة؟",
    "من هو النبي محمد؟",
    "كيف نصلي؟",
    "ما هي شروط الحج؟"
]

for question in questions:
    result = system.answer_question(question)
    print(f"\nQuestion: {question}")
    print(f"Type: {result['question_type']}")
    print(f"Keywords: {', '.join(result['keywords'])}")
    print(f"Has Answer: {result['has_answer']}")
```

### Evaluation

```python
from enhanced_quran_qa import EvaluationEngine

# Create evaluation data
eval_data = [
    {
        "question": "ما هي أركان الإسلام الخمسة؟",
        "ground_truth": ["hadith_2"],
        "has_answer": True
    },
    {
        "question": "كيف نصلي؟",
        "ground_truth": ["hadith_1"],
        "has_answer": True
    }
]

# Evaluate system
evaluator = EvaluationEngine()
results = evaluator.evaluate_system(system, eval_data)

print("Evaluation Results:")
for metric, value in results.items():
    print(f"{metric}: {value:.3f}")
```

## System Features

### 1. Question Processing
- **Arabic Text Normalization**: Removes diacritics and normalizes text
- **Question Type Detection**: Identifies factoid vs non-factoid questions
- **Keyword Extraction**: Extracts meaningful terms while removing stop words
- **Entity Recognition**: Identifies Islamic terms and proper nouns

### 2. Multi-Method Retrieval
- **TF-IDF Retrieval**: Traditional information retrieval
- **Semantic Similarity**: Uses multilingual sentence transformers
- **Keyword Matching**: Direct term overlap scoring
- **Hybrid Approach**: Combines multiple retrieval methods

### 3. Advanced Ranking
- **Multi-criteria Scoring**: Combines TF-IDF, semantic, and keyword scores
- **Source Type Preference**: Intelligent preference for Qur'an vs Hadith
- **Entity Matching**: Rewards sources containing question entities
- **Configurable Weights**: Adjustable importance for different criteria

### 4. Answer Detection
- **Relevance Thresholding**: Only returns sources above relevance threshold
- **No-Answer Detection**: Identifies when no relevant sources exist
- **Confidence Scoring**: Provides confidence scores for each source

## Performance Optimization

### 1. Pre-computed Embeddings
```python
# Pre-compute embeddings for faster retrieval
system.precompute_embeddings()
```

### 2. Caching
```python
# Enable caching for repeated queries
system.enable_caching()
```

### 3. Batch Processing
```python
# Process multiple questions efficiently
questions = ["question1", "question2", "question3"]
results = system.batch_answer_questions(questions)
```

## Evaluation Metrics

The system supports comprehensive evaluation using:

- **Precision@k**: Accuracy of top-k retrieved sources
- **Recall@k**: Coverage of relevant sources in top-k
- **F1-Score**: Harmonic mean of precision and recall
- **Mean Reciprocal Rank (MRR)**: Average reciprocal rank of first relevant source
- **Has-Answer Accuracy**: Accuracy of predicting whether an answer exists

## Customization

### 1. Custom Ranking Weights
```python
# Modify ranking weights
system.enhanced_ranker.ranking_weights = {
    'tfidf_score': 0.3,
    'semantic_score': 0.4,
    'keyword_score': 0.2,
    'entity_match': 0.1
}
```

### 2. Custom Thresholds
```python
# Adjust relevance thresholds
system.enhanced_retriever.tfidf_threshold = 0.15
system.enhanced_retriever.semantic_threshold = 0.4
```

### 3. Custom Stop Words
```python
# Add custom Arabic stop words
system.question_analyzer.arabic_stop_words.update(['custom_word'])
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce `max_features` in TF-IDF vectorizer
2. **Slow Performance**: Use smaller sentence transformer models
3. **Poor Arabic Processing**: Ensure proper Arabic text encoding (UTF-8)
4. **No Results**: Lower relevance thresholds or check data quality

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
system = EnhancedQuranQASystem(debug=True)
```

## Contributing

To contribute to this system:

1. **Fork the repository**
2. **Create a feature branch**
3. **Add tests for new functionality**
4. **Ensure all tests pass**
5. **Submit a pull request**

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@inproceedings{quran-qa-2024,
  title={Enhanced Qur'an QA System for Shared Task 2024},
  author={Your Name},
  booktitle={Proceedings of the Qur'an QA Shared Task},
  year={2024}
}
```

## Contact

For questions or support, please contact: [your-email@example.com]

---

**Note**: This system is designed for research purposes and should be used responsibly. Always verify results against authoritative Islamic sources.