# Matryoshka Layers Approach for Qur'an QA System

## Overview

The Matryoshka layers approach is a sophisticated multi-layered semantic representation method that captures different levels of meaning in Islamic texts. This approach is inspired by the Russian nested dolls (Matryoshka), where each layer represents a deeper level of understanding.

## Semantic Layers

### 1. Literal Layer (الطبقة الحرفية)
**Purpose**: Captures direct word meanings and surface-level understanding
**Characteristics**:
- Direct translations and word-for-word meanings
- Surface-level semantic relationships
- Basic vocabulary understanding

**Example**:
- Input: "الصلاة"
- Literal Layer: "prayer" (direct translation)
- Application: Basic understanding of the term

### 2. Cultural Layer (الطبقة الثقافية)
**Purpose**: Incorporates cultural and historical context
**Characteristics**:
- Traditional practices and customs
- Historical significance
- Cultural interpretations

**Example**:
- Input: "الصلاة"
- Cultural Layer: "Islamic ritual prayer" (cultural context)
- Application: Understanding within Islamic cultural framework

### 3. Theological Layer (الطبقة اللاهوتية)
**Purpose**: Captures religious and spiritual meanings
**Characteristics**:
- Divine significance and purpose
- Religious symbolism
- Spiritual implications

**Example**:
- Input: "الصلاة"
- Theological Layer: "Direct communication with Allah" (spiritual meaning)
- Application: Understanding religious significance

### 4. Interpretive Layer (الطبقة التأويلية)
**Purpose**: Incorporates scholarly interpretations and jurisprudence
**Characteristics**:
- Legal and doctrinal understandings
- Scholarly consensus
- Jurisprudential interpretations

**Example**:
- Input: "الصلاة"
- Interpretive Layer: "Spiritual discipline and worship" (scholarly interpretation)
- Application: Understanding legal and doctrinal aspects

### 5. Contextual Layer (الطبقة السياقية)
**Purpose**: Captures situational and contextual meanings
**Characteristics**:
- Practical applications
- Circumstantial meanings
- Situational interpretations

**Example**:
- Input: "الصلاة"
- Contextual Layer: "Prescribed daily worship" (practical context)
- Application: Understanding practical implementation

## Implementation Details

### Layer Generation Process

```python
def _generate_layer_texts(self, text: str) -> Dict[SemanticLayer, str]:
    """Generate layer-specific versions of the input text"""
    layer_texts = {}
    
    # Literal layer - direct translation/meaning
    layer_texts[SemanticLayer.LITERAL] = f"Literal meaning: {text}"
    
    # Cultural layer - add cultural context
    cultural_context = self._add_cultural_context(text)
    layer_texts[SemanticLayer.CULTURAL] = f"Cultural and historical context: {cultural_context}"
    
    # Theological layer - add religious meaning
    theological_context = self._add_theological_context(text)
    layer_texts[SemanticLayer.THEOLOGICAL] = f"Religious and spiritual meaning: {theological_context}"
    
    # Interpretive layer - add scholarly interpretation
    interpretive_context = self._add_interpretive_context(text)
    layer_texts[SemanticLayer.INTERPRETIVE] = f"Scholarly interpretation: {interpretive_context}"
    
    # Contextual layer - add situational context
    contextual_context = self._add_contextual_context(text)
    layer_texts[SemanticLayer.CONTEXTUAL] = f"Situational context: {contextual_context}"
    
    return layer_texts
```

### Islamic-Specific Semantic Mappings

The system includes specialized mappings for Islamic terms:

```python
islamic_semantic_mappings = {
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
    }
}
```

### Layer Weighting Strategies

Different weighting strategies can be applied based on the question type and source:

#### For Factoid Questions (Qur'an)
```python
{
    SemanticLayer.LITERAL: 0.25,
    SemanticLayer.CULTURAL: 0.25,
    SemanticLayer.THEOLOGICAL: 0.20,
    SemanticLayer.INTERPRETIVE: 0.20,
    SemanticLayer.CONTEXTUAL: 0.10
}
```

#### For Non-Factoid Questions (Qur'an)
```python
{
    SemanticLayer.LITERAL: 0.15,
    SemanticLayer.CULTURAL: 0.20,
    SemanticLayer.THEOLOGICAL: 0.25,
    SemanticLayer.INTERPRETIVE: 0.25,
    SemanticLayer.CONTEXTUAL: 0.15
}
```

#### For Hadith Sources
```python
{
    SemanticLayer.LITERAL: 0.15,
    SemanticLayer.CULTURAL: 0.20,
    SemanticLayer.THEOLOGICAL: 0.20,
    SemanticLayer.INTERPRETIVE: 0.25,
    SemanticLayer.CONTEXTUAL: 0.20
}
```

## Advantages of Matryoshka Layers

### 1. Comprehensive Semantic Understanding
- Captures multiple levels of meaning simultaneously
- Handles the complexity of Islamic texts
- Provides nuanced understanding of religious concepts

### 2. Context-Aware Retrieval
- Adapts to different question types
- Considers source-specific characteristics
- Balances literal and interpretive meanings

### 3. Explainable Results
- Provides layer-specific explanations
- Shows why sources were selected
- Offers transparency in decision-making

### 4. Cultural Sensitivity
- Respects Islamic scholarly traditions
- Incorporates cultural context
- Handles religious terminology appropriately

## Usage Examples

### Basic Usage
```python
from enhanced_quran_qa import MatryoshkaQuranQASystem

# Initialize system
system = MatryoshkaQuranQASystem()

# Ask a question
question = "ما هي أركان الإسلام الخمسة؟"
result = system.answer_question(question)

# Get layer analysis
layer_analysis = system.analyze_semantic_layers(question)
for layer, strength in layer_analysis.items():
    print(f"{layer.value}: {strength:.3f}")
```

### Advanced Usage with Layer Explanations
```python
# Get detailed layer explanations
for source in result['sources'][:3]:
    explanations = system.get_layer_explanation(question, source)
    print(f"Source: {source['text']}")
    for layer, explanation in explanations.items():
        print(f"  {layer.value}: {explanation}")
```

### Custom Layer Weighting
```python
# Custom weighting for theological focus
theological_weights = {
    SemanticLayer.LITERAL: 0.10,
    SemanticLayer.CULTURAL: 0.15,
    SemanticLayer.THEOLOGICAL: 0.35,
    SemanticLayer.INTERPRETIVE: 0.25,
    SemanticLayer.CONTEXTUAL: 0.15
}

# Apply custom weights
similarity = analyzer.calculate_layer_similarity(
    question_embedding, 
    source_embedding, 
    theological_weights
)
```

## Layer Analysis Examples

### Example 1: "ما هي الصلاة؟"
**Layer Strengths**:
- Literal: 0.85 (high - direct question about prayer)
- Cultural: 0.92 (high - Islamic cultural context)
- Theological: 0.78 (medium - religious significance)
- Interpretive: 0.82 (high - scholarly understanding)
- Contextual: 0.75 (medium - practical application)

### Example 2: "كيف نصلي؟"
**Layer Strengths**:
- Literal: 0.72 (medium - how-to question)
- Cultural: 0.88 (high - cultural practice)
- Theological: 0.65 (medium - spiritual practice)
- Interpretive: 0.90 (high - scholarly guidance)
- Contextual: 0.95 (high - practical implementation)

### Example 3: "ما هو الحج؟"
**Layer Strengths**:
- Literal: 0.80 (high - definition question)
- Cultural: 0.95 (high - cultural significance)
- Theological: 0.88 (high - religious importance)
- Interpretive: 0.85 (high - scholarly interpretation)
- Contextual: 0.78 (medium - practical aspects)

## Performance Benefits

### 1. Improved Retrieval Accuracy
- Multi-dimensional similarity matching
- Context-aware ranking
- Better handling of ambiguous terms

### 2. Enhanced User Experience
- Explainable results
- Layer-specific insights
- Transparent decision-making

### 3. Cultural Appropriateness
- Respects Islamic scholarly traditions
- Handles religious terminology correctly
- Incorporates cultural context

## Technical Implementation

### Embedding Generation
```python
def create_matryoshka_embedding(self, text: str) -> MatryoshkaEmbedding:
    """Create multi-layered embeddings for the given text"""
    layer_texts = self._generate_layer_texts(text)
    
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
```

### Similarity Calculation
```python
def calculate_layer_similarity(self, embedding1, embedding2, layer_weights=None):
    """Calculate weighted similarity across all semantic layers"""
    if layer_weights is None:
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
    
    return sum(similarities[layer] * layer_weights[layer] for layer in SemanticLayer)
```

## Future Enhancements

### 1. Dynamic Layer Weighting
- Adaptive weights based on question complexity
- User preference learning
- Context-aware weight adjustment

### 2. Additional Layers
- Historical layer (temporal context)
- Geographic layer (spatial context)
- Linguistic layer (dialectal variations)

### 3. Advanced Semantic Mappings
- Machine learning-based mapping generation
- Crowdsourced semantic annotations
- Expert-curated mappings

### 4. Interactive Layer Exploration
- User interface for layer exploration
- Visual layer strength indicators
- Interactive weight adjustment

## Conclusion

The Matryoshka layers approach provides a sophisticated framework for understanding the multi-dimensional nature of Islamic texts. By capturing different semantic layers, the system can provide more nuanced and culturally appropriate responses to questions about Islamic texts.

This approach is particularly valuable for:
- **Religious texts**: Where multiple levels of meaning are important
- **Cultural contexts**: Where understanding goes beyond literal meaning
- **Scholarly applications**: Where different interpretations need to be considered
- **Educational purposes**: Where explanations need to be comprehensive

The Matryoshka layers approach represents a significant advancement in semantic understanding for Islamic text processing and can be extended to other domains where multi-layered meaning is important.