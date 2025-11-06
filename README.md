# AI Content Guardrails System

A comprehensive content filtering and safety system for AI applications that prevents harmful, inappropriate, or sensitive content from being processed or generated.

## Overview

This system implements multiple layers of content filtering using machine learning models and rule-based approaches to ensure safe AI interactions. It monitors both user inputs and AI-generated responses, logging all violations to a SQLite database for audit and compliance purposes.

## Features

### Content Filtering Categories

#### 1. **Content Filters** (ML-based)
- **Hate Speech Detection**: Uses Facebook's RoBERTa model to identify hate speech
- **Insult Detection**: Detects toxic and insulting language
- **NSFW Content**: Filters sexual and inappropriate content
- **Violence Detection**: Identifies violent or harmful content
- **Misconduct Detection**: Catches general misconduct and toxic behavior

#### 2. **Denied Topics** (Keyword-based)
- **Politics**: Blocks political discussions and election-related content
- **Religion**: Prevents religious discussions and debates
- **Legal Advice**: Blocks requests for legal counsel or advice

#### 3. **Word Filters** (Pattern-based)
- **Profanity Filter**: Comprehensive list of inappropriate words and phrases
- **Custom Blacklist**: Extensible word filtering system

#### 4. **Sensitive Information** (Regex-based)
- **PII Detection**: Identifies and blocks personally identifiable information
  - Email addresses
  - Social Security Numbers (SSN)
  - Phone numbers
  - Credit card numbers
  - Passport numbers

## System Architecture

### Core Components

#### `GuardrailDB` Class
- Manages SQLite database operations
- Stores violation logs with detailed metadata
- Tracks user behavior and compliance metrics

#### `ContentAnalyzer` Class
- Lazy-loads ML models for performance optimization
- Manages multiple Hugging Face transformer models
- Provides classification services for content analysis

#### `Guardrails` Class
- Main orchestrator for all filtering operations
- Implements individual check methods for each filter type
- Coordinates between analyzers and database logging

## ML Models Used

| Filter Type | Model | Source |
|-------------|-------|--------|
| Hate Speech | `facebook/roberta-hate-speech-dynabench-r4-target` | Hugging Face |
| Toxicity | `unitary/toxic-bert` | Hugging Face |
| Insults | `martin-ha/toxic-comment-model` | Hugging Face |
| NSFW Content | `michellejieli/NSFW_text_classifier` | Hugging Face |
| Misconduct | `unitary/unbiased-toxic-roberta` | Hugging Face |

## Installation

### Prerequisites
```bash
pip install transformers torch sqlite3 openai google-generativeai
```

### Dependencies
- `transformers`: Hugging Face transformers library
- `torch`: PyTorch for ML model inference
- `sqlite3`: Database operations
- `openai`: OpenAI API integration
- `google-generativeai`: Google Gemini API integration
- `re`: Regular expressions for pattern matching
- `logging`: System logging
- `uuid`: Unique identifier generation
- `datetime`: Timestamp management

## Configuration

### API Setup
```python
# Replace with your actual API key
your_api_key = 'YOUR_GEMINI_API_KEY'
genai.configure(api_key=your_api_key)
```

### Database Setup
The system automatically creates a SQLite database at:
```
backend/app/databases/omega.db
```

Database schema includes:
- `guardrail_id`: Unique violation identifier
- `user_id`: User identifier
- `tenant_id`: Tenant/organization identifier
- `category`: Type of violation
- `purpose`: Reason for the filter
- `filtering_type`: Method used for detection
- `strength_of_filter`: Severity level
- `created_at`: Timestamp
- `example`: Sample of flagged content
- `response_message`: System response

## Usage

### Basic Implementation
```python
from guardrails_system import process_input, generate_response

# Check user input
user_text = "Your input text here"
is_safe, message = process_input(user_text, "user123", "tenant456")

if is_safe:
    # Generate AI response with guardrails
    response, error = generate_response(user_text, "user123", "tenant456")
    print(response)
else:
    print(message)  # Violation message
```

### Running the System
```bash
python "guardrails.py"
```

## Filter Thresholds

| Filter Type | Threshold | Strength |
|-------------|-----------|----------|
| Hate Speech | 0.7 | High |
| Insults | 0.7 | High |
| NSFW Content | 0.7 | High |
| Violence | 0.8 | High |
| Misconduct | 0.7 | High |
| Politics | Keyword Match | Medium |
| Religion | Keyword Match | Medium |
| Legal Advice | Keyword Match | Medium |
| Profanity | Exact Match | High |
| PII | Regex Match | High |

## Security Features

- **Input Sanitization**: All user inputs are validated before processing
- **Response Filtering**: AI-generated responses are checked before delivery
- **Audit Logging**: All violations are logged with full context
- **Multi-tenant Support**: Isolated filtering per organization
- **Performance Optimization**: Lazy model loading to reduce memory usage

## Error Handling

The system includes comprehensive error handling:
- Database connection failures
- ML model loading errors
- API communication issues
- Invalid input handling
- Graceful degradation on component failures

## Monitoring and Compliance

### Violation Tracking
- Real-time violation detection
- Historical violation analysis
- User behavior patterns
- Compliance reporting

### Database Queries
```sql
-- View recent violations
SELECT * FROM guardrails ORDER BY created_at DESC LIMIT 10;

-- Count violations by category
SELECT category, COUNT(*) FROM guardrails GROUP BY category;

-- User violation history
SELECT * FROM guardrails WHERE user_id = 'specific_user';
```

## Customization

### Adding New Filters
1. Create a new check method in the `Guardrails` class
2. Add the filter to the `self.filters` dictionary
3. Implement the detection logic
4. Update the database schema if needed

### Modifying Thresholds
Adjust confidence thresholds in individual check methods:
```python
if result['score'] > 0.8:  # Modify threshold here
```

### Extending Keyword Lists
Add new keywords to existing lists:
```python
politics_keywords = ['politics', 'election', 'your_new_keyword']
```

## Performance Considerations

- **Model Caching**: ML models are loaded once and reused
- **Database Pooling**: Connection reuse for better performance
- **Async Processing**: Consider implementing async operations for high-volume usage
- **Memory Management**: Models are loaded on-demand to optimize memory usage

## Future Enhancements

- Real-time dashboard for monitoring
- Custom model training capabilities
- Advanced PII detection
- Multi-language support
- API rate limiting
- Webhook notifications for violations
- Machine learning model updates
- Performance analytics

