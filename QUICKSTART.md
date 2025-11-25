# Quick Start Guide

## Installation

1. **Install dependencies**:
   ```bash
   cd ml-hierarchical-domain-classifier
   pip install -r requirements.txt
   ```

2. **Verify installation**:
   ```bash
   python tests/test_basic.py
   ```
   Should show: `Ran 12 tests in X.XXXs - OK`

## Usage

### 1. Run Quick Demo

Get a feel for how the classifier works:

```bash
python demo.py
```

This will:
- Train on 5,000 sample domains
- Show predictions on test domains
- Display performance metrics
- Save a demo model

### 2. Train Full Classifier

Train on the complete dataset:

```bash
python train_classifier.py
```

Options:
```bash
# Quick test with 10,000 samples
python train_classifier.py --sample 10000

# Custom parameters
python train_classifier.py \
  --data ../Data/dns_training_data.csv \
  --output models/my_model \
  --min-support 20 \
  --min-precision 0.7
```

### 3. Use Trained Model

```python
import sys
sys.path.insert(0, 'src')

from hierarchical_classifier import HierarchicalDomainClassifier

# Load trained model
classifier = HierarchicalDomainClassifier()
classifier.load('models/hierarchical_classifier')

# Predict single domain
prediction, confidence, level = classifier.predict('suspicious-domain.xyz')
print(f"Prediction: {prediction}, Confidence: {confidence:.2%}")

# Batch prediction
domains = ['google.com', 'malware.xyz', 'facebook.com']
results = classifier.predict_batch(domains)
print(results)

# Get detailed explanation
explanation = classifier.explain_prediction('ads.tracker.net')
print(explanation)
```

### 4. Explore Analysis Notebook

```bash
cd notebooks
jupyter notebook 01_pattern_analysis.ipynb
```

This notebook shows:
- TLD distribution analysis
- Domain pattern extraction
- Rule generation process
- Feature importance

## Project Structure

```
ml-hierarchical-domain-classifier/
├── src/
│   ├── feature_extraction.py      # Extract domain features
│   ├── rule_generator.py          # ML-based rule generation
│   ├── trie_structure.py          # Hierarchical trie
│   └── hierarchical_classifier.py # Main classifier
├── notebooks/
│   └── 01_pattern_analysis.ipynb  # Data analysis
├── tests/
│   └── test_basic.py              # Unit tests
├── models/                         # Saved models
├── data/                           # Processed data
├── train_classifier.py            # Training script
├── demo.py                        # Demo script
└── README.md                      # Documentation
```

## Key Features

### Hierarchical Classification

The classifier uses a 3-level hierarchy:

1. **TLD Level**: `.com`, `.xyz`, `.top`, etc.
2. **Domain Pattern**: `high_digits`, `very_long`, `normal`, etc.
3. **Subdomain Pattern**: `none`, `www_only`, `deep`, etc.

Example:
```
'malicious123.xyz' → ('xyz', 'high_digits', 'none') → Malicious (95%)
'www.google.com'   → ('com', 'normal', 'www_only') → Legitimate (90%)
```

### Performance

- **Lookup Speed**: ~50,000-100,000 domains/second
- **Time Complexity**: O(log n) or better
- **Space Complexity**: O(n) for n unique signatures

### Explainability

Every prediction includes:
- Prediction (0=legitimate, 1=malicious)
- Confidence score
- Match level (how specific the match was)
- Hierarchical signature
- Key features
- Human-readable reasoning

## Examples

### Example 1: Simple Prediction

```python
from hierarchical_classifier import HierarchicalDomainClassifier

classifier = HierarchicalDomainClassifier()
classifier.load('models/demo_model')

# Predict
pred, conf, level = classifier.predict('ads.doubleclick.net')
print(f"Malicious: {pred == 1}, Confidence: {conf:.1%}")
```

### Example 2: Batch Processing

```python
import pandas as pd

# Load domains from file
domains = pd.read_csv('domains.txt', header=None, names=['domain'])

# Batch predict
results = classifier.predict_batch(domains['domain'].tolist())

# Save results
results.to_csv('predictions.csv', index=False)
```

### Example 3: Integration with DNS Filtering

```python
def should_block_domain(domain):
    """Check if domain should be blocked"""
    pred, conf, level = classifier.predict(domain)

    # Block if classified as malicious with high confidence
    if pred == 1 and conf > 0.7:
        return True

    # Allow otherwise
    return False

# Use in DNS filter
domain = get_dns_query()
if should_block_domain(domain):
    return "BLOCKED"
else:
    return resolve_dns(domain)
```

### Example 4: Continuous Learning

```python
# Start with trained model
classifier.load('models/current_model')

# Collect new labeled data
new_domains = pd.read_csv('new_labeled_data.csv')

# Retrain with combined data
all_data = pd.concat([old_data, new_domains])
classifier.train(all_data)

# Save updated model
classifier.save('models/updated_model')
```

## Performance Tuning

### Increase Precision (reduce false positives)

```python
classifier = HierarchicalDomainClassifier(
    min_support=20,      # Increase support requirement
    min_precision=0.8    # Increase precision threshold
)
```

### Increase Recall (catch more threats)

```python
classifier = HierarchicalDomainClassifier(
    min_support=5,       # Decrease support requirement
    min_precision=0.5    # Decrease precision threshold
)
```

### Balance Speed vs Accuracy

```python
# Faster but less specific
pred, conf, level = classifier.predict(domain)
if level >= 1:  # Accept TLD-level matches
    return pred

# Slower but more accurate
if level >= 3:  # Only accept full signature matches
    return pred
else:
    # Fall back to full ML model
    return ml_model.predict(extract_features(domain))
```

## Troubleshooting

### Import Errors

```bash
# Add src to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Low Accuracy

- Increase training data size
- Adjust `min_support` and `min_precision` parameters
- Check class balance in training data

### Slow Training

- Use `--sample` flag to train on subset
- Reduce decision tree `max_depth`
- Limit number of threshold rules

## Next Steps

1. **Integrate with existing system**: Use classifier in DNS service or proxy
2. **Add content analysis**: Fetch and analyze page content for new domains
3. **Build API**: Wrap classifier in REST API for remote access
4. **Deploy to production**: Containerize with Docker
5. **Monitor performance**: Track accuracy and speed metrics
6. **Retrain regularly**: Update model with new threat data

## Further Reading

- [ARCHITECTURE.md](ARCHITECTURE.md) - Detailed architecture documentation
- [README.md](README.md) - Project overview
- Notebooks - Data analysis and experimentation

## Support

For questions or issues, check:
1. Unit tests: `python tests/test_basic.py`
2. Demo script: `python demo.py`
3. Documentation: Read ARCHITECTURE.md
