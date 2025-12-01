# ML-Optimized Hierarchical Domain Classifier

**A three-tier hybrid system combining rule-based lookups, ML caching, and content-based machine learning for intelligent domain classification.**

---

## 🎯 What We're Doing

We've built a **three-tier domain classification system** that demonstrates **when machine learning actually makes sense versus using simple rules**. The system detects malicious and advertising domains by combining:

1. **Ground truth lookups** (198K labeled domains - instant, 100% accurate)
2. **ML prediction cache** (fast trie lookups for previously classified domains)
3. **Intelligent content analysis** (ML for truly unknown domains)

### The Core Problem

- **Simple rules alone:** Can catch obvious patterns but fail on new/disguised domains
- **ML-only approach:** Accurate but painfully slow (2-5 seconds per domain)
- **Our solution:** Use rules for known domains, cache ML predictions, only fetch content when necessary

### Why This Justifies Machine Learning

**Domain features alone** (length, TLD, characters) can be handled with simple rules - doesn't need ML.

**Content analysis** requires ML to detect complex patterns:
- Ad network combinations (is 5 tracking scripts suspicious? 20?)
- JavaScript behavior (legitimate sites use popups too - what's malicious?)
- Content structure (ads vs commerce sites both have product listings)
- Obfuscation techniques (encoded scripts, hidden iframes)

**These patterns are too complex for simple rules → ML is truly justified here!**

---

## 🏗️ Architecture

### Three-Tier System

```
┌─────────────────────────────────────────────────────────────┐
│                    INCOMING DOMAIN                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────▼─────────────┐
         │   TIER 1: RULE-BASED      │
         │   198K labeled domains    │
         │   (Ground Truth)          │
         └─────────────┬─────────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
      [FOUND]                   [NOT FOUND]
         │                           │
         ▼                           ▼
    ┌─────────┐         ┌────────────────────┐
    │ RETURN  │         │  TIER 2: TRIE      │
    │ 100%    │         │  Cached ML Results │
    └─────────┘         └─────────┬──────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                 [FOUND]                   [NOT FOUND]
                    │                           │
                    ▼                           ▼
               ┌─────────┐         ┌─────────────────────┐
               │ RETURN  │         │  TIER 3: CONTENT ML │
               │ CACHED  │         │  Fetch & Analyze    │
               └─────────┘         └──────────┬──────────┘
                                              │
                                              ▼
                                   ┌──────────────────────┐
                                   │ CACHE IN TRIE        │
                                   │ RETURN RESULT        │
                                   └──────────────────────┘
```

### Tier 1: Rule-Based Lookup (198K Domains)
- **Purpose**: Ground truth for known domains
- **Time**: ~0.004 ms (dictionary lookup)
- **Accuracy**: 100% (labeled training data)
- **Coverage**: 198,377 domains from `labeled_domains.csv`

### Tier 2: Trie Cache (Growing)
- **Purpose**: Fast lookups for previously classified domains
- **Time**: ~0.2 ms (trie pattern matching)
- **Data**: Cached ML predictions and domain signatures
- **Learning**: Grows as new domains are classified

### Tier 3: Content-Based ML (Unknown Domains)
- **Purpose**: Intelligent classification for truly unknown domains
- **Time**: 2-5 seconds (fetch HTML + analyze)
- **Features**: 30+ content-based features (ad networks, JS, tracking, etc.)
- **Model**: Random Forest classifier

---

## 🔬 Why ML is Justified Here

### Content Analysis Features (30+ features)

The content-based ML analyzes:

#### 1. **Ad Network Detection**
- Presence of 20+ known ad networks (doubleclick, adsense, etc.)
- Ad network domain counts
- Tracking pixel detection

#### 2. **JavaScript Analysis**
- Popup/popunder code detection
- Redirect patterns (`window.location`, `location.href`)
- Code obfuscation (`eval()`, `fromCharCode`)
- Suspicious patterns

#### 3. **Content Structure**
- Script-to-content ratio
- Iframe usage
- External link ratios
- Third-party domain analysis

#### 4. **Tracking & Analytics**
- Google Analytics presence
- Facebook Pixel detection
- 1x1 tracking pixels
- Cookie tracking scripts

#### 5. **Content Quality**
- Text-to-HTML ratio
- Meaningful content detection
- Ad keyword density
- Overlay/modal detection

**These patterns are too complex for simple rules → ML is necessary!**

---

## 📁 Project Structure

```
ml-hierarchical-domain-classifier/
├── src/
│   ├── content_feature_extractor.py    # HTML/JS feature extraction
│   ├── content_fetcher.py              # HTTP content fetching
│   └── hybrid_classifier.py            # Two-tier classifier
│
├── data/
│   ├── labeled_domains.csv             # 198K labeled domains
│   ├── parsed_domains.csv              # All parsed domains
│   ├── content_training_cache.csv      # Cached HTML content
│   └── evaluation_results.csv          # Performance metrics
│
├── models/
│   └── hybrid_classifier/
│       ├── trie.json                   # Trained trie (1074 entries)
│       ├── content_model.pkl           # Random Forest model
│       └── stats.json                  # Model statistics
│
├── run_classifier.py                   # Unified tool (demo/monitor/benchmark)
├── train_hybrid_classifier.py          # Training script
├── proxy_server.py                     # HTTP proxy server
└── parse_domains.py                    # Domain parser
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Use the Classifier

The model is already trained. Use the unified tool:

```bash
# Quick demo
python run_classifier.py demo

# Interactive monitoring with performance metrics
python run_classifier.py monitor

# Performance benchmark
python run_classifier.py benchmark
python run_classifier.py benchmark -n 5000  # 5000 iterations
```

### 3. Monitor Mode Example

```bash
python run_classifier.py monitor
```

**Output:**
```
Three-Tier Classification:
  1. Rule-based lookup (198K labeled domains)
  2. Trie cache (previous ML predictions)
  3. Content ML (for unknowns)

> google.com

  🟢 LEGITIMATE
  Confidence: 100%
  Method: Tier 1: Rule-based
  ⚡ Time: 0.0042 ms

  📊 Session: 1 reqs | Avg: 0.00 ms
     Rules: 1 | Trie: 0 | ML: 0
```

### 4. HTTP Proxy Server (Optional)

```bash
python proxy_server.py --port 8080
```

Intercepts browser traffic and classifies domains in real-time.

---

## 📊 Performance Metrics

### Tier 1: Rule-Based (198K Domains)
- **Speed**: ~0.004 ms
- **Accuracy**: 100% (ground truth)
- **Method**: Dictionary lookup
- **Coverage**: 198,377 labeled domains

### Tier 2: Trie Cache
- **Speed**: ~0.2 ms
- **Accuracy**: Varies (cached ML predictions)
- **Method**: Trie pattern matching
- **Coverage**: Growing (caches new ML results)

### Tier 3: Content ML
- **Speed**: 2,000-5,000 ms (2-5 seconds)
- **Features**: 30+ content-based features
- **Model**: Random Forest (100 trees)
- **Coverage**: All unknown domains

### Overall System
- **Expected hit rate**: 99%+ Tier 1 + Tier 2 (real-world usage)
- **Average response**: < 1 ms for known domains
- **Learning**: System improves over time

---

## 🔧 Usage Examples

### Basic Classification

```python
from src.hybrid_classifier import HybridDomainClassifier

# Load trained model
classifier = HybridDomainClassifier()
classifier.load('models/hybrid_classifier')

# Classify a domain
prediction, confidence, method = classifier.predict('example.com')

print(f"Prediction: {prediction}")  # 0=legitimate, 1=malicious
print(f"Confidence: {confidence:.2%}")
print(f"Method: {method}")  # 'trie' or 'content_ml'
```

### Statistics

```python
stats = classifier.get_statistics()

print(f"Rule hits: {stats['rule_hits']}")
print(f"Trie hits: {stats['trie_hits']}")
print(f"Content fetches: {stats['content_fetches']}")
print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
```

---

## 🎓 Why This Matters

### Educational Value

This project demonstrates:
- **When ML is appropriate** vs when simple rules suffice
- **Three-tier architecture** balancing speed, accuracy, and intelligence
- **Content-based feature engineering** (30+ features from HTML/JS)
- **Learning systems** that cache and improve over time
- **Real-world trade-offs** between performance and accuracy

### Key Insight

**Domain features alone** can be handled with simple rules - doesn't need ML.

**Content analysis** (ad networks, JavaScript behavior, obfuscation) requires ML to detect complex patterns that rules cannot capture.

This is justified, practical ML - not "ML for ML's sake"!

---

## 🔍 Content Features Explained

### Most Important Features (from training)

1. **text_to_html_ratio** (9.88% importance)
   - Ratio of actual text to HTML markup
   - Ad sites often have more markup than content

2. **ad_keyword_density** (7.84% importance)
   - Density of ad-related keywords per 10K characters
   - Keywords: "advertisement", "sponsored", "tracking", etc.

3. **text_content_length** (7.54% importance)
   - Length of actual text content (excluding tags)
   - Ad sites often have minimal real content

4. **num_divs** (7.24% importance)
   - Number of div elements
   - Ad sites use many divs for ad slots

5. **num_scripts** (5.31% importance)
   - Number of script tags
   - Ad sites load many tracking/ad scripts

---

## 📚 Dataset

- **198,377 labeled domains** from network-traffic-project
- **Malicious**: 99,081 (ad/tracking domains)
- **Legitimate**: 99,296 (legitimate sites)
- **Features**: 18 domain features + 33 content features

---

## ✨ Summary

**Three-tier hybrid system** that demonstrates intelligent use of ML:

1. **Tier 1 (Rules):** 198K labeled domains → Instant, 100% accurate
2. **Tier 2 (Cache):** ML predictions cached → Fast, learning system
3. **Tier 3 (ML):** Content analysis → Slow but handles unknowns

**Key insight:** Use simple rules where possible, ML where necessary. The system proves ML is justified for content analysis (complex patterns) but overkill for simple domain lookups (use ground truth instead).

**Real-world applicable:** Can be deployed as DNS filter, HTTP proxy, or browser extension.