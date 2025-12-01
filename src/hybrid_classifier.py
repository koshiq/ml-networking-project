#!/usr/bin/env python3
"""
Hybrid Domain Classifier
Combines fast trie lookup with content-based ML classification.

Architecture:
1. Check trie for known domains (fast path - O(1))
2. If unknown, fetch content and classify with ML (slow path)
3. Cache result in trie for future lookups (learning system)

This is where ML is truly justified!
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
import time
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from feature_extraction import DomainFeatureExtractor
from trie_structure import RuleBasedTrie
from content_feature_extractor import ContentFeatureExtractor
from content_fetcher import ContentFetcher
from rule_based_lookup import RuleBasedLookup


class HybridDomainClassifier:
    """
    Three-tier hybrid classifier:
    - Tier 1 (Fastest): Rule-based lookup from labeled training data (198K domains)
    - Tier 2 (Fast): Trie-based cache for ML predictions
    - Tier 3 (Slow): Content-based ML for truly unknown domains

    This justifies ML usage by analyzing complex content patterns!
    """

    def __init__(self,
                 enable_content_fetch: bool = True,
                 fetch_timeout: int = 10,
                 cache_new_results: bool = True,
                 enable_rules: bool = True):
        """
        Initialize hybrid classifier.

        Args:
            enable_content_fetch: Enable content fetching for unknown domains
            fetch_timeout: Timeout for HTTP requests (seconds)
            cache_new_results: Cache new classifications in trie
            enable_rules: Enable rule-based lookup (Tier 1)
        """
        # Tier 1: Rule-based lookup (ground truth from training data)
        self.rule_lookup = RuleBasedLookup() if enable_rules else None

        # Tier 2: Fast domain-based trie cache
        self.domain_extractor = DomainFeatureExtractor()
        self.trie = RuleBasedTrie()

        # Tier 3: Content-based ML
        self.content_extractor = ContentFeatureExtractor()
        self.content_fetcher = ContentFetcher(timeout=fetch_timeout) if enable_content_fetch else None
        self.content_model = None  # ML model for content classification

        # Configuration
        self.enable_content_fetch = enable_content_fetch
        self.cache_new_results = cache_new_results
        self.enable_rules = enable_rules

        # Statistics
        self.stats = {
            'rule_hits': 0,
            'trie_hits': 0,
            'trie_misses': 0,
            'content_fetches': 0,
            'content_fetch_failures': 0,
            'cache_updates': 0
        }

    def train_domain_trie(self, df: pd.DataFrame, domain_col: str = 'domain',
                         label_col: str = 'label'):
        """
        Train the fast trie-based lookup (Tier 1).

        Args:
            df: DataFrame with domains and labels
            domain_col: Column name for domains
            label_col: Column name for labels
        """
        print("=" * 70)
        print("TIER 1: Training Trie-based Domain Lookup (Fast Path)")
        print("=" * 70)

        signatures = []
        for domain in df[domain_col]:
            sig = self.domain_extractor.extract_hierarchical_signature(domain)
            signatures.append(sig)

        # Create signature dataframe
        sig_df = pd.DataFrame(signatures, columns=['tld', 'sld_pattern', 'subdomain_pattern'])
        df_with_sig = pd.concat([df[[domain_col, label_col]].reset_index(drop=True),
                                 sig_df.reset_index(drop=True)], axis=1)

        # Aggregate by signature
        sig_stats = df_with_sig.groupby(['tld', 'sld_pattern', 'subdomain_pattern']).agg({
            label_col: ['count', 'mean']
        }).reset_index()
        sig_stats.columns = ['tld', 'sld_pattern', 'subdomain_pattern', 'count', 'malicious_ratio']

        # Insert into trie
        for _, row in sig_stats.iterrows():
            if row['count'] >= 5:  # Minimum support
                signature = (row['tld'], row['sld_pattern'], row['subdomain_pattern'])
                prediction = 1 if row['malicious_ratio'] >= 0.5 else 0
                confidence = row['malicious_ratio'] if prediction == 1 else (1 - row['malicious_ratio'])

                self.trie.insert_with_fallback(
                    signature,
                    prediction,
                    confidence,
                    metadata={'count': int(row['count']), 'source': 'training'}
                )

        print(f"✓ Built trie with {self.trie.total_entries} entries")

    def train_content_model(self, content_df: pd.DataFrame,
                           content_col: str = 'html_content',
                           label_col: str = 'label'):
        """
        Train content-based ML classifier (Tier 2).

        This is where ML is TRULY JUSTIFIED - analyzing complex HTML patterns!

        Args:
            content_df: DataFrame with HTML content and labels
            content_col: Column name for HTML content
            label_col: Column name for labels
        """
        print("\n" + "=" * 70)
        print("TIER 2: Training Content-based ML Classifier (Slow Path)")
        print("=" * 70)
        print("THIS IS WHERE ML IS TRULY JUSTIFIED!")
        print("Analyzing complex HTML/JavaScript patterns that rules cannot capture")
        print("=" * 70)

        # Extract content features
        print(f"\n[1/3] Extracting content features from {len(content_df)} pages...")
        features_list = []

        for idx, (_, row) in enumerate(content_df.iterrows()):
            if idx % 100 == 0:
                print(f"  Progress: {idx}/{len(content_df)}", end='\r')

            html = row[content_col]
            url = row.get('url', row.get('domain', 'http://example.com'))
            features = self.content_extractor.extract_features(html, url)
            features_list.append(features)

        print(f"  Progress: {len(content_df)}/{len(content_df)} ✓")

        # Create feature DataFrame
        X = pd.DataFrame(features_list)
        y = content_df[label_col]

        print(f"\n[2/3] Training Random Forest on {len(X.columns)} content features...")
        print(f"  Features include:")
        print(f"    - Ad network detection ({len(self.content_extractor.AD_NETWORKS)} networks)")
        print(f"    - JavaScript analysis (popups, redirects, obfuscation)")
        print(f"    - Tracking pixel detection")
        print(f"    - Content structure analysis")
        print(f"    - Third-party domain analysis")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train Random Forest
        self.content_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )

        start_time = time.time()
        self.content_model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Evaluate
        print(f"\n[3/3] Evaluating content-based classifier...")
        y_pred = self.content_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print("\n" + "=" * 70)
        print("CONTENT-BASED ML CLASSIFIER PERFORMANCE")
        print("=" * 70)
        print(f"  Training time: {train_time:.2f}s")
        print(f"  Accuracy:      {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision:     {precision:.4f} ({precision*100:.2f}%)")
        print(f"  Recall:        {recall:.4f} ({recall*100:.2f}%)")
        print(f"  F1-Score:      {f1:.4f} ({f1*100:.2f}%)")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.content_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n  Top 10 Most Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"    {row['feature']:<35} {row['importance']:.4f}")

        print("=" * 70)

    def predict(self, domain: str, fetch_if_unknown: bool = None) -> Tuple[int, float, str]:
        """
        Predict if domain is malicious or legitimate.

        Three-tier approach:
        1. Check rule-based lookup (198K labeled domains - ground truth)
        2. Check trie cache (previous ML predictions)
        3. If unknown, fetch content and classify with ML (slow but accurate)

        Args:
            domain: Domain to classify
            fetch_if_unknown: Override default content fetch behavior

        Returns:
            (prediction, confidence, method)
            - prediction: 0 (legitimate), 1 (malicious)
            - confidence: Confidence score [0, 1]
            - method: 'rules', 'trie', or 'content_ml'
        """
        if fetch_if_unknown is None:
            fetch_if_unknown = self.enable_content_fetch

        # TIER 1: Rule-based lookup (fastest, most accurate)
        if self.enable_rules and self.rule_lookup and self.rule_lookup.loaded:
            result = self.rule_lookup.lookup(domain)
            if result is not None:
                self.stats['rule_hits'] += 1
                return result  # (prediction, confidence, 'rules')

        # TIER 2: Trie cache lookup
        signature = self.domain_extractor.extract_hierarchical_signature(domain)
        pred, conf, match_level = self.trie.lookup(signature)

        if match_level > 0:
            # Found in trie - fast path!
            self.stats['trie_hits'] += 1
            return pred, conf, 'trie'

        # TIER 3: Content-based ML (unknown domain)
        self.stats['trie_misses'] += 1

        if not fetch_if_unknown or not self.content_fetcher or not self.content_model:
            # Cannot fetch or classify - return default
            return 0, 0.5, 'default'

        # Fetch content
        print(f"  [Unknown domain] Fetching content from {domain}...")
        self.stats['content_fetches'] += 1

        html = self.content_fetcher.fetch_domain(domain)

        if not html:
            self.stats['content_fetch_failures'] += 1
            return 0, 0.5, 'fetch_failed'

        # Extract content features
        features = self.content_extractor.extract_features(html, f'http://{domain}')
        X = pd.DataFrame([features])

        # Predict with content model
        prediction = self.content_model.predict(X)[0]
        confidence = self.content_model.predict_proba(X)[0][prediction]

        # Cache result in trie for future lookups
        if self.cache_new_results:
            self.trie.insert_with_fallback(
                signature,
                prediction,
                confidence,
                metadata={'source': 'content_ml', 'cached_at': time.time()}
            )
            self.stats['cache_updates'] += 1

        return prediction, confidence, 'content_ml'

    def explain_prediction(self, domain: str) -> Dict[str, Any]:
        """Explain how a domain was classified"""
        pred, conf, method = self.predict(domain)

        explanation = {
            'domain': domain,
            'prediction': 'malicious' if pred == 1 else 'legitimate',
            'confidence': conf,
            'method': method,
            'signature': self.domain_extractor.extract_hierarchical_signature(domain)
        }

        if method == 'trie':
            explanation['reasoning'] = "Found in cached trie (fast lookup)"
        elif method == 'content_ml':
            explanation['reasoning'] = "Analyzed content with ML (fetched and classified)"
        else:
            explanation['reasoning'] = "Unknown domain, could not fetch content"

        return explanation

    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics"""
        rule_hits = self.stats.get('rule_hits', 0)
        total_requests = rule_hits + self.stats['trie_hits'] + self.stats['trie_misses']

        if total_requests > 0:
            rule_hit_rate = rule_hits / total_requests
            cache_hit_rate = (rule_hits + self.stats['trie_hits']) / total_requests
        else:
            rule_hit_rate = 0.0
            cache_hit_rate = 0.0

        return {
            **self.stats,
            'total_requests': total_requests,
            'rule_hit_rate': rule_hit_rate,
            'cache_hit_rate': cache_hit_rate,
            'trie_entries': self.trie.total_entries,
            'rule_entries': len(self.rule_lookup.domain_labels) if self.rule_lookup and self.rule_lookup.loaded else 0
        }

    def save(self, model_dir: str):
        """Save hybrid model"""
        import os
        import json
        os.makedirs(model_dir, exist_ok=True)

        # Save trie
        self.trie.save(f"{model_dir}/trie.json")

        # Save content model
        if self.content_model:
            with open(f"{model_dir}/content_model.pkl", 'wb') as f:
                pickle.dump(self.content_model, f)

        # Save statistics
        with open(f"{model_dir}/stats.json", 'w') as f:
            json.dump(self.stats, f, indent=2)

        print(f"✓ Hybrid model saved to {model_dir}/")

    def load(self, model_dir: str):
        """Load hybrid model"""
        import json

        # Load rule-based lookup
        if self.enable_rules and self.rule_lookup:
            try:
                self.rule_lookup.load('data/labeled_domains.csv')
            except Exception as e:
                print(f"⚠ Could not load rule-based lookup: {e}")
                self.enable_rules = False

        # Load trie
        self.trie.load(f"{model_dir}/trie.json")

        # Load content model
        try:
            with open(f"{model_dir}/content_model.pkl", 'rb') as f:
                self.content_model = pickle.load(f)
        except FileNotFoundError:
            print("⚠ Content model not found, only trie loaded")

        # Load statistics
        try:
            with open(f"{model_dir}/stats.json", 'r') as f:
                loaded_stats = json.load(f)
                # Merge with default stats to ensure all keys exist
                self.stats.update(loaded_stats)
        except FileNotFoundError:
            pass

        # Ensure rule_hits exists in stats
        if 'rule_hits' not in self.stats:
            self.stats['rule_hits'] = 0

        print(f"✓ Hybrid model loaded from {model_dir}/")


if __name__ == '__main__':
    print("Hybrid Domain Classifier - Loaded")
    print("Combines trie lookup + content-based ML")
    print("This is where ML is truly justified!")