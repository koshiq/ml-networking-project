"""
Main hierarchical classifier implementation.
Combines feature extraction, rule generation, and trie-based lookup.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
import time

from feature_extraction import DomainFeatureExtractor
from rule_generator import MLRuleGenerator, Rule
from trie_structure import HierarchicalDomainTrie, RuleBasedTrie


class HierarchicalDomainClassifier:
    """
    ML-optimized hierarchical domain classifier.

    Uses a three-level approach:
    1. TLD-based filtering
    2. Domain pattern matching
    3. Subdomain analysis

    Achieves O(log n) or better through hierarchical trie structure.
    """

    def __init__(self, min_support: int = 10, min_precision: float = 0.6):
        self.feature_extractor = DomainFeatureExtractor()
        self.rule_generator = MLRuleGenerator(min_support, min_precision)
        self.trie = RuleBasedTrie()
        self.is_trained = False
        self.performance_stats = {}

    def train(self, df: pd.DataFrame, domain_col: str = 'domain',
             label_col: str = 'label'):
        """
        Train the classifier on labeled domain data.

        Args:
            df: DataFrame with domains and labels
            domain_col: Column name for domains
            label_col: Column name for labels (0=legitimate, 1=malicious)
        """
        print("=" * 60)
        print("Training Hierarchical Domain Classifier")
        print("=" * 60)

        start_time = time.time()

        # Step 1: Extract features
        print("\n[1/4] Extracting features...")
        feature_list = []
        signatures = []

        for domain in df[domain_col]:
            features = self.feature_extractor.extract_features(domain)
            feature_list.append(features)
            signatures.append(self.feature_extractor.extract_hierarchical_signature(domain))

        df_features = pd.DataFrame(feature_list)
        df_full = pd.concat([df[[domain_col, label_col]].reset_index(drop=True),
                            df_features.reset_index(drop=True)], axis=1)

        print(f"   Extracted {len(df_features.columns)} features from {len(df)} domains")

        # Step 2: Generate rules
        print("\n[2/4] Generating ML-optimized rules...")
        numerical_features = list(df_features.select_dtypes(include=[np.number]).columns)
        rules = self.rule_generator.generate_all_rules(
            df_full,
            feature_columns=numerical_features + ['tld'],
            label_col=label_col
        )

        # Filter and optimize rules
        high_quality_rules = [r for r in rules if r.precision >= 0.7 and r.coverage >= 0.001]
        print(f"   Filtered to {len(high_quality_rules)} high-quality rules (P>=0.7, Cov>=0.001)")

        # Step 3: Build trie structure
        print("\n[3/4] Building hierarchical trie structure...")
        signature_df = pd.DataFrame(signatures, columns=['tld', 'sld_pattern', 'subdomain_pattern'])
        df_with_sig = pd.concat([df_full, signature_df], axis=1)

        # Aggregate signatures with their labels
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
                    metadata={'count': int(row['count']), 'source': 'training_data'}
                )

        print(f"   Built trie with {self.trie.total_entries} entries")
        print(f"   Trie statistics: {self.trie.get_statistics()}")

        # Step 4: Evaluate on training data
        print("\n[4/4] Evaluating classifier...")
        predictions = []
        confidences = []
        match_levels = []

        for domain in df[domain_col]:
            pred, conf, level = self.predict(domain)
            predictions.append(pred if pred is not None else 0)
            confidences.append(conf)
            match_levels.append(level)

        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        y_true = df[label_col]
        y_pred = predictions

        self.performance_stats = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'avg_confidence': np.mean(confidences),
            'match_level_distribution': pd.Series(match_levels).value_counts().to_dict(),
            'training_time': time.time() - start_time
        }

        print("\n" + "=" * 60)
        print("Training Performance Metrics:")
        print("=" * 60)
        print(f"  Accuracy:  {self.performance_stats['accuracy']:.4f}")
        print(f"  Precision: {self.performance_stats['precision']:.4f}")
        print(f"  Recall:    {self.performance_stats['recall']:.4f}")
        print(f"  F1-Score:  {self.performance_stats['f1_score']:.4f}")
        print(f"  Avg Confidence: {self.performance_stats['avg_confidence']:.4f}")
        print(f"\nMatch Level Distribution:")
        for level, count in sorted(self.performance_stats['match_level_distribution'].items()):
            print(f"  Level {level}: {count} ({count/len(df)*100:.1f}%)")
        print(f"\nTraining time: {self.performance_stats['training_time']:.2f}s")
        print("=" * 60)

        self.is_trained = True

    def predict(self, domain: str) -> Tuple[Optional[int], float, int]:
        """
        Predict if domain is malicious or legitimate.

        Returns:
            (prediction, confidence, match_level)
            - prediction: 0 (legitimate), 1 (malicious), None (unknown)
            - confidence: Confidence score [0, 1]
            - match_level: 0 (no match), 1 (TLD), 2 (domain pattern), 3 (full match)
        """
        signature = self.feature_extractor.extract_hierarchical_signature(domain)
        prediction, confidence, match_level = self.trie.lookup(signature)

        return prediction, confidence, match_level

    def predict_batch(self, domains: List[str]) -> pd.DataFrame:
        """
        Predict for multiple domains.

        Returns DataFrame with predictions, confidences, and match levels.
        """
        results = []
        for domain in domains:
            pred, conf, level = self.predict(domain)
            results.append({
                'domain': domain,
                'prediction': pred if pred is not None else 0,
                'confidence': conf,
                'match_level': level,
                'label': 'malicious' if pred == 1 else 'legitimate'
            })

        return pd.DataFrame(results)

    def explain_prediction(self, domain: str) -> Dict[str, Any]:
        """
        Explain why a domain was classified a certain way.
        Returns detailed information about the prediction.
        """
        # Extract features and signature
        features = self.feature_extractor.extract_features(domain)
        signature = self.feature_extractor.extract_hierarchical_signature(domain)
        prediction, confidence, match_level = self.predict(domain)

        explanation = {
            'domain': domain,
            'prediction': 'malicious' if prediction == 1 else 'legitimate',
            'confidence': confidence,
            'match_level': match_level,
            'signature': {
                'tld': signature[0],
                'domain_pattern': signature[1],
                'subdomain_pattern': signature[2]
            },
            'key_features': {
                'domain_length': features['domain_length'],
                'sld_length': features['sld_length'],
                'num_subdomains': features['num_subdomains'],
                'sld_entropy': features['sld_entropy'],
                'digit_ratio': features['sld_digit_ratio']
            },
            'reasoning': self._generate_reasoning(features, signature, prediction, match_level)
        }

        return explanation

    def _generate_reasoning(self, features: Dict, signature: Tuple,
                           prediction: Optional[int], match_level: int) -> str:
        """Generate human-readable reasoning for prediction"""
        if match_level == 0:
            return "No matching pattern found in training data. Using default classification."
        elif match_level == 1:
            return f"Based on TLD pattern: '{signature[0]}' has characteristic risk profile."
        elif match_level == 2:
            return f"Based on TLD ({signature[0]}) and domain pattern ({signature[1]})."
        else:
            return f"Full signature match: TLD={signature[0]}, pattern={signature[1]}, subdomains={signature[2]}."

    def save(self, model_dir: str):
        """Save trained model"""
        import os
        os.makedirs(model_dir, exist_ok=True)

        # Save trie
        self.trie.save(f"{model_dir}/trie.json")

        # Save rules
        self.rule_generator.save_rules(f"{model_dir}/rules.json")

        # Save performance stats
        import json
        with open(f"{model_dir}/performance.json", 'w') as f:
            json.dump(self.performance_stats, f, indent=2)

        print(f"Model saved to {model_dir}/")

    def load(self, model_dir: str):
        """Load trained model"""
        # Load trie
        self.trie.load(f"{model_dir}/trie.json")

        # Load rules
        self.rule_generator.load_rules(f"{model_dir}/rules.json")

        # Load performance stats
        import json
        with open(f"{model_dir}/performance.json", 'r') as f:
            self.performance_stats = json.load(f)

        self.is_trained = True
        print(f"Model loaded from {model_dir}/")

    def benchmark_lookup_time(self, domains: List[str]) -> Dict[str, float]:
        """Benchmark lookup performance"""
        import time

        start = time.time()
        for domain in domains:
            self.predict(domain)
        total_time = time.time() - start

        return {
            'total_time': total_time,
            'avg_time_ms': (total_time / len(domains)) * 1000,
            'lookups_per_second': len(domains) / total_time
        }


if __name__ == '__main__':
    print("Hierarchical Domain Classifier module loaded successfully")
    print("Use this module to train and deploy ML-optimized domain classifiers")
