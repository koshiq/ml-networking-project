#!/usr/bin/env python3
"""
Demo script for Hierarchical Domain Classifier.
Quick demonstration of classifier capabilities.
"""

import sys
import os
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from hierarchical_classifier import HierarchicalDomainClassifier


def demo_quick_training():
    """Demonstrate quick training on sample data"""
    print("=" * 70)
    print("DEMO: Quick Training with Sample Data")
    print("=" * 70)

    # Load sample data
    print("\nLoading sample from DNS training data...")
    df = pd.read_csv('../Data/dns_training_data.csv')
    df_sample = df.sample(n=5000, random_state=42)  # 5K samples for quick demo
    print(f"Using {len(df_sample)} domains for demo")

    # Initialize and train
    classifier = HierarchicalDomainClassifier(min_support=5, min_precision=0.65)
    classifier.train(df_sample, domain_col='domain', label_col='label')

    return classifier


def demo_predictions(classifier):
    """Demonstrate predictions on various domain types"""
    print("\n" + "=" * 70)
    print("DEMO: Predictions on Test Domains")
    print("=" * 70)

    test_domains = [
        # Legitimate domains
        'google.com',
        'github.com',
        'stackoverflow.com',
        'wikipedia.org',

        # Suspicious patterns
        'ads.doubleclick.net',
        'tracker.analytics.com',
        'malware123.xyz',
        'very-long-suspicious-name-with-many-keywords.top',

        # Subdomain-heavy
        'deep.nested.subdomain.tracking.site.com',
        'www.example.com',

        # High-entropy/random
        'xn83jd9s.tk',
        'a1b2c3d4.click',
    ]

    for domain in test_domains:
        explanation = classifier.explain_prediction(domain)
        print(f"\n{'─' * 70}")
        print(f"Domain: {domain}")
        print(f"{'─' * 70}")
        print(f"  Prediction:  {explanation['prediction'].upper()}")
        print(f"  Confidence:  {explanation['confidence']:.1%}")
        print(f"  Match Level: {explanation['match_level']}/3")
        print(f"  Signature:")
        print(f"    └─ TLD: {explanation['signature']['tld']}")
        print(f"    └─ Domain Pattern: {explanation['signature']['domain_pattern']}")
        print(f"    └─ Subdomain Pattern: {explanation['signature']['subdomain_pattern']}")
        print(f"  Key Features:")
        for feature, value in explanation['key_features'].items():
            print(f"    └─ {feature}: {value}")
        print(f"  Reasoning: {explanation['reasoning']}")


def demo_batch_prediction(classifier):
    """Demonstrate batch prediction"""
    print("\n" + "=" * 70)
    print("DEMO: Batch Prediction")
    print("=" * 70)

    domains = [
        'google.com', 'ads.tracker.net', 'facebook.com',
        'malicious.xyz', 'github.com', 'spam123.top'
    ]

    results = classifier.predict_batch(domains)
    print("\nBatch prediction results:")
    print(results.to_string(index=False))


def demo_performance_analysis(classifier):
    """Demonstrate performance analysis"""
    print("\n" + "=" * 70)
    print("DEMO: Performance Analysis")
    print("=" * 70)

    # Load test data
    df_test = pd.read_csv('../Data/dns_training_data.csv').sample(n=1000, random_state=123)

    # Benchmark
    benchmark = classifier.benchmark_lookup_time(df_test['domain'].tolist())

    print("\nLookup Performance:")
    print(f"  Total time: {benchmark['total_time']:.3f}s for {len(df_test)} domains")
    print(f"  Average time per lookup: {benchmark['avg_time_ms']:.4f} ms")
    print(f"  Throughput: {benchmark['lookups_per_second']:.0f} lookups/second")

    # Time complexity analysis
    print(f"\nTime Complexity Analysis:")
    print(f"  Target: O(log n) or better")
    print(f"  Actual: ~O(1) to O(log n) depending on match level")
    print(f"  Average match level: {classifier.performance_stats.get('match_level_distribution', {})}")


def demo_trie_statistics(classifier):
    """Show trie structure statistics"""
    print("\n" + "=" * 70)
    print("DEMO: Trie Structure Statistics")
    print("=" * 70)

    stats = classifier.trie.get_statistics()
    print("\nHierarchical Trie Structure:")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Number of unique TLDs: {stats['num_tlds']}")
    print(f"  Level 1 nodes (TLD): {stats['levels']['tld_nodes']}")
    print(f"  Level 2 nodes (Domain Patterns): {stats['levels']['pattern_nodes']}")
    print(f"  Level 3 nodes (Full Signatures): {stats['levels']['terminal_nodes']}")

    avg_branching = stats['levels']['pattern_nodes'] / stats['levels']['tld_nodes'] if stats['levels']['tld_nodes'] > 0 else 0
    print(f"\n  Average branching factor: {avg_branching:.2f}")
    print(f"  Space complexity: O(n) where n = {stats['total_entries']}")


def main():
    print("\n" + "=" * 70)
    print(" Hierarchical Domain Classifier - Interactive Demo")
    print("=" * 70)
    print("\nThis demo showcases the ML-optimized hierarchical approach")
    print("for efficient malicious domain detection.\n")

    # Train classifier
    classifier = demo_quick_training()

    # Run demos
    demo_predictions(classifier)
    demo_batch_prediction(classifier)
    demo_performance_analysis(classifier)
    demo_trie_statistics(classifier)

    # Save model
    print("\n" + "=" * 70)
    print("Saving Demo Model")
    print("=" * 70)
    classifier.save('models/demo_model')

    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Hierarchical structure enables O(log n) lookups")
    print("  2. ML-optimized rule generation improves accuracy")
    print("  3. Trie-based storage is space-efficient")
    print("  4. High throughput suitable for real-time filtering")
    print("\nNext steps:")
    print("  - Run full training: python train_classifier.py")
    print("  - Explore analysis notebook: notebooks/01_pattern_analysis.ipynb")
    print("  - Integrate with DNS service or proxy")
    print("=" * 70)


if __name__ == '__main__':
    main()
