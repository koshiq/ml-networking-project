#!/usr/bin/env python3
"""
Training script for Hierarchical Domain Classifier.
Trains the classifier on DNS training data and saves the model.
"""

import sys
import os
import pandas as pd
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from hierarchical_classifier import HierarchicalDomainClassifier


def main():
    parser = argparse.ArgumentParser(
        description='Train Hierarchical Domain Classifier'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='../Data/dns_training_data.csv',
        help='Path to training data CSV'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/hierarchical_classifier',
        help='Output directory for trained model'
    )
    parser.add_argument(
        '--min-support',
        type=int,
        default=10,
        help='Minimum support for rule generation'
    )
    parser.add_argument(
        '--min-precision',
        type=float,
        default=0.6,
        help='Minimum precision for rules'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Sample N rows for quick testing'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Hierarchical Domain Classifier - Training Script")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Data: {args.data}")
    print(f"  Output: {args.output}")
    print(f"  Min Support: {args.min_support}")
    print(f"  Min Precision: {args.min_precision}")
    if args.sample:
        print(f"  Sample size: {args.sample}")
    print()

    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} domains")

    if args.sample:
        print(f"Sampling {args.sample} rows for testing...")
        df = df.sample(n=min(args.sample, len(df)), random_state=42)

    print(f"Label distribution:")
    print(df['label'].value_counts())
    print()

    # Initialize classifier
    classifier = HierarchicalDomainClassifier(
        min_support=args.min_support,
        min_precision=args.min_precision
    )

    # Train
    classifier.train(df, domain_col='domain', label_col='label')

    # Save model
    print(f"\nSaving model to {args.output}/...")
    classifier.save(args.output)

    # Benchmark
    print("\nBenchmarking lookup performance...")
    test_domains = df['domain'].sample(n=min(1000, len(df))).tolist()
    benchmark = classifier.benchmark_lookup_time(test_domains)
    print(f"  Average lookup time: {benchmark['avg_time_ms']:.4f} ms")
    print(f"  Lookups per second: {benchmark['lookups_per_second']:.0f}")

    # Test predictions
    print("\n" + "=" * 70)
    print("Sample Predictions:")
    print("=" * 70)

    test_samples = [
        'google.com',
        'facebook.com',
        'ads.tracker.example.com',
        'malicious123.xyz',
        'very-long-suspicious-domain-name.top'
    ]

    for domain in test_samples:
        explanation = classifier.explain_prediction(domain)
        print(f"\nDomain: {domain}")
        print(f"  Prediction: {explanation['prediction']}")
        print(f"  Confidence: {explanation['confidence']:.3f}")
        print(f"  Match Level: {explanation['match_level']}")
        print(f"  Signature: TLD={explanation['signature']['tld']}, "
              f"Pattern={explanation['signature']['domain_pattern']}, "
              f"Subdomains={explanation['signature']['subdomain_pattern']}")
        print(f"  Reasoning: {explanation['reasoning']}")

    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
