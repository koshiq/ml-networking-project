#!/usr/bin/env python3
"""
Training Script for Hybrid Classifier

This script:
1. Trains Tier 1 (trie) on domain features (fast lookup)
2. Fetches HTML content from sample domains
3. Trains Tier 2 (ML) on content features (slow but accurate)
4. Saves the hybrid model

This approach justifies ML by analyzing complex HTML patterns!
"""

import sys
import os
import pandas as pd
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from hybrid_classifier import HybridDomainClassifier
from content_fetcher import ContentFetcher
from content_feature_extractor import ContentFeatureExtractor


def collect_content_samples(df: pd.DataFrame, domain_col: str = 'domain',
                            label_col: str = 'label', num_samples: int = 500,
                            balance: bool = True) -> pd.DataFrame:
    """
    Fetch HTML content from sample domains for training.

    Args:
        df: DataFrame with domains and labels
        domain_col: Column name for domains
        label_col: Column name for labels
        num_samples: Number of domains to fetch
        balance: Balance malicious/benign samples

    Returns:
        DataFrame with domains, labels, and HTML content
    """
    print("=" * 70)
    print("COLLECTING HTML CONTENT FOR ML TRAINING")
    print("=" * 70)
    print(f"Target: {num_samples} samples")
    print("This may take several minutes...")
    print()

    # Balance classes if requested
    if balance:
        malicious = df[df[label_col] == 1].sample(n=min(num_samples//2, sum(df[label_col] == 1)), random_state=42)
        benign = df[df[label_col] == 0].sample(n=min(num_samples//2, sum(df[label_col] == 0)), random_state=42)
        sample_df = pd.concat([malicious, benign]).sample(frac=1, random_state=42)  # Shuffle
    else:
        sample_df = df.sample(n=min(num_samples, len(df)), random_state=42)

    print(f"Selected {len(sample_df)} domains to fetch:")
    print(f"  Malicious: {sum(sample_df[label_col] == 1)}")
    print(f"  Benign: {sum(sample_df[label_col] == 0)}")
    print()

    # Fetch content
    fetcher = ContentFetcher(timeout=8, max_retries=1)
    results = []

    for idx, (_, row) in enumerate(sample_df.iterrows()):
        domain = row[domain_col]
        label = row[label_col]

        print(f"[{idx+1}/{len(sample_df)}] Fetching {domain}...", end=' ')

        html = fetcher.fetch_domain(domain)

        if html:
            print(f"✓ ({len(html)} chars)")
            results.append({
                'domain': domain,
                'label': label,
                'html_content': html,
                'url': f'http://{domain}'
            })
        else:
            print("✗ Failed")

        # Progress update
        if (idx + 1) % 10 == 0:
            success_rate = len(results) / (idx + 1)
            print(f"  Progress: {len(results)} successful ({success_rate*100:.1f}%)")

    fetcher.close()

    print()
    print(f"✓ Successfully fetched content from {len(results)} domains")
    print(f"  Success rate: {len(results)/len(sample_df)*100:.1f}%")

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description='Train Hybrid Domain Classifier (Trie + Content ML)'
    )
    parser.add_argument(
        '--domain-data',
        type=str,
        default='data/labeled_domains.csv',
        help='CSV with domains and labels for trie training'
    )
    parser.add_argument(
        '--content-samples',
        type=int,
        default=500,
        help='Number of domains to fetch for content training'
    )
    parser.add_argument(
        '--use-cached-content',
        type=str,
        default=None,
        help='Path to pre-fetched content CSV (skips fetching)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/hybrid_classifier',
        help='Output directory for trained model'
    )
    parser.add_argument(
        '--skip-content-training',
        action='store_true',
        help='Skip content model training (trie only)'
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("HYBRID DOMAIN CLASSIFIER - TRAINING")
    print("=" * 70)
    print("\nTwo-Tier Architecture:")
    print("  Tier 1: Trie-based lookup (fast, O(1))")
    print("  Tier 2: Content-based ML (slow, high accuracy)")
    print("\nThis approach JUSTIFIES ML by analyzing complex HTML patterns!")
    print("=" * 70)

    # Load domain data
    print(f"\nLoading domain data from {args.domain_data}...")
    df = pd.read_csv(args.domain_data)
    print(f"✓ Loaded {len(df)} labeled domains")
    print(f"  Malicious: {sum(df['label'] == 1)}")
    print(f"  Benign: {sum(df['label'] == 0)}")

    # Initialize classifier
    classifier = HybridDomainClassifier(
        enable_content_fetch=True,
        fetch_timeout=8,
        cache_new_results=True
    )

    # TIER 1: Train trie on all domain data
    print(f"\nTraining Tier 1 on {len(df)} domains...")
    classifier.train_domain_trie(df, domain_col='domain', label_col='label')

    # TIER 2: Train content model
    if not args.skip_content_training:
        # Get content data
        if args.use_cached_content and os.path.exists(args.use_cached_content):
            print(f"\nLoading pre-fetched content from {args.use_cached_content}...")
            content_df = pd.read_csv(args.use_cached_content)
            print(f"✓ Loaded {len(content_df)} pages with content")
        else:
            print("\nCollecting HTML content for ML training...")
            content_df = collect_content_samples(
                df,
                domain_col='domain',
                label_col='label',
                num_samples=args.content_samples,
                balance=True
            )

            # Save collected content for future use
            cache_path = 'data/content_training_cache.csv'
            content_df.to_csv(cache_path, index=False)
            print(f"✓ Cached content to {cache_path}")

        # Train content model
        if len(content_df) >= 50:  # Need minimum samples
            classifier.train_content_model(
                content_df,
                content_col='html_content',
                label_col='label'
            )
        else:
            print(f"\n⚠ Not enough content samples ({len(content_df)}), skipping content training")
            print("  Need at least 50 samples for ML training")
    else:
        print("\n⚠ Skipping content model training (--skip-content-training)")

    # Save model
    print(f"\nSaving hybrid model to {args.output}/...")
    os.makedirs(args.output, exist_ok=True)
    classifier.save(args.output)

    # Show statistics
    stats = classifier.get_statistics()
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Trie entries: {stats['trie_entries']}")
    print(f"Content model trained: {'Yes' if classifier.content_model else 'No'}")
    print(f"\nModel saved to: {args.output}/")
    print("\nNext steps:")
    print("  1. Test the hybrid classifier: python demo_hybrid.py")
    print("  2. Run the proxy server: python proxy_server.py")
    print("=" * 70)


if __name__ == '__main__':
    main()