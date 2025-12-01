#!/usr/bin/env python3
"""
Rule-Based Domain Lookup

Fast dictionary lookup using the labeled training data (198K domains).
This is Tier 1 - fastest and most accurate for known domains.
"""

import pandas as pd
from typing import Optional, Tuple


class RuleBasedLookup:
    """
    Simple dictionary-based lookup for known domains.
    Uses the labeled_domains.csv as ground truth.
    """

    def __init__(self):
        self.domain_labels = {}  # domain -> label mapping
        self.loaded = False

    def load(self, csv_path: str = 'data/labeled_domains.csv'):
        """
        Load labeled domains from CSV.

        Args:
            csv_path: Path to labeled_domains.csv
        """
        print(f"Loading rule-based lookup from {csv_path}...")

        df = pd.read_csv(csv_path)

        # Create dictionary for O(1) lookup
        self.domain_labels = dict(zip(df['domain'], df['label']))

        self.loaded = True
        print(f"✓ Loaded {len(self.domain_labels):,} labeled domains")

    def lookup(self, domain: str) -> Optional[Tuple[int, float, str]]:
        """
        Lookup a domain in the rule-based list.

        Args:
            domain: Domain to lookup

        Returns:
            Tuple of (prediction, confidence, method) if found, None otherwise
            - prediction: 0 (legitimate) or 1 (malicious)
            - confidence: 1.0 (100% - this is ground truth)
            - method: 'rules'
        """
        if not self.loaded:
            return None

        # Exact match
        if domain in self.domain_labels:
            label = self.domain_labels[domain]
            return (label, 1.0, 'rules')

        # Try with/without 'www.'
        if domain.startswith('www.'):
            base_domain = domain[4:]
            if base_domain in self.domain_labels:
                label = self.domain_labels[base_domain]
                return (label, 1.0, 'rules')
        else:
            www_domain = f'www.{domain}'
            if www_domain in self.domain_labels:
                label = self.domain_labels[www_domain]
                return (label, 1.0, 'rules')

        # Not found
        return None

    def get_stats(self) -> dict:
        """Get statistics about the rule-based lookup"""
        if not self.loaded:
            return {
                'total_domains': 0,
                'legitimate_domains': 0,
                'malicious_domains': 0
            }

        values = list(self.domain_labels.values())
        return {
            'total_domains': len(self.domain_labels),
            'legitimate_domains': values.count(0),
            'malicious_domains': values.count(1)
        }


if __name__ == '__main__':
    # Test the lookup
    lookup = RuleBasedLookup()
    lookup.load('data/labeled_domains.csv')

    print("\nStats:", lookup.get_stats())

    print("\nTesting lookups:")
    test_domains = [
        'google.com',
        'doubleclick.net',
        'unknown-domain-12345.com'
    ]

    for domain in test_domains:
        result = lookup.lookup(domain)
        if result:
            pred, conf, method = result
            label = "MALICIOUS" if pred == 1 else "LEGITIMATE"
            print(f"  {domain:30s} → {label} (confidence: {conf:.0%}, method: {method})")
        else:
            print(f"  {domain:30s} → NOT FOUND")
