"""
Feature extraction module for domain analysis.
Extracts hierarchical features from domain names for classification.
"""

from collections import Counter
import numpy as np
import re


class DomainFeatureExtractor:
    """Extract features from domain names for hierarchical classification"""

    def __init__(self):
        pass

    @staticmethod
    def extract_tld(domain: str) -> str:
        """Extract top-level domain"""
        parts = domain.split('.')
        if len(parts) >= 2:
            return parts[-1]
        return parts[0] if parts else ''

    @staticmethod
    def extract_sld(domain: str) -> str:
        """Extract second-level domain (main domain)"""
        parts = domain.split('.')
        if len(parts) >= 2:
            return parts[-2]
        return parts[0] if parts else ''

    @staticmethod
    def extract_subdomains(domain: str) -> list:
        """Extract all subdomains"""
        parts = domain.split('.')
        if len(parts) > 2:
            return parts[:-2]
        return []

    @staticmethod
    def calculate_entropy(text: str) -> float:
        """Calculate Shannon entropy of text"""
        if not text:
            return 0.0

        char_freq = Counter(text)
        length = len(text)
        entropy = -sum(
            (freq / length) * np.log2(freq / length)
            for freq in char_freq.values()
        )
        return entropy

    def extract_features(self, domain: str) -> dict:
        """
        Extract comprehensive features from domain name.

        Returns hierarchical features organized by:
        - Level 1: TLD features
        - Level 2: Domain pattern features
        - Level 3: Subdomain features
        """
        parts = domain.split('.')
        tld = self.extract_tld(domain)
        sld = self.extract_sld(domain)
        subdomains = self.extract_subdomains(domain)

        features = {
            # Level 1: TLD features
            'tld': tld,
            'tld_length': len(tld),

            # Level 2: Domain pattern features
            'domain_length': len(domain),
            'sld': sld,
            'sld_length': len(sld),
            'sld_entropy': self.calculate_entropy(sld),
            'sld_digit_ratio': sum(c.isdigit() for c in sld) / len(sld) if sld else 0,
            'sld_vowel_ratio': sum(c.lower() in 'aeiou' for c in sld) / len(sld) if sld else 0,

            # Level 3: Subdomain features
            'num_subdomains': len(subdomains),
            'num_dots': domain.count('.'),
            'has_www': 1 if domain.startswith('www.') else 0,

            # Additional pattern features
            'num_hyphens': domain.count('-'),
            'num_digits': sum(c.isdigit() for c in domain),
            'num_underscores': domain.count('_'),
            'consecutive_digits': max(
                [len(match) for match in re.findall(r'\d+', domain)] + [0]
            ),

            # Statistical features
            'domain_entropy': self.calculate_entropy(domain),
        }

        # Add subdomain-specific features if present
        if subdomains:
            features['longest_subdomain'] = max(len(s) for s in subdomains)
            features['avg_subdomain_length'] = sum(len(s) for s in subdomains) / len(subdomains)
        else:
            features['longest_subdomain'] = 0
            features['avg_subdomain_length'] = 0

        return features

    def extract_hierarchical_signature(self, domain: str) -> tuple:
        """
        Extract hierarchical signature for trie-based lookup.
        Returns (tld, sld_pattern, subdomain_pattern)
        """
        features = self.extract_features(domain)

        # TLD level
        tld = features['tld']

        # Domain pattern level (categorize SLD)
        sld_pattern = self._categorize_sld(features)

        # Subdomain pattern level
        subdomain_pattern = self._categorize_subdomains(features)

        return (tld, sld_pattern, subdomain_pattern)

    def _categorize_sld(self, features: dict) -> str:
        """Categorize second-level domain into pattern type"""
        length = features['sld_length']
        digit_ratio = features['sld_digit_ratio']
        entropy = features['sld_entropy']

        # Pattern categories
        if digit_ratio > 0.5:
            return 'high_digits'
        elif length > 20:
            return 'very_long'
        elif length > 15:
            return 'long'
        elif entropy > 4.0:
            return 'high_entropy'
        elif entropy < 2.0:
            return 'low_entropy'
        else:
            return 'normal'

    def _categorize_subdomains(self, features: dict) -> str:
        """Categorize subdomain structure into pattern type"""
        num_subdomains = features['num_subdomains']
        has_www = features['has_www']

        if num_subdomains == 0:
            return 'none'
        elif has_www and num_subdomains == 1:
            return 'www_only'
        elif num_subdomains >= 3:
            return 'deep'
        elif num_subdomains == 2:
            return 'moderate'
        else:
            return 'simple'


if __name__ == '__main__':
    # Test the feature extractor
    extractor = DomainFeatureExtractor()

    test_domains = [
        'google.com',
        'www.example.com',
        'ads.tracker.example.com',
        'malicious123.xyz',
        'very-long-suspicious-domain-name.top'
    ]

    for domain in test_domains:
        print(f"\nDomain: {domain}")
        features = extractor.extract_features(domain)
        for key, value in features.items():
            print(f"  {key}: {value}")

        signature = extractor.extract_hierarchical_signature(domain)
        print(f"  Hierarchical signature: {signature}")
