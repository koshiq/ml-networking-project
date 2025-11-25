"""
Basic unit tests for hierarchical domain classifier components.
"""

import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from feature_extraction import DomainFeatureExtractor
from trie_structure import HierarchicalDomainTrie


class TestFeatureExtraction(unittest.TestCase):
    """Test feature extraction module"""

    def setUp(self):
        self.extractor = DomainFeatureExtractor()

    def test_tld_extraction(self):
        """Test TLD extraction"""
        self.assertEqual(self.extractor.extract_tld('google.com'), 'com')
        self.assertEqual(self.extractor.extract_tld('example.co.uk'), 'uk')
        self.assertEqual(self.extractor.extract_tld('test.xyz'), 'xyz')

    def test_sld_extraction(self):
        """Test second-level domain extraction"""
        self.assertEqual(self.extractor.extract_sld('google.com'), 'google')
        self.assertEqual(self.extractor.extract_sld('www.example.com'), 'example')
        self.assertEqual(self.extractor.extract_sld('test.xyz'), 'test')

    def test_subdomain_extraction(self):
        """Test subdomain extraction"""
        self.assertEqual(self.extractor.extract_subdomains('google.com'), [])
        self.assertEqual(self.extractor.extract_subdomains('www.google.com'), ['www'])
        self.assertEqual(self.extractor.extract_subdomains('a.b.c.example.com'), ['a', 'b', 'c'])

    def test_feature_extraction(self):
        """Test comprehensive feature extraction"""
        features = self.extractor.extract_features('www.google.com')

        self.assertEqual(features['tld'], 'com')
        self.assertEqual(features['sld'], 'google')
        self.assertEqual(features['num_subdomains'], 1)
        self.assertEqual(features['has_www'], 1)
        self.assertGreater(features['domain_length'], 0)
        self.assertGreater(features['sld_entropy'], 0)

    def test_hierarchical_signature(self):
        """Test hierarchical signature generation"""
        sig = self.extractor.extract_hierarchical_signature('www.google.com')
        self.assertEqual(len(sig), 3)
        self.assertEqual(sig[0], 'com')  # TLD
        self.assertIsInstance(sig[1], str)  # Domain pattern
        self.assertIsInstance(sig[2], str)  # Subdomain pattern


class TestTrieStructure(unittest.TestCase):
    """Test hierarchical trie structure"""

    def setUp(self):
        self.trie = HierarchicalDomainTrie()

    def test_insertion(self):
        """Test trie insertion"""
        self.trie.insert(('com', 'normal', 'www_only'), 0, 0.95)
        self.assertEqual(self.trie.total_entries, 1)

    def test_lookup_exact_match(self):
        """Test exact match lookup"""
        signature = ('xyz', 'high_digits', 'none')
        self.trie.insert(signature, 1, 0.90)

        pred, conf, level = self.trie.lookup(signature)
        self.assertEqual(pred, 1)
        self.assertEqual(conf, 0.90)
        self.assertEqual(level, 3)

    def test_lookup_no_match(self):
        """Test lookup with no match"""
        self.trie.insert(('com', 'normal', 'www_only'), 0, 0.95)

        pred, conf, level = self.trie.lookup(('xyz', 'unknown', 'none'))
        self.assertIsNone(pred)
        self.assertEqual(conf, 0.0)
        self.assertEqual(level, 0)

    def test_fallback_matching(self):
        """Test hierarchical fallback matching"""
        # Insert with fallback
        self.trie.insert_with_fallback(('xyz', 'high_digits', 'none'), 1, 0.90)

        # Should match at TLD level even with different pattern
        pred, conf, level = self.trie.lookup(('xyz', 'different', 'none'))
        self.assertEqual(pred, 1)
        self.assertGreater(conf, 0)
        self.assertGreater(level, 0)

    def test_multiple_insertions(self):
        """Test multiple insertions"""
        signatures = [
            (('com', 'normal', 'www_only'), 0, 0.95),
            (('xyz', 'high_digits', 'none'), 1, 0.90),
            (('net', 'normal', 'none'), 0, 0.85),
        ]

        for sig, pred, conf in signatures:
            self.trie.insert(sig, pred, conf)

        self.assertEqual(self.trie.total_entries, 3)

        # Verify all can be looked up
        for sig, expected_pred, _ in signatures:
            pred, _, _ = self.trie.lookup(sig)
            self.assertEqual(pred, expected_pred)

    def test_statistics(self):
        """Test trie statistics"""
        self.trie.insert(('com', 'normal', 'www_only'), 0, 0.95)
        self.trie.insert(('com', 'normal', 'none'), 0, 0.90)
        self.trie.insert(('xyz', 'high_digits', 'none'), 1, 0.85)

        stats = self.trie.get_statistics()
        self.assertEqual(stats['total_entries'], 3)
        self.assertGreater(stats['num_tlds'], 0)


class TestIntegration(unittest.TestCase):
    """Integration tests"""

    def test_end_to_end_workflow(self):
        """Test complete workflow from domain to prediction"""
        extractor = DomainFeatureExtractor()
        trie = HierarchicalDomainTrie()

        # Training data
        training_domains = [
            ('google.com', 0),
            ('facebook.com', 0),
            ('malicious.xyz', 1),
            ('spam.xyz', 1),
        ]

        # Extract signatures and insert into trie
        for domain, label in training_domains:
            sig = extractor.extract_hierarchical_signature(domain)
            trie.insert_with_fallback(sig, label, 0.90)

        # Test predictions
        test_domain = 'test.xyz'
        sig = extractor.extract_hierarchical_signature(test_domain)
        pred, conf, level = trie.lookup(sig)

        # Should predict malicious due to .xyz TLD
        self.assertEqual(pred, 1)
        self.assertGreater(conf, 0)


if __name__ == '__main__':
    unittest.main()
