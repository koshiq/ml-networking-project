"""
Efficient hierarchical trie structure for domain classification.
Implements O(log n) lookup through hierarchical organization.
"""

from typing import Dict, Any, Optional, List, Tuple
import json


class TrieNode:
    """Node in the hierarchical trie structure"""

    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_terminal = False
        self.prediction: Optional[int] = None
        self.confidence: float = 0.0
        self.metadata: Dict[str, Any] = {}

    def __repr__(self):
        return f"TrieNode(terminal={self.is_terminal}, pred={self.prediction}, conf={self.confidence:.3f}, children={len(self.children)})"


class HierarchicalDomainTrie:
    """
    Hierarchical trie for efficient domain classification.

    Structure:
    Level 1: TLD (e.g., 'com', 'net', 'xyz')
    Level 2: Domain pattern (e.g., 'high_digits', 'very_long', 'normal')
    Level 3: Subdomain pattern (e.g., 'none', 'www_only', 'deep')

    This enables O(1) to O(log n) lookups depending on specificity.
    """

    def __init__(self):
        self.root = TrieNode()
        self.total_entries = 0
        self.default_prediction = 0  # Default: legitimate

    def insert(self, signature: Tuple[str, str, str],
               prediction: int, confidence: float = 1.0,
               metadata: Dict[str, Any] = None):
        """
        Insert a domain signature into the trie.

        Args:
            signature: (tld, domain_pattern, subdomain_pattern)
            prediction: 0 (legitimate) or 1 (malicious)
            confidence: Confidence score for this prediction
            metadata: Additional information (e.g., rule source, precision)
        """
        tld, domain_pattern, subdomain_pattern = signature

        # Navigate to TLD level
        if tld not in self.root.children:
            self.root.children[tld] = TrieNode()
        tld_node = self.root.children[tld]

        # Navigate to domain pattern level
        if domain_pattern not in tld_node.children:
            tld_node.children[domain_pattern] = TrieNode()
        domain_node = tld_node.children[domain_pattern]

        # Navigate to subdomain pattern level (terminal)
        if subdomain_pattern not in domain_node.children:
            domain_node.children[subdomain_pattern] = TrieNode()
        terminal_node = domain_node.children[subdomain_pattern]

        # Set terminal node properties
        terminal_node.is_terminal = True
        terminal_node.prediction = prediction
        terminal_node.confidence = confidence
        if metadata:
            terminal_node.metadata = metadata

        self.total_entries += 1

    def lookup(self, signature: Tuple[str, str, str]) -> Tuple[Optional[int], float, int]:
        """
        Look up a domain signature in the trie.

        Returns:
            (prediction, confidence, match_level)
            - prediction: 0 (legitimate), 1 (malicious), or None (no match)
            - confidence: Confidence score
            - match_level: 0 (no match), 1 (TLD), 2 (domain pattern), 3 (full match)
        """
        tld, domain_pattern, subdomain_pattern = signature

        # Try full match (level 3)
        if tld in self.root.children:
            tld_node = self.root.children[tld]

            if domain_pattern in tld_node.children:
                domain_node = tld_node.children[domain_pattern]

                if subdomain_pattern in domain_node.children:
                    terminal_node = domain_node.children[subdomain_pattern]
                    if terminal_node.is_terminal:
                        return (terminal_node.prediction, terminal_node.confidence, 3)

                # Partial match: TLD + domain pattern (level 2)
                # Check if domain_node has a default prediction
                if domain_node.is_terminal:
                    return (domain_node.prediction, domain_node.confidence * 0.8, 2)

            # Partial match: TLD only (level 1)
            if tld_node.is_terminal:
                return (tld_node.prediction, tld_node.confidence * 0.6, 1)

        # No match
        return (None, 0.0, 0)

    def insert_with_fallback(self, signature: Tuple[str, str, str],
                            prediction: int, confidence: float = 1.0,
                            metadata: Dict[str, Any] = None):
        """
        Insert with fallback predictions at higher levels.
        This enables partial matching for unknown patterns.
        """
        tld, domain_pattern, subdomain_pattern = signature

        # Insert at all levels for fallback
        # Level 1: TLD only
        if tld not in self.root.children:
            self.root.children[tld] = TrieNode()
        tld_node = self.root.children[tld]

        # Update TLD-level statistics
        if not tld_node.is_terminal:
            tld_node.is_terminal = True
            tld_node.prediction = prediction
            tld_node.confidence = confidence * 0.6

        # Level 2: TLD + domain pattern
        if domain_pattern not in tld_node.children:
            tld_node.children[domain_pattern] = TrieNode()
        domain_node = tld_node.children[domain_pattern]

        if not domain_node.is_terminal:
            domain_node.is_terminal = True
            domain_node.prediction = prediction
            domain_node.confidence = confidence * 0.8

        # Level 3: Full signature (terminal)
        if subdomain_pattern not in domain_node.children:
            domain_node.children[subdomain_pattern] = TrieNode()
        terminal_node = domain_node.children[subdomain_pattern]

        terminal_node.is_terminal = True
        terminal_node.prediction = prediction
        terminal_node.confidence = confidence
        if metadata:
            terminal_node.metadata = metadata

        self.total_entries += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get trie statistics"""
        stats = {
            'total_entries': self.total_entries,
            'num_tlds': len(self.root.children),
            'levels': {}
        }

        # Count nodes at each level
        tld_count = 0
        pattern_count = 0
        terminal_count = 0

        for tld, tld_node in self.root.children.items():
            tld_count += 1
            for pattern, pattern_node in tld_node.children.items():
                pattern_count += 1
                terminal_count += len(pattern_node.children)

        stats['levels'] = {
            'tld_nodes': tld_count,
            'pattern_nodes': pattern_count,
            'terminal_nodes': terminal_count
        }

        return stats

    def save(self, filepath: str):
        """Save trie to JSON file"""
        def node_to_dict(node: TrieNode) -> dict:
            return {
                'is_terminal': node.is_terminal,
                'prediction': node.prediction,
                'confidence': node.confidence,
                'metadata': node.metadata,
                'children': {k: node_to_dict(v) for k, v in node.children.items()}
            }

        trie_data = {
            'total_entries': self.total_entries,
            'default_prediction': self.default_prediction,
            'root': node_to_dict(self.root)
        }

        with open(filepath, 'w') as f:
            json.dump(trie_data, f, indent=2)
        print(f"Trie saved to {filepath}")

    def load(self, filepath: str):
        """Load trie from JSON file"""
        def dict_to_node(data: dict) -> TrieNode:
            node = TrieNode()
            node.is_terminal = data.get('is_terminal', False)
            node.prediction = data.get('prediction')
            node.confidence = data.get('confidence', 0.0)
            node.metadata = data.get('metadata', {})
            node.children = {k: dict_to_node(v) for k, v in data.get('children', {}).items()}
            return node

        with open(filepath, 'r') as f:
            trie_data = json.load(f)

        self.total_entries = trie_data['total_entries']
        self.default_prediction = trie_data['default_prediction']
        self.root = dict_to_node(trie_data['root'])
        print(f"Trie loaded from {filepath}")


class RuleBasedTrie(HierarchicalDomainTrie):
    """
    Extended trie that stores rules at each level.
    Enables rule-based classification with hierarchical fallback.
    """

    def __init__(self):
        super().__init__()
        self.rules_by_level = {
            1: [],  # TLD-level rules
            2: [],  # Domain pattern rules
            3: []   # Full signature rules
        }

    def insert_rule(self, signature: Tuple[str, str, str],
                   prediction: int, confidence: float,
                   rule_metadata: Dict[str, Any]):
        """Insert a rule with its metadata"""
        self.insert_with_fallback(signature, prediction, confidence, rule_metadata)

        # Store rule at appropriate level
        level = 3  # Default to most specific
        if signature[2] == '*':
            level = 2
        if signature[1] == '*':
            level = 1

        self.rules_by_level[level].append({
            'signature': signature,
            'prediction': prediction,
            'confidence': confidence,
            'metadata': rule_metadata
        })

    def get_applicable_rules(self, signature: Tuple[str, str, str]) -> List[Dict]:
        """Get all rules applicable to a signature (from general to specific)"""
        tld, domain_pattern, subdomain_pattern = signature
        applicable = []

        # Check TLD-level rules
        for rule in self.rules_by_level[1]:
            if rule['signature'][0] == tld:
                applicable.append(rule)

        # Check domain pattern rules
        for rule in self.rules_by_level[2]:
            if rule['signature'][0] == tld and rule['signature'][1] == domain_pattern:
                applicable.append(rule)

        # Check full signature rules
        for rule in self.rules_by_level[3]:
            if rule['signature'] == signature:
                applicable.append(rule)

        return applicable


if __name__ == '__main__':
    # Test hierarchical trie
    print("Testing HierarchicalDomainTrie...")

    trie = HierarchicalDomainTrie()

    # Insert some test signatures
    test_data = [
        (('xyz', 'high_digits', 'none'), 1, 0.95),
        (('com', 'normal', 'www_only'), 0, 0.90),
        (('top', 'very_long', 'deep'), 1, 0.85),
        (('net', 'normal', 'none'), 0, 0.80),
    ]

    for sig, pred, conf in test_data:
        trie.insert_with_fallback(sig, pred, conf)

    print(f"\nTrie statistics: {trie.get_statistics()}")

    # Test lookups
    test_lookups = [
        ('xyz', 'high_digits', 'none'),
        ('xyz', 'normal', 'none'),  # Partial match: TLD only
        ('com', 'normal', 'www_only'),
        ('unknown', 'unknown', 'unknown'),  # No match
    ]

    print("\nTest lookups:")
    for sig in test_lookups:
        pred, conf, level = trie.lookup(sig)
        print(f"  {sig}")
        print(f"    -> Prediction: {pred}, Confidence: {conf:.3f}, Match level: {level}")
