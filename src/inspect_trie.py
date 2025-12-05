#!/usr/bin/env python3
"""
Trie Inspector - View what's stored in the trie
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trie_structure import HierarchicalDomainTrie


def inspect_trie(trie_path: str):
    """Load and display trie contents"""

    if not os.path.exists(trie_path):
        print(f"❌ Trie file not found: {trie_path}")
        print("\nRun example_usage.py first to create the trie!")
        return

    print("=" * 70)
    print("Trie Inspector")
    print("=" * 70)
    print(f"\nLoading trie from: {trie_path}\n")

    # Load trie
    trie = HierarchicalDomainTrie()
    trie.load(trie_path)

    # Get statistics
    stats = trie.get_statistics()

    print("📊 Trie Statistics:")
    print(f"   Total entries:     {stats['total_entries']}")
    print(f"   TLD nodes:         {stats['levels']['tld_nodes']}")
    print(f"   Pattern nodes:     {stats['levels']['pattern_nodes']}")
    print(f"   Terminal nodes:    {stats['levels']['terminal_nodes']}")

    # Show all entries
    print("\n" + "=" * 70)
    print("Stored Domains")
    print("=" * 70)

    entries = []

    # Walk through trie and collect all entries
    def walk_trie(node, path, level):
        if node.is_terminal:
            entries.append({
                'signature': path,
                'prediction': node.prediction,
                'confidence': node.confidence,
                'metadata': node.metadata
            })

        for key, child in node.children.items():
            walk_trie(child, path + (key,), level + 1)

    walk_trie(trie.root, (), 0)

    # Group by prediction
    ads = [e for e in entries if e['prediction'] == 1]
    legitimate = [e for e in entries if e['prediction'] == 0]

    print(f"\n🚫 Advertisements: {len(ads)}")
    print("-" * 70)
    for i, entry in enumerate(ads, 1):
        sig = entry['signature']
        conf = entry['confidence']
        meta = entry['metadata']
        domain = meta.get('domain', 'N/A')
        source = meta.get('source', 'N/A')

        print(f"\n{i}. Domain: {domain}")
        print(f"   Signature: {sig}")
        print(f"   Confidence: {conf:.2f}")
        print(f"   Source: {source}")

    print(f"\n✅ Legitimate Sites: {len(legitimate)}")
    print("-" * 70)
    for i, entry in enumerate(legitimate, 1):
        sig = entry['signature']
        conf = entry['confidence']
        meta = entry['metadata']
        domain = meta.get('domain', 'N/A')
        source = meta.get('source', 'N/A')

        print(f"\n{i}. Domain: {domain}")
        print(f"   Signature: {sig}")
        print(f"   Confidence: {conf:.2f}")
        print(f"   Source: {source}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total domains classified: {len(entries)}")
    print(f"Ads blocked: {len(ads)}")
    print(f"Legitimate allowed: {len(legitimate)}")
    print(f"Block rate: {len(ads) / max(len(entries), 1) * 100:.1f}%")


def main():
    # Default trie paths to check
    possible_paths = [
        'data/updated_domain_trie.json',
        'data/enhanced_domain_trie.json',
        'data/domain_trie.json',
    ]

    # Check if user provided path
    if len(sys.argv) > 1:
        trie_path = sys.argv[1]
    else:
        # Find first existing trie file
        trie_path = None
        for path in possible_paths:
            if os.path.exists(path):
                trie_path = path
                break

        if not trie_path:
            print("❌ No trie file found!")
            print("\nSearched for:")
            for path in possible_paths:
                print(f"  - {path}")
            print("\nRun one of these first:")
            print("  uv run python src/example_usage.py")
            print("  uv run python src/example_contextual.py")
            return

    inspect_trie(trie_path)


if __name__ == '__main__':
    main()
