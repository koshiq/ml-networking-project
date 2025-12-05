#!/usr/bin/env python3
"""
Quick summary viewer for proxy_trie.json
Shows what domains are blocked vs allowed in an easy-to-read format
"""

import json
import sys
from pathlib import Path

def collect_domains(node, path="root", domains=None):
    """Recursively collect all terminal nodes (actual domains)"""
    if domains is None:
        domains = []

    if node.get('is_terminal', False):
        domains.append({
            'path': path,
            'prediction': node.get('prediction'),
            'confidence': node.get('confidence', 0.0),
            'domain': node.get('metadata', {}).get('domain', 'unknown'),
            'source': node.get('metadata', {}).get('source', 'unknown')
        })

    for key, child in node.get('children', {}).items():
        collect_domains(child, f"{path} → {key}", domains)

    return domains


def main():
    # Load trie
    trie_path = sys.argv[1] if len(sys.argv) > 1 else 'data/proxy_trie.json'

    if not Path(trie_path).exists():
        print(f"❌ File not found: {trie_path}")
        print("\nUsage: python src/view_trie_summary.py [path/to/trie.json]")
        print("Example: python src/view_trie_summary.py data/proxy_trie.json")
        return

    with open(trie_path, 'r') as f:
        trie = json.load(f)

    # Collect all domains
    domains = collect_domains(trie['root'])

    # Separate ads and legitimate
    ads = [d for d in domains if d['prediction'] == 1]
    legitimate = [d for d in domains if d['prediction'] == 0]

    # Sort by confidence (descending)
    ads.sort(key=lambda x: x['confidence'], reverse=True)
    legitimate.sort(key=lambda x: x['domain'])

    # Print summary
    print("=" * 80)
    print("🌳 Trie Summary")
    print("=" * 80)
    print(f"📊 Total domains classified: {len(domains)}")
    print(f"🚫 Blocked (ads):            {len(ads)}")
    print(f"✅ Allowed (legitimate):     {len(legitimate)}")
    print("=" * 80)

    # Print blocked domains
    if ads:
        print("\n🚫 BLOCKED DOMAINS (Advertisements & Tracking):")
        print("-" * 80)
        for i, d in enumerate(ads, 1):
            conf_emoji = "🔴" if d['confidence'] >= 0.7 else "🟡"
            print(f"{i:2}. {conf_emoji} {d['domain']}")
            print(f"    Confidence: {d['confidence']:.2f} | Path: {d['path']}")

    # Print legitimate domains (first 20)
    if legitimate:
        print("\n✅ ALLOWED DOMAINS (Legitimate Services):")
        print("-" * 80)
        show_count = min(20, len(legitimate))
        for i, d in enumerate(legitimate[:show_count], 1):
            print(f"{i:2}. {d['domain']}")

        if len(legitimate) > show_count:
            print(f"\n    ... and {len(legitimate) - show_count} more legitimate domains")

    # Stats by TLD
    print("\n📊 Breakdown by TLD:")
    print("-" * 80)
    tld_stats = {}
    for d in domains:
        path_parts = d['path'].split(' → ')
        if len(path_parts) >= 2:
            tld = path_parts[1]  # First element after root
            if tld not in tld_stats:
                tld_stats[tld] = {'total': 0, 'blocked': 0, 'allowed': 0}
            tld_stats[tld]['total'] += 1
            if d['prediction'] == 1:
                tld_stats[tld]['blocked'] += 1
            else:
                tld_stats[tld]['allowed'] += 1

    # Sort by total domains
    tld_list = sorted(tld_stats.items(), key=lambda x: x[1]['total'], reverse=True)

    for tld, stats in tld_list[:10]:
        block_pct = (stats['blocked'] / stats['total'] * 100) if stats['total'] > 0 else 0
        bar = "█" * (stats['blocked']) + "░" * (stats['allowed'])
        print(f".{tld:12} | {stats['total']:3} total | 🚫 {stats['blocked']:2} | ✅ {stats['allowed']:2} | {block_pct:5.1f}% ads")

    print("\n" + "=" * 80)
    print("💡 Tip: Run 'python src/inspect_trie.py data/proxy_trie.json' for full details")
    print("=" * 80)


if __name__ == '__main__':
    main()
