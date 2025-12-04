#!/usr/bin/env python3
"""
Example usage of URL Classifier with Groq API
"""

import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trie_structure import HierarchicalDomainTrie
from src.url_classifier import URLClassifier
from src.config_loader import config


def main():
    print("=" * 70)
    print("URL Classifier with Groq + Gemma 2 9B")
    print("=" * 70)

    # Step 1: Initialize or load trie
    print("\n[1] Initializing trie...")
    trie = HierarchicalDomainTrie()

    # Pre-populate with some known ad domains
    known_ads = [
        (('com', 'normal', 'none'), 'doubleclick.com', 1.0),
        (('net', 'normal', 'none'), 'adserver.net', 1.0),
        (('com', 'normal', 'none'), 'googlesyndication.com', 1.0),
        (('com', 'normal', 'single'), 'googleadservices.com', 1.0),  # ads.googleadservices.com
        (('xyz', 'high_digits', 'none'), 'ad123xyz.xyz', 0.95),
    ]

    for sig, domain, conf in known_ads:
        trie.insert(sig, prediction=1, confidence=conf,
                   metadata={'source': 'known_ad', 'domain': domain})

    print(f"   ✓ Loaded {len(known_ads)} known ad domains")

    # Step 2: Initialize classifier
    print("\n[2] Initializing classifier...")
    print(f"   Configuration: {config}")

    is_valid = config.validate()

    if is_valid:
        print(f"   ✓ Using Groq API with {config.model_name}")
        classifier = URLClassifier(trie, groq_api_key=config.groq_api_key, model_name=config.model_name)
    else:
        print("   ⚠️  Using heuristics only (no API key)")
        classifier = URLClassifier(trie, groq_api_key=None)

    # Step 3: Test URLs
    print("\n[3] Testing URL classification...")
    print("=" * 70)

    test_urls = [
        # Known ads (should be instant cache hits)
        'http://doubleclick.com/ad/123',
        'http://ads.googleadservices.com/pixel',

        # Unknown domains (will trigger background classification)
        'http://example.com',
        'http://github.com',
        'http://stackoverflow.com',

        # Suspicious looking domains
        'http://click-here-now.xyz',
        'http://track123.top',
    ]

    print("\n⚡ User-facing responses (instant, non-blocking):\n")

    for url in test_urls:
        result = classifier.check_url(url)

        action_emoji = "🚫" if result['action'] == 'block' else "✅"
        source_emoji = "💾" if result['source'] == 'cache' else "🔄"

        print(f"{action_emoji} {url}")
        print(f"   {source_emoji} {result['source']:<20} | Action: {result['action'].upper():<7} | Confidence: {result['confidence']:.2f}")

        time.sleep(0.2)  # Simulate real user requests

    # Step 4: Show immediate statistics
    print("\n" + "=" * 70)
    print("[4] Immediate Statistics")
    print("=" * 70)
    stats = classifier.get_statistics()

    print(f"Total checks:          {stats['total_checks']}")
    print(f"Cache hits:            {stats['cache_hits']} ({stats['cache_hit_rate']})")
    print(f"Cache misses:          {stats['cache_misses']}")
    print(f"Pending classification: {stats['classifications_pending']}")
    print(f"Queue size:            {stats['queue_size']}")

    # Step 5: Wait for background processing
    print("\n" + "=" * 70)
    print("[5] Background Classification in Progress...")
    print("=" * 70)
    print("(This happens asynchronously - users never wait for this!)\n")

    if stats['classifications_pending'] > 0:
        classifier.wait_for_pending(timeout=120)
    else:
        print("No classifications pending.")

    # Step 6: Final statistics
    print("\n" + "=" * 70)
    print("[6] Final Statistics")
    print("=" * 70)
    final_stats = classifier.get_statistics()

    print(f"Total checks:           {final_stats['total_checks']}")
    print(f"Cache hits:             {final_stats['cache_hits']} ({final_stats['cache_hit_rate']})")
    print(f"Ads detected:           {final_stats['ads_detected']}")
    print(f"Legitimate detected:    {final_stats['legitimate_detected']}")
    print(f"Classifications done:   {final_stats['classifications_completed']}")

    print("\nTrie Statistics:")
    trie_stats = final_stats['trie_stats']
    print(f"  Total entries:        {trie_stats['total_entries']}")
    print(f"  TLD nodes:            {trie_stats['levels']['tld_nodes']}")
    print(f"  Pattern nodes:        {trie_stats['levels']['pattern_nodes']}")
    print(f"  Terminal nodes:       {trie_stats['levels']['terminal_nodes']}")

    # Step 7: Save updated trie
    print("\n" + "=" * 70)
    print("[7] Saving Updated Trie")
    print("=" * 70)

    os.makedirs('data', exist_ok=True)
    trie_path = 'data/updated_domain_trie.json'
    classifier.save_trie(trie_path)
    print(f"✓ Trie saved to: {trie_path}")

    # Step 8: Test cache hits after classification
    print("\n" + "=" * 70)
    print("[8] Testing Cache After Background Classification")
    print("=" * 70)
    print("\nRe-checking previously unknown URLs (should now be cache hits):\n")

    recheck_urls = ['http://example.com', 'http://github.com']

    for url in recheck_urls:
        result = classifier.check_url(url)

        action_emoji = "🚫" if result['action'] == 'block' else "✅"
        source_emoji = "💾" if result['source'] == 'cache' else "🔄"

        print(f"{action_emoji} {url}")
        print(f"   {source_emoji} {result['source']:<20} | Action: {result['action'].upper():<7} | Confidence: {result['confidence']:.2f}")

    print("\n" + "=" * 70)
    print("✓ Demo Complete!")
    print("=" * 70)

    classifier.close()


if __name__ == '__main__':
    main()
