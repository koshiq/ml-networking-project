#!/usr/bin/env python3
"""
Enhanced URL Classifier Demo with Contextual Awareness
Shows how network metadata + context improves classification accuracy
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trie_structure import HierarchicalDomainTrie
from src.url_classifier import URLClassifier
from src.config_loader import config


def main():
    print("=" * 80)
    print("Enhanced URL Classifier - Network Metadata + Contextual Awareness")
    print("=" * 80)

    # Initialize
    trie = HierarchicalDomainTrie()

    # Pre-populate with known ads
    known_ads = [
        (('com', 'normal', 'none'), 'doubleclick.com', 1.0),
        (('com', 'normal', 'none'), 'googlesyndication.com', 1.0),
        (('net', 'normal', 'none'), 'adserver.net', 1.0),
    ]

    for sig, domain, conf in known_ads:
        trie.insert(sig, prediction=1, confidence=conf,
                   metadata={'source': 'known_ad', 'domain': domain})

    print(f"\n✓ Loaded {len(known_ads)} known ad domains")

    # Initialize classifier
    is_valid = config.validate()
    if is_valid:
        print(f"✓ Using Groq API with {config.model_name}")
        classifier = URLClassifier(trie, groq_api_key=config.groq_api_key,
                                  model_name=config.model_name)
    else:
        print("⚠️  Using heuristics only (no API key)")
        classifier = URLClassifier(trie, groq_api_key=None)

    print("\n" + "=" * 80)
    print("Scenario 1: User browsing NBA.com (Sports Content)")
    print("=" * 80)
    print("\nSimulating user on nba.com viewing game highlights...")

    # Test contextual classification
    test_cases_nba = [
        {
            'url': 'http://espn.com',
            'description': 'ESPN - Sports site (RELEVANT)',
            'expected': 'ALLOW'
        },
        {
            'url': 'http://nike.com',
            'description': 'Nike - Sports apparel (RELEVANT)',
            'expected': 'ALLOW'
        },
        {
            'url': 'http://dating-singles.xyz',
            'description': 'Dating site (IRRELEVANT - likely ad)',
            'expected': 'BLOCK'
        },
        {
            'url': 'http://weight-loss-miracle.com',
            'description': 'Weight loss (IRRELEVANT - likely ad)',
            'expected': 'BLOCK'
        },
    ]

    for test in test_cases_nba:
        result = classifier.check_url(
            test['url'],
            current_page_url='https://nba.com/game/highlights',
            page_topic='sports'
        )

        action_emoji = "🚫" if result['action'] == 'block' else "✅"
        source_emoji = "💾" if result['source'] == 'cache' else "🔄"

        print(f"\n{action_emoji} {test['description']}")
        print(f"   URL: {test['url']}")
        print(f"   {source_emoji} {result['source']:<22} | Expected: {test['expected']}")
        time.sleep(0.1)

    print("\n" + "=" * 80)
    print("Scenario 2: User browsing News Site")
    print("=" * 80)
    print("\nSimulating user on cnn.com reading news article...")

    test_cases_news = [
        {
            'url': 'http://reuters.com',
            'description': 'Reuters - News site (RELEVANT)',
            'expected': 'ALLOW'
        },
        {
            'url': 'http://casino-jackpot.bet',
            'description': 'Casino/gambling (IRRELEVANT - likely ad)',
            'expected': 'BLOCK'
        },
        {
            'url': 'http://tracker-analytics.net',
            'description': 'Analytics tracker (AD)',
            'expected': 'BLOCK'
        },
    ]

    for test in test_cases_news:
        result = classifier.check_url(
            test['url'],
            current_page_url='https://cnn.com/world/article',
            page_topic='news'
        )

        action_emoji = "🚫" if result['action'] == 'block' else "✅"
        source_emoji = "💾" if result['source'] == 'cache' else "🔄"

        print(f"\n{action_emoji} {test['description']}")
        print(f"   URL: {test['url']}")
        print(f"   {source_emoji} {result['source']:<22} | Expected: {test['expected']}")
        time.sleep(0.1)

    print("\n" + "=" * 80)
    print("Scenario 3: Quick Filter Tests (Pre-LLM Blocking)")
    print("=" * 80)
    print("\nThese should be instantly blocked by network metadata analysis...")

    quick_filter_tests = [
        ('http://tracking-pixel.com/1x1.gif', 'Tracking pixel (tiny image)'),
        ('http://analytics.doubleclick.net/collect', 'DoubleClick analytics'),
        ('http://beacon.facebook.com/track', 'Facebook tracking beacon'),
    ]

    for url, description in quick_filter_tests:
        result = classifier.check_url(url)

        action_emoji = "🚫" if result['action'] == 'block' else "✅"

        print(f"\n{action_emoji} {description}")
        print(f"   URL: {url}")
        print(f"   Should be caught by quick filter")
        time.sleep(0.1)

    # Wait for background classification
    print("\n" + "=" * 80)
    print("Background Classification in Progress...")
    print("=" * 80)

    pending = classifier.stats['classifications_pending']
    if pending > 0:
        print(f"\n⏳ Waiting for {pending} classifications to complete...")
        print("(This demonstrates async processing - users never wait!)\n")
        classifier.wait_for_pending(timeout=120)
    else:
        print("\n✓ No pending classifications (all were cache hits or quick-filtered)\n")

    # Final statistics
    print("\n" + "=" * 80)
    print("Final Statistics")
    print("=" * 80)

    stats = classifier.get_statistics()

    print(f"\n📊 Classification Stats:")
    print(f"   Total checks:           {stats['total_checks']}")
    print(f"   Cache hits:             {stats['cache_hits']} ({stats['cache_hit_rate']})")
    print(f"   Cache misses:           {stats['cache_misses']}")
    print(f"   Ads detected:           {stats['ads_detected']}")
    print(f"   Legitimate detected:    {stats['legitimate_detected']}")
    print(f"   Classifications done:   {stats['classifications_completed']}")

    print(f"\n🌲 Trie Stats:")
    trie_stats = stats['trie_stats']
    print(f"   Total entries:          {trie_stats['total_entries']}")
    print(f"   TLD nodes:              {trie_stats['levels']['tld_nodes']}")
    print(f"   Terminal nodes:         {trie_stats['levels']['terminal_nodes']}")

    # Save updated trie
    os.makedirs('data', exist_ok=True)
    trie_path = 'data/enhanced_domain_trie.json'
    classifier.save_trie(trie_path)
    print(f"\n💾 Trie saved to: {trie_path}")

    print("\n" + "=" * 80)
    print("Key Improvements Demonstrated:")
    print("=" * 80)
    print("""
✓ Network Metadata Analysis
  - HTTP headers reveal tracking pixels, ad servers
  - Content-Type detection (images, scripts)
  - Response size analysis (tiny = tracking)
  - Redirect chain detection

✓ Contextual Relevance
  - Dating ads blocked on sports sites
  - Gambling ads blocked on news sites
  - Content relevance to user's current page
  - Topic-aware classification

✓ Quick Pre-Filtering
  - Instant blocking without LLM call
  - Saves API tokens and time
  - 98%+ confidence for obvious cases

✓ Enhanced LLM Prompt
  - Rich metadata sent to Gemma 2 9B
  - Contextual information included
  - Clear classification criteria
  - Higher accuracy than content-only
    """)

    print("=" * 80)
    print("✅ Enhanced Demo Complete!")
    print("=" * 80)

    classifier.close()


if __name__ == '__main__':
    main()
