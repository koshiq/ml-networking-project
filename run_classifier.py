#!/usr/bin/env python3
"""
Unified Hybrid Domain Classifier Tool

Modes:
  demo       - Quick demonstration
  monitor    - Interactive domain testing with performance metrics
  benchmark  - Performance testing
  proxy      - HTTP proxy server
"""

import sys
import os
import time
import statistics
import argparse
from typing import List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from hybrid_classifier import HybridDomainClassifier


# ============================================================================
# DEMO MODE
# ============================================================================

def run_demo():
    """Quick demonstration of the classifier"""
    print("\n" + "=" * 80)
    print(" " * 25 + "HYBRID CLASSIFIER DEMO")
    print("=" * 80)

    classifier = HybridDomainClassifier(enable_content_fetch=False)

    try:
        classifier.load('models/hybrid_classifier')
        print("✓ Model loaded\n")
    except Exception as e:
        print(f"✗ Error: {e}\nRun: python train_hybrid_classifier.py")
        return

    # Test domains
    test_domains = [
        ('google.com', 'Legitimate'),
        ('ads.doubleclick.net', 'Ad network'),
        ('github.com', 'Legitimate'),
        ('googlesyndication.com', 'Ad server'),
    ]

    for domain, desc in test_domains:
        pred, conf, method = classifier.predict(domain, fetch_if_unknown=False)
        label = "MALICIOUS" if pred == 1 else "LEGITIMATE"
        print(f"{domain:30s} → {label:10s} ({conf:6.2%}, {method})")

    print("\n" + "=" * 80)


# ============================================================================
# MONITOR MODE
# ============================================================================

def run_monitor():
    """Interactive monitoring with performance metrics"""
    print("=" * 80)
    print(" " * 20 + "LIVE DOMAIN MONITOR - PERFORMANCE MODE")
    print("=" * 80)
    print()

    classifier = HybridDomainClassifier(enable_content_fetch=True, cache_new_results=True)

    try:
        classifier.load('models/hybrid_classifier')
        stats = classifier.get_statistics()
        print(f"✓ Loaded:")
        print(f"  - Rule-based list: {stats['rule_entries']:,} domains")
        print(f"  - Trie cache: {classifier.trie.total_entries:,} entries\n")
    except Exception as e:
        print(f"✗ Error: {e}")
        return

    print("Three-Tier Classification:")
    print("  1. Rule-based lookup (198K labeled domains)")
    print("  2. Trie cache (previous ML predictions)")
    print("  3. Content ML (for unknowns)\n")
    print("Type domains to classify (or 'quit' to exit)")
    print("Examples: google.com, ads.doubleclick.net\n")

    request_count = 0
    total_time = 0

    while True:
        try:
            domain = input("> ").strip()

            if not domain or domain.lower() in ['quit', 'exit', 'q']:
                break

            # Clean domain
            domain = domain.replace('http://', '').replace('https://', '').split('/')[0]

            # Classify with timing
            start = time.perf_counter()
            pred, conf, method = classifier.predict(domain, fetch_if_unknown=True)
            elapsed_ms = (time.perf_counter() - start) * 1000

            request_count += 1
            total_time += elapsed_ms

            # Display result
            label = "🔴 MALICIOUS" if pred == 1 else "🟢 LEGITIMATE"
            perf = "⚡" if elapsed_ms < 1 else ("🚀" if elapsed_ms < 10 else "🐌")

            # Determine tier description
            tier_desc = {
                'rules': 'Tier 1: Rule-based',
                'trie': 'Tier 2: Trie cache',
                'content_ml': 'Tier 3: Content ML'
            }.get(method, method)

            print(f"\n  {label}")
            print(f"  Confidence: {conf:.2%}")
            print(f"  Method: {tier_desc}")
            print(f"  {perf} Time: {elapsed_ms:.4f} ms")

            # Running stats
            stats = classifier.get_statistics()
            avg = total_time / request_count
            print(f"\n  📊 Session: {request_count} reqs | Avg: {avg:.2f} ms")
            print(f"     Rules: {stats['rule_hits']} | Trie: {stats['trie_hits']} | "
                  f"ML: {stats['content_fetches']}\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"⚠️  Error: {e}\n")

    # Summary
    if request_count > 0:
        print(f"\n{'=' * 80}")
        print(f"  Total: {request_count} requests")
        print(f"  Average time: {total_time/request_count:.4f} ms")
        print(f"{'=' * 80}\n")


# ============================================================================
# BENCHMARK MODE
# ============================================================================

def run_benchmark(num_tests=1000):
    """Performance benchmarking"""
    print("=" * 80)
    print(" " * 25 + "PERFORMANCE BENCHMARK")
    print("=" * 80)
    print()

    classifier = HybridDomainClassifier(enable_content_fetch=False, cache_new_results=False)

    try:
        classifier.load('models/hybrid_classifier')
    except Exception as e:
        print(f"✗ Error: {e}")
        return

    # Test domains (mix)
    domains = [
        'google.com', 'facebook.com', 'youtube.com', 'amazon.com',
        'doubleclick.net', 'googlesyndication.com', 'adnxs.com',
        'twitter.com', 'github.com', 'linkedin.com'
    ]

    print(f"Running {num_tests} classifications...\n")

    timings = []
    start_total = time.perf_counter()

    for i in range(num_tests):
        domain = domains[i % len(domains)]
        start = time.perf_counter()
        pred, conf, method = classifier.predict(domain, fetch_if_unknown=False)
        elapsed = (time.perf_counter() - start) * 1000
        timings.append(elapsed)

    total_time = time.perf_counter() - start_total

    # Results
    print("=" * 80)
    print("RESULTS:")
    print("=" * 80)
    print(f"  Total requests:    {num_tests:,}")
    print(f"  Total time:        {total_time:.4f} seconds")
    print(f"  Average time:      {statistics.mean(timings):.4f} ms")
    print(f"  Median time:       {statistics.median(timings):.4f} ms")
    print(f"  Min time:          {min(timings):.4f} ms")
    print(f"  Max time:          {max(timings):.4f} ms")
    print(f"  Throughput:        {num_tests/total_time:,.0f} requests/second")
    print("=" * 80)
    print(f"\n⚡ Average response time: {statistics.mean(timings):.4f} ms")
    print(f"🚀 That's ~{1000/statistics.mean(timings):.0f}x faster than typical web requests!\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Hybrid Domain Classifier - Unified Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_classifier.py demo          # Quick demo
  python run_classifier.py monitor       # Interactive testing
  python run_classifier.py benchmark     # Performance test
  python run_classifier.py benchmark -n 5000  # 5000 iterations
        """
    )

    parser.add_argument(
        'mode',
        choices=['demo', 'monitor', 'benchmark'],
        help='Operation mode'
    )

    parser.add_argument(
        '-n', '--num-tests',
        type=int,
        default=1000,
        help='Number of tests for benchmark mode (default: 1000)'
    )

    args = parser.parse_args()

    if args.mode == 'demo':
        run_demo()
    elif args.mode == 'monitor':
        run_monitor()
    elif args.mode == 'benchmark':
        run_benchmark(args.num_tests)


if __name__ == '__main__':
    main()
