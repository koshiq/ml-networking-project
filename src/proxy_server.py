#!/usr/bin/env python3
"""
Real-time Network Traffic Monitoring Proxy
Intercepts HTTP/HTTPS requests and classifies URLs in real-time using URLClassifier.
"""

import sys
import os
from mitmproxy import http
from mitmproxy.tools.main import mitmdump
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from trie_structure import HierarchicalDomainTrie
from url_classifier import URLClassifier
from config_loader import config


class AdBlockerProxy:
    """
    mitmproxy addon that intercepts traffic and classifies URLs in real-time.

    Workflow:
    1. Intercept every HTTP/HTTPS request
    2. Check URL against trie
    3. If blocked → Return 204 No Content (blocked)
    4. If allowed → Let request through
    5. Background worker classifies new URLs and updates trie
    """

    def __init__(self):
        print("\n" + "=" * 70)
        print("Ad Blocker Proxy - Real-time Network Monitor")
        print("=" * 70)

        # Initialize trie (load existing or create new)
        trie_path = 'data/proxy_trie.json'
        self.trie = HierarchicalDomainTrie()

        if os.path.exists(trie_path):
            print(f"\n✓ Loading existing trie from {trie_path}")
            self.trie.load(trie_path)
            stats = self.trie.get_statistics()
            print(f"  Loaded {stats['total_entries']} classified domains")
        else:
            print(f"\n✓ Starting with empty trie (will learn from traffic)")

        # Initialize classifier
        print("\n✓ Initializing URL classifier...")
        if config.validate():
            print(f"  Using Groq API with {config.model_name}")
            self.classifier = URLClassifier(
                self.trie,
                groq_api_key=config.groq_api_key,
                model_name=config.model_name
            )
        else:
            print("  Using heuristics only (no API key)")
            self.classifier = URLClassifier(self.trie, groq_api_key=None)

        # Statistics
        self.stats = {
            'total_requests': 0,
            'blocked': 0,
            'allowed': 0,
            'cache_hits': 0,
            'pending_classification': 0
        }

        print("\n" + "=" * 70)
        print("Proxy is ready! Configure your browser to use:")
        print("  HTTP Proxy:  localhost:8080")
        print("  HTTPS Proxy: localhost:8080")
        print("=" * 70)
        print("\nMonitoring traffic... (Press Ctrl+C to stop and save)\n")

    def request(self, flow: http.HTTPFlow) -> None:
        """
        Intercept every HTTP/HTTPS request.
        Called by mitmproxy for each request.
        """
        self.stats['total_requests'] += 1

        # Extract the actual destination host (not localhost)
        host = flow.request.host
        port = flow.request.port
        scheme = flow.request.scheme
        path = flow.request.path

        # Construct the actual URL being requested
        if port in (80, 443):
            # Standard ports - don't include in URL
            url = f"{scheme}://{host}{path}"
        else:
            # Non-standard port - include it
            url = f"{scheme}://{host}:{port}{path}"

        # Skip proxy's own localhost traffic
        if host in ('localhost', '127.0.0.1') or host.endswith('.local'):
            return

        # Check if URL should be blocked
        result = self.classifier.check_url(url)

        if result['action'] == 'block':
            # BLOCK: Return empty response
            self.stats['blocked'] += 1

            if result['source'] == 'cache':
                self.stats['cache_hits'] += 1
                print(f"🚫 BLOCKED (cache): {host}")
            else:
                print(f"🚫 BLOCKED: {host}")

            # Return 204 No Content (ad blocked)
            flow.response = http.Response.make(
                204,  # No Content
                b"",
                {"Content-Type": "text/html"}
            )

        else:
            # ALLOW: Let request through
            self.stats['allowed'] += 1

            if result['source'] == 'cache':
                self.stats['cache_hits'] += 1
                print(f"✅ ALLOWED (cache): {host}")
            else:
                self.stats['pending_classification'] += 1
                print(f"✅ ALLOWED (pending): {host} - classifying in background...")

    def response(self, flow: http.HTTPFlow) -> None:
        """
        Called after response is received.
        We can analyze response headers/content here for better classification.
        """
        # Optional: Could extract more metadata from response
        pass

    def done(self):
        """
        Called when proxy is shutting down.
        Save the trie before exit.
        """
        print("\n\n" + "=" * 70)
        print("Shutting down proxy...")
        print("=" * 70)

        # Wait for pending classifications
        pending = self.stats['pending_classification']
        if pending > 0:
            print(f"\nWaiting for {pending} pending classifications to complete...")
            self.classifier.wait_for_pending(timeout=30)

        # Save trie
        trie_path = 'data/proxy_trie.json'
        os.makedirs('data', exist_ok=True)
        self.classifier.save_trie(trie_path)

        # Print final statistics
        print("\n" + "=" * 70)
        print("Session Statistics")
        print("=" * 70)
        print(f"Total requests:        {self.stats['total_requests']}")
        print(f"Blocked:               {self.stats['blocked']}")
        print(f"Allowed:               {self.stats['allowed']}")
        print(f"Cache hits:            {self.stats['cache_hits']}")
        print(f"Pending (classified):  {self.stats['pending_classification']}")

        # Trie statistics
        print("\n" + "=" * 70)
        print("Trie Statistics")
        print("=" * 70)
        trie_stats = self.trie.get_statistics()
        print(f"Total domains learned: {trie_stats['total_entries']}")
        print(f"TLD nodes:             {trie_stats['levels']['tld_nodes']}")
        print(f"Saved to:              {trie_path}")

        print("\n✓ Trie saved successfully!")
        print("=" * 70 + "\n")

        # Cleanup
        self.classifier.close()


# mitmproxy addon entry point
addons = [AdBlockerProxy()]


def main():
    """
    Start the proxy server.
    This is called when running: python src/proxy_server.py
    """
    import sys

    # mitmproxy arguments
    sys.argv = [
        'mitmdump',
        '--listen-host', '127.0.0.1',
        '--listen-port', '8080',
        '--ssl-insecure',  # Allow self-signed certs for testing
        '-s', __file__,     # Load this script as addon
    ]

    # Start mitmdump
    mitmdump()


if __name__ == '__main__':
    print("Starting Ad Blocker Proxy Server...")
    print("Install mitmproxy first: uv pip install mitmproxy")
    print()
    main()
