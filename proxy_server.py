#!/usr/bin/env python3
"""
HTTP Proxy Server with ML-based Domain Classification

This proxy:
1. Intercepts HTTP requests
2. Checks domain against hybrid classifier
3. Blocks malicious domains
4. Forwards legitimate requests
5. Learns from new domains (content-based ML)
"""

import sys
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from hybrid_classifier import HybridDomainClassifier


class ProxyHandler(BaseHTTPRequestHandler):
    """HTTP proxy request handler with ML classification"""

    classifier = None  # Will be set by main()

    def do_GET(self):
        """Handle GET requests"""
        self._handle_request()

    def do_POST(self):
        """Handle POST requests"""
        self._handle_request()

    def do_CONNECT(self):
        """Handle CONNECT requests (HTTPS)"""
        # For simplicity, we'll just classify the domain
        # Full HTTPS proxying requires SSL certificate handling
        host = self.path.split(':')[0]
        self._classify_and_respond(host, is_connect=True)

    def _handle_request(self):
        """Handle HTTP request with classification"""
        # Parse URL
        parsed = urlparse(self.path)
        domain = parsed.netloc if parsed.netloc else parsed.path.split('/')[0]

        self._classify_and_respond(domain, is_connect=False)

    def _classify_and_respond(self, domain: str, is_connect: bool):
        """Classify domain and send response"""
        if not domain:
            self.send_error(400, "Bad Request")
            return

        print(f"\n[REQUEST] {domain}")

        # Classify domain
        try:
            prediction, confidence, method = self.classifier.predict(domain, fetch_if_unknown=True)

            label = "MALICIOUS" if prediction == 1 else "LEGITIMATE"
            print(f"  → {label} (confidence: {confidence:.2%}, method: {method})")

            if prediction == 1:
                # Block malicious domain
                self._send_blocked_response(domain, confidence)
            else:
                # Allow legitimate domain
                if is_connect:
                    self.send_response(200, 'Connection Established')
                    self.end_headers()
                else:
                    self._send_allowed_response(domain)

        except Exception as e:
            print(f"  → ERROR: {e}")
            self.send_error(500, f"Classification error: {e}")

    def _send_blocked_response(self, domain: str, confidence: float):
        """Send blocked page for malicious domains"""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Domain Blocked</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 600px;
            margin: 50px auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .blocked {{
            background: #fff;
            border-left: 4px solid #d32f2f;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #d32f2f; }}
        .confidence {{ color: #666; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="blocked">
        <h1>⚠️ Domain Blocked</h1>
        <p><strong>Domain:</strong> {domain}</p>
        <p><strong>Reason:</strong> Classified as malicious by ML-based analysis</p>
        <p class="confidence"><strong>Confidence:</strong> {confidence:.1%}</p>
        <hr>
        <p>This domain was blocked by the Hybrid Domain Classifier.</p>
        <p>The classification was based on content analysis and domain patterns.</p>
    </div>
</body>
</html>"""

        self.send_response(403)
        self.send_header('Content-Type', 'text/html')
        self.send_header('Content-Length', len(html))
        self.end_headers()
        self.wfile.write(html.encode())

    def _send_allowed_response(self, domain: str):
        """Send info page for allowed domains (in real proxy, forward request)"""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Domain Allowed</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 600px;
            margin: 50px auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .allowed {{
            background: #fff;
            border-left: 4px solid #4caf50;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #4caf50; }}
    </style>
</head>
<body>
    <div class="allowed">
        <h1>✓ Domain Allowed</h1>
        <p><strong>Domain:</strong> {domain}</p>
        <p><strong>Status:</strong> Classified as legitimate</p>
        <hr>
        <p>In a production proxy, this request would be forwarded to the target server.</p>
        <p>For this demo, we're showing this page instead.</p>
    </div>
</body>
</html>"""

        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.send_header('Content-Length', len(html))
        self.end_headers()
        self.wfile.write(html.encode())

    def log_message(self, format, *args):
        """Suppress default logging"""
        pass


def main():
    parser = argparse.ArgumentParser(
        description='HTTP Proxy Server with ML-based Domain Classification'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Proxy port (default: 8080)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/hybrid_classifier',
        help='Path to trained model'
    )
    parser.add_argument(
        '--no-fetch',
        action='store_true',
        help='Disable content fetching for unknown domains'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("HTTP PROXY SERVER - ML-BASED DOMAIN CLASSIFICATION")
    print("=" * 70)

    # Load classifier
    print(f"\nLoading hybrid classifier from {args.model}...")
    classifier = HybridDomainClassifier(
        enable_content_fetch=not args.no_fetch,
        cache_new_results=True
    )

    try:
        classifier.load(args.model)
        print("✓ Classifier loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading classifier: {e}")
        print("\nPlease train the model first:")
        print("  python train_hybrid_classifier.py")
        return

    # Set classifier for handler
    ProxyHandler.classifier = classifier

    # Start server
    server = HTTPServer(('0.0.0.0', args.port), ProxyHandler)

    print(f"\n✓ Proxy server started on port {args.port}")
    print("\nConfiguration:")
    print(f"  Content fetch: {'Enabled' if not args.no_fetch else 'Disabled'}")
    print(f"  Trie entries: {classifier.trie.total_entries}")
    print(f"  Content model: {'Loaded' if classifier.content_model else 'Not available'}")

    print("\n" + "=" * 70)
    print("SERVER READY")
    print("=" * 70)
    print(f"\nConfigure your browser to use proxy:")
    print(f"  Host: localhost")
    print(f"  Port: {args.port}")
    print(f"\nOr test with curl:")
    print(f"  curl -x http://localhost:{args.port} http://example.com")
    print("\nPress Ctrl+C to stop")
    print("=" * 70 + "\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down proxy server...")
        server.shutdown()

        # Show final statistics
        stats = classifier.get_statistics()
        print("\n" + "=" * 70)
        print("SESSION STATISTICS")
        print("=" * 70)
        print(f"  Total requests:   {stats['total_requests']}")
        print(f"  Trie hits:        {stats['trie_hits']}")
        print(f"  Content fetches:  {stats['content_fetches']}")
        print(f"  Cache updates:    {stats['cache_updates']}")
        print(f"  Cache hit rate:   {stats['cache_hit_rate']:.1%}")
        print("=" * 70)


if __name__ == '__main__':
    main()