#!/usr/bin/env python3
"""
URL Classification System
Implements asynchronous classification of URLs using LLM and trie caching.

Workflow:
1. Check URL against in-memory trie (microsecond-level lookup)
2. If not found, fetch and analyze content asynchronously
3. Classify using LLM API
4. Add to trie for future instant lookups
"""

import asyncio
import threading
from typing import Optional, Dict, Any, Tuple
from urllib.parse import urlparse
from queue import Queue
import time
import json
import re

from trie_structure import HierarchicalDomainTrie
from content_fetcher import ContentFetcher


class URLClassifier:
    """
    Asynchronous URL classifier with trie caching.

    User-facing requests are non-blocking - classification happens in background.
    """

    def __init__(self, trie: HierarchicalDomainTrie,
                 groq_api_key: Optional[str] = None,
                 llm_api_url: str = "https://api.groq.com/openai/v1/chat/completions",
                 model_name: str = "llama-3.1-8b-instant"):
        """
        Initialize the classifier.

        Args:
            trie: Pre-loaded hierarchical domain trie
            groq_api_key: Groq API key (get free at console.groq.com)
            llm_api_url: URL for the LLM API (default: Groq)
            model_name: Name of the LLM model (default: llama-3.1-8b-instant)
        """
        self.trie = trie
        self.fetcher = ContentFetcher()
        self.llm_api_url = llm_api_url
        self.model_name = model_name
        self.groq_api_key = groq_api_key

        # Track user's browsing context for contextual relevance
        self.user_context = {
            'current_page': None,
            'current_domain': None,
            'page_topic': None
        }

        # Background processing queue
        self.classification_queue = Queue()

        # Statistics
        self.stats = {
            'total_checks': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'classifications_pending': 0,
            'classifications_completed': 0,
            'ads_detected': 0,
            'legitimate_detected': 0
        }

        # Start background worker thread
        self.worker_thread = threading.Thread(target=self._background_worker, daemon=True)
        self.worker_thread.start()

        print("URLClassifier initialized - background worker started")

    def check_url(self, url: str, current_page_url: Optional[str] = None,
                  page_topic: Optional[str] = None) -> Dict[str, Any]:
        """
        Check if URL should be blocked (non-blocking, user-facing).

        This is the main entry point for user requests.
        Returns immediately with cached result or queues for classification.

        Args:
            url: Full URL to check
            current_page_url: The page the user is currently viewing (for context)
            page_topic: Topic/category of current page (e.g., "sports", "news", "shopping")

        Returns:
            Dictionary with:
                - action: 'block' or 'allow'
                - source: 'cache' or 'pending_classification'
                - confidence: float (0.0 to 1.0)
                - domain: extracted domain
        """
        self.stats['total_checks'] += 1

        # Update user context
        if current_page_url:
            self.user_context['current_page'] = current_page_url
            self.user_context['current_domain'] = self._extract_domain(current_page_url)
        if page_topic:
            self.user_context['page_topic'] = page_topic

        # Extract domain from URL
        domain = self._extract_domain(url)

        # Generate domain signature for trie lookup
        signature = self._create_signature(domain)

        # Check trie (microsecond-level lookup)
        prediction, confidence, match_level = self.trie.lookup(signature)

        if prediction is not None:
            # Cache hit - immediate response
            self.stats['cache_hits'] += 1

            action = 'block' if prediction == 1 else 'allow'

            return {
                'action': action,
                'source': 'cache',
                'confidence': confidence,
                'match_level': match_level,
                'domain': domain,
                'signature': signature
            }

        else:
            # Cache miss - queue for background classification
            self.stats['cache_misses'] += 1
            self.stats['classifications_pending'] += 1

            # Add to background processing queue with context
            self.classification_queue.put({
                'url': url,
                'domain': domain,
                'signature': signature,
                'timestamp': time.time(),
                'context': dict(self.user_context)  # Copy current context
            })

            # Return default action (allow) while classification happens in background
            return {
                'action': 'allow',  # Default to allow, will be cached if it's an ad
                'source': 'pending_classification',
                'confidence': 0.0,
                'domain': domain,
                'signature': signature
            }

    def _background_worker(self):
        """
        Background worker that processes classification queue.
        Runs continuously in a separate thread.
        """
        print("Background classification worker started")

        while True:
            try:
                # Get next item from queue (blocking)
                item = self.classification_queue.get(timeout=1)

                # Process classification
                self._classify_and_cache(item)

                # Mark task as done
                self.classification_queue.task_done()

            except:
                # Queue empty or timeout - continue waiting
                continue

    def _classify_and_cache(self, item: Dict[str, Any]):
        """
        Fetch content, classify with LLM, and add to trie.

        Args:
            item: Dictionary with url, domain, signature, context
        """
        url = item['url']
        domain = item['domain']
        signature = item['signature']
        context = item.get('context', {})

        print(f"\n[Background] Classifying: {domain}")

        # Step 1: Fetch content WITH metadata
        fetch_result = self.fetcher.fetch_with_metadata(url)

        if not fetch_result['content'] and fetch_result['error']:
            print(f"[Background] Failed to fetch {domain}: {fetch_result['error']}")
            self.stats['classifications_pending'] -= 1
            return

        # Step 1.5: Pre-filter obvious tracking pixels/ads
        quick_decision = self._quick_filter(fetch_result, domain)
        if quick_decision is not None:
            is_ad, confidence, reason = quick_decision
            print(f"[Background] ⚡ Quick filter: {reason}")
        else:
            # Step 2: Classify with LLM using full metadata
            is_ad, confidence = self._classify_with_llm(domain, fetch_result, context)

        # Step 3: Add to trie for future lookups
        prediction = 1 if is_ad else 0

        metadata = {
            'classified_at': time.time(),
            'source': 'llm_classification',
            'domain': domain
        }

        self.trie.insert(signature, prediction, confidence, metadata)

        # Update statistics
        self.stats['classifications_pending'] -= 1
        self.stats['classifications_completed'] += 1

        if is_ad:
            self.stats['ads_detected'] += 1
            print(f"[Background] ✓ Classified as AD: {domain} (confidence: {confidence:.2f})")
        else:
            self.stats['legitimate_detected'] += 1
            print(f"[Background] ✓ Classified as LEGITIMATE: {domain} (confidence: {confidence:.2f})")

    def _quick_filter(self, fetch_result: Dict[str, Any], domain: str) -> Optional[Tuple[bool, float, str]]:
        """
        Quick pre-filtering for obvious ads/tracking before calling LLM.
        Returns (is_ad, confidence, reason) or None if needs LLM classification.

        Args:
            fetch_result: Result from fetch_with_metadata()
            domain: Domain name

        Returns:
            (is_ad, confidence, reason) or None
        """
        headers = fetch_result.get('headers', {})
        content = fetch_result.get('content', '')
        status_code = fetch_result.get('status_code')
        final_url = fetch_result.get('final_url', '')

        # 1. Check for tracking pixels (tiny images)
        content_type = headers.get('Content-Type', '').lower()
        content_length = int(headers.get('Content-Length', len(content) if content else 0))

        if 'image/' in content_type and content_length < 1000:
            return (True, 0.98, f"Tracking pixel detected (image, {content_length} bytes)")

        # 2. Check for common ad server headers
        server = headers.get('Server', '').lower()
        x_served_by = headers.get('X-Served-By', '').lower()

        ad_servers = ['doubleclick', 'adserver', 'googlesyndication', 'advertising']
        for ad_server in ad_servers:
            if ad_server in server or ad_server in x_served_by:
                return (True, 0.95, f"Ad server detected in headers: {ad_server}")

        # 3. Check for javascript/tracking scripts with suspicious names
        if content_type == 'application/javascript' or content_type == 'text/javascript':
            script_indicators = ['analytics', 'tracking', 'pixel', 'adserver', 'beacon']
            for indicator in script_indicators:
                if indicator in domain.lower() or (content and indicator in content[:500].lower()):
                    return (True, 0.90, f"Tracking script detected: {indicator}")

        # 4. Check for redirect chains (common in ad networks)
        if final_url and final_url != fetch_result.get('url'):
            redirect_count = len(final_url.split('/')) - len(fetch_result.get('url', '').split('/'))
            if redirect_count > 2:
                return (True, 0.85, f"Multiple redirects detected (ad chain)")

        # 5. Check for empty or minimal content (often tracking endpoints)
        if content and len(content.strip()) < 100 and status_code == 200:
            return (True, 0.80, "Minimal content (likely tracking endpoint)")

        # No quick decision - needs LLM
        return None

    def _classify_with_llm(self, domain: str, fetch_result: Dict[str, Any],
                          context: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Classify content using Groq API with Gemma 2 9B.
        Uses network metadata + contextual relevance for accurate classification.

        Args:
            domain: Domain name
            fetch_result: Complete fetch result with headers, content, etc.
            context: User browsing context (current page, topic)

        Returns:
            (is_advertisement, confidence_score)
        """
        # Check if API key is configured
        if not self.groq_api_key:
            print(f"[Background] No Groq API key configured, using heuristics")
            content = fetch_result.get('content', '')
            return self._heuristic_classification(domain, content)

        # Extract data from fetch result
        content = fetch_result.get('content', '')
        headers = fetch_result.get('headers', {})
        status_code = fetch_result.get('status_code')
        final_url = fetch_result.get('final_url', '')
        fetch_time = fetch_result.get('fetch_time', 0)

        # Truncate content for LLM (800 chars to save tokens)
        truncated_content = content[:800] if content and len(content) > 800 else (content or 'No content available')

        # Extract relevant headers
        content_type = headers.get('Content-Type', 'unknown')
        server = headers.get('Server', 'unknown')
        content_length = headers.get('Content-Length', 'unknown')
        cache_control = headers.get('Cache-Control', 'none')
        set_cookie = 'Yes (tracking cookies)' if headers.get('Set-Cookie') else 'No'

        # Build context information
        context_info = ""
        if context.get('current_domain'):
            context_info = f"\nUser is currently browsing: {context['current_domain']}"
            if context.get('page_topic'):
                context_info += f" (Topic: {context['page_topic']})"

        # Detect redirect
        redirected = "Yes" if final_url and final_url != fetch_result.get('url') else "No"

        # Create enhanced system message
        system_message = """You are an expert cybersecurity analyst specializing in identifying advertisements, tracking domains, and potentially malicious content.

Your task is to classify domains based on:
1. Network metadata (headers, response characteristics)
2. Content analysis (HTML, scripts, text)
3. Contextual relevance (does this content match what the user is viewing?)

Key principles:
- Tracking pixels, analytics scripts, and ad servers are ADVERTISEMENTS
- Content unrelated to the user's current page topic is likely an ADVERTISEMENT
- Legitimate websites have proper HTML structure and relevant content
- Cross-site tracking mechanisms indicate ADVERTISEMENT"""

        user_prompt = f"""Analyze this resource and classify it as ADVERTISEMENT or LEGITIMATE.

=== DOMAIN INFORMATION ===
Domain: {domain}
Final URL: {final_url}
Redirected: {redirected}

=== NETWORK METADATA ===
HTTP Status: {status_code}
Content-Type: {content_type}
Server: {server}
Content-Length: {content_length}
Cache-Control: {cache_control}
Sets Cookies: {set_cookie}
Response Time: {fetch_time:.2f}s
{context_info}

=== CONTENT PREVIEW (first 800 chars) ===
{truncated_content}

=== CLASSIFICATION CRITERIA ===
Consider these factors:

1. **Domain Analysis**:
   - Does the domain contain ad-related keywords (ads, track, analytics, doubleclick, pixel, banner, syndication)?
   - Is this a known ad network or tracking service?

2. **Network Signals**:
   - Small content size + image type = likely tracking pixel
   - JavaScript with tracking/analytics keywords = likely ad script
   - Multiple redirects = common in ad chains
   - Tracking cookies being set = advertising/tracking purpose

3. **Content Analysis**:
   - Does content suggest advertising, tracking, or analytics services?
   - Is there minimal/no meaningful content? (tracking endpoints)
   - Are there ad-serving scripts or iframes?

4. **Contextual Relevance** (IMPORTANT):
   - If user is on an NBA/sports website, content about dating, gambling, weight loss, or adult products is likely an ADVERTISEMENT
   - If user is on a news site, unrelated product promotions are ADVERTISEMENTS
   - Content should be topically relevant to the page the user is viewing

5. **Legitimate Signals**:
   - Proper HTML structure with meaningful content
   - Content matches the domain purpose
   - No tracking/analytics keywords in domain or content
   - Contextually relevant to user's browsing

=== YOUR RESPONSE ===
Respond with ONLY ONE WORD:
- "ADVERTISEMENT" if this is an ad, tracking domain, or contextually irrelevant content
- "LEGITIMATE" if this is a genuine website with relevant, appropriate content

Classification:"""

        try:
            import requests

            # Groq API request (OpenAI-compatible format)
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.2,  # Lower for more consistent classification
                "max_tokens": 20,     # Only need one word response
                "top_p": 0.95,
                "stream": False
            }

            response = requests.post(
                self.llm_api_url,
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                llm_response = result['choices'][0]['message']['content'].strip().upper()

                # Parse response
                if 'ADVERTISEMENT' in llm_response:
                    return (True, 0.92)  # High confidence with enhanced prompt
                elif 'LEGITIMATE' in llm_response:
                    return (False, 0.92)
                else:
                    # Unclear response, use heuristics
                    print(f"[Background] Unclear LLM response: {llm_response}, using heuristics")
                    return self._heuristic_classification(domain, content or '')

            elif response.status_code == 429:
                # Rate limit hit
                print(f"[Background] Rate limit reached, using heuristics")
                return self._heuristic_classification(domain, content)

            else:
                # API error, fallback to heuristics
                print(f"[Background] Groq API error: {response.status_code} - {response.text[:200]}")
                return self._heuristic_classification(domain, content or '')

        except Exception as e:
            print(f"[Background] LLM classification error: {e}")
            # Fallback to heuristic classification
            return self._heuristic_classification(domain, content or '')

    def _heuristic_classification(self, domain: str, content: str) -> Tuple[bool, float]:
        """
        Fallback heuristic-based classification.
        Used when LLM is unavailable.

        Args:
            domain: Domain name
            content: HTML content

        Returns:
            (is_advertisement, confidence_score)
        """
        score = 0
        indicators = []

        # Check domain for ad-related keywords
        ad_keywords = ['ad', 'ads', 'track', 'analytics', 'pixel', 'banner',
                      'doubleclick', 'adserver', 'tracking', 'metrics']

        domain_lower = domain.lower()
        for keyword in ad_keywords:
            if keyword in domain_lower:
                score += 3
                indicators.append(f"domain_keyword:{keyword}")

        # Check for suspicious TLDs
        suspicious_tlds = ['.xyz', '.top', '.click', '.link', '.bid']
        for tld in suspicious_tlds:
            if domain.endswith(tld):
                score += 2
                indicators.append(f"suspicious_tld:{tld}")

        # Check content for ad indicators
        content_lower = content.lower()
        ad_content_keywords = ['advertisement', 'sponsored', 'tracking pixel',
                              'google analytics', 'facebook pixel']

        for keyword in ad_content_keywords:
            if keyword in content_lower:
                score += 1
                indicators.append(f"content_keyword:{keyword}")

        # Classification threshold
        is_ad = score >= 3
        confidence = min(0.7, score * 0.15)  # Lower confidence for heuristics

        if indicators:
            print(f"[Background] Heuristic indicators: {', '.join(indicators)}")

        return (is_ad, confidence)

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        if not url.startswith(('http://', 'https://')):
            url = f'http://{url}'

        parsed = urlparse(url)
        return parsed.netloc or parsed.path

    def _create_signature(self, domain: str) -> Tuple[str, str, str]:
        """
        Create hierarchical signature for domain.

        Returns:
            (tld, domain_pattern, subdomain_pattern)
        """
        parts = domain.split('.')

        # Extract TLD
        if len(parts) >= 2:
            tld = parts[-1]
        else:
            tld = 'unknown'

        # Analyze domain pattern
        if len(parts) >= 2:
            domain_name = parts[-2]

            # Check for high digits
            digit_ratio = sum(c.isdigit() for c in domain_name) / max(len(domain_name), 1)
            if digit_ratio > 0.3:
                domain_pattern = 'high_digits'
            elif len(domain_name) > 20:
                domain_pattern = 'very_long'
            else:
                domain_pattern = 'normal'
        else:
            domain_pattern = 'unknown'

        # Analyze subdomain pattern
        if len(parts) > 2:
            if parts[0] == 'www' and len(parts) == 3:
                subdomain_pattern = 'www_only'
            elif len(parts) > 3:
                subdomain_pattern = 'deep'
            else:
                subdomain_pattern = 'single'
        else:
            subdomain_pattern = 'none'

        return (tld, domain_pattern, subdomain_pattern)

    def get_statistics(self) -> Dict[str, Any]:
        """Get classification statistics."""
        cache_hit_rate = (self.stats['cache_hits'] / max(self.stats['total_checks'], 1)) * 100

        return {
            **self.stats,
            'cache_hit_rate': f"{cache_hit_rate:.2f}%",
            'trie_stats': self.trie.get_statistics(),
            'queue_size': self.classification_queue.qsize()
        }

    def wait_for_pending(self, timeout: int = 60):
        """
        Wait for all pending classifications to complete.
        Useful for testing or shutdown.

        Args:
            timeout: Maximum seconds to wait
        """
        print(f"\nWaiting for {self.stats['classifications_pending']} pending classifications...")
        self.classification_queue.join()
        print("All classifications completed!")

    def save_trie(self, filepath: str):
        """Save the updated trie to disk."""
        self.trie.save(filepath)

    def close(self):
        """Clean up resources."""
        self.fetcher.close()


# Example usage and testing
if __name__ == '__main__':
    print("=" * 60)
    print("URL Classifier - Asynchronous Classification Demo")
    print("=" * 60)

    # Initialize trie (load existing or create new)
    trie = HierarchicalDomainTrie()

    # Pre-populate with some known ad domains for demo
    known_ads = [
        (('com', 'normal', 'none'), 'doubleclick.com'),
        (('net', 'normal', 'none'), 'adserver.net'),
        (('xyz', 'high_digits', 'deep'), 'ad123xyz.xyz'),
    ]

    print("\nPre-populating trie with known ad domains...")
    for sig, domain in known_ads:
        trie.insert(sig, prediction=1, confidence=1.0,
                   metadata={'source': 'known_ad', 'domain': domain})

    # Initialize classifier (without API key for demo - will use heuristics)
    # To use Groq API, get your free API key at: https://console.groq.com
    # classifier = URLClassifier(trie, groq_api_key="your_groq_api_key_here")
    classifier = URLClassifier(trie, groq_api_key=None)

    # Test URLs
    test_urls = [
        'http://doubleclick.com',  # Known ad (cache hit)
        'http://example.com',       # Unknown (will classify in background)
        'http://google.com',        # Unknown (will classify in background)
        'http://ad123xyz.xyz',      # Known ad (cache hit)
    ]

    print("\n" + "=" * 60)
    print("Testing URL Classification (User-Facing)")
    print("=" * 60)

    # Simulate user requests (non-blocking)
    for url in test_urls:
        result = classifier.check_url(url)

        print(f"\nURL: {url}")
        print(f"  Action: {result['action'].upper()}")
        print(f"  Source: {result['source']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Domain: {result['domain']}")

        # Small delay to simulate real usage
        time.sleep(0.1)

    # Show statistics
    print("\n" + "=" * 60)
    print("Statistics (Immediate)")
    print("=" * 60)
    stats = classifier.get_statistics()
    for key, value in stats.items():
        if key != 'trie_stats':
            print(f"  {key}: {value}")

    # Wait for background classifications to complete
    print("\n" + "=" * 60)
    print("Background Classification In Progress...")
    print("=" * 60)
    print("(In production, this happens asynchronously without blocking users)")

    classifier.wait_for_pending(timeout=120)

    # Show final statistics
    print("\n" + "=" * 60)
    print("Final Statistics")
    print("=" * 60)
    stats = classifier.get_statistics()
    for key, value in stats.items():
        if key != 'trie_stats':
            print(f"  {key}: {value}")

    print("\nTrie Statistics:")
    for key, value in stats['trie_stats'].items():
        print(f"  {key}: {value}")

    # Save updated trie
    print("\n" + "=" * 60)
    output_file = 'data/updated_trie.json'
    print(f"Saving updated trie to: {output_file}")
    classifier.save_trie(output_file)

    classifier.close()
    print("\n✓ Demo completed!")
