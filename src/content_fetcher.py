#!/usr/bin/env python3
"""
Content Fetcher
Fetches webpage content for unknown domains.
Acts as HTTP client to retrieve HTML for analysis.
"""

import requests
import time
from typing import Optional, Dict, Any
from urllib.parse import urlparse
import warnings

# Suppress SSL warnings for simplicity
warnings.filterwarnings('ignore', message='Unverified HTTPS request')


class ContentFetcher:
    """
    Fetches webpage content for ML-based classification.
    """

    def __init__(self, timeout: int = 10, max_retries: int = 2):
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })

    def fetch_content(self, url: str) -> Optional[str]:
        """
        Fetch HTML content from URL.

        Args:
            url: Full URL to fetch (e.g., http://example.com or https://example.com)

        Returns:
            HTML content as string, or None if fetch failed
        """
        # Ensure URL has scheme
        if not url.startswith(('http://', 'https://')):
            url = f'http://{url}'

        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    url,
                    timeout=self.timeout,
                    verify=False,  # Skip SSL verification for simplicity
                    allow_redirects=True,
                    stream=False
                )

                # Check if response is HTML
                content_type = response.headers.get('Content-Type', '')
                if 'text/html' not in content_type.lower():
                    return None

                # Get content (limited to 1MB to prevent memory issues)
                if len(response.content) > 1_000_000:
                    return response.text[:1_000_000]

                return response.text

            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                    continue
                return None

            except requests.exceptions.ConnectionError:
                return None

            except requests.exceptions.TooManyRedirects:
                return None

            except Exception as e:
                return None

        return None

    def fetch_with_metadata(self, url: str) -> Dict[str, Any]:
        """
        Fetch content with additional metadata.

        Returns:
            Dictionary with content, status_code, headers, etc.
        """
        if not url.startswith(('http://', 'https://')):
            url = f'http://{url}'

        result = {
            'url': url,
            'content': None,
            'status_code': None,
            'headers': {},
            'final_url': None,
            'fetch_time': 0,
            'error': None
        }

        start_time = time.time()

        try:
            response = self.session.get(
                url,
                timeout=self.timeout,
                verify=False,
                allow_redirects=True
            )

            result['status_code'] = response.status_code
            result['headers'] = dict(response.headers)
            result['final_url'] = response.url

            # Check content type
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' in content_type.lower():
                if len(response.content) > 1_000_000:
                    result['content'] = response.text[:1_000_000]
                else:
                    result['content'] = response.text

        except requests.exceptions.Timeout:
            result['error'] = 'timeout'
        except requests.exceptions.ConnectionError:
            result['error'] = 'connection_error'
        except requests.exceptions.TooManyRedirects:
            result['error'] = 'too_many_redirects'
        except Exception as e:
            result['error'] = str(e)

        result['fetch_time'] = time.time() - start_time

        return result

    def fetch_domain(self, domain: str) -> Optional[str]:
        """
        Fetch content from a domain (tries both http and https).

        Args:
            domain: Domain name (e.g., example.com)

        Returns:
            HTML content or None
        """
        # Try HTTPS first (more common now)
        content = self.fetch_content(f'https://{domain}')
        if content:
            return content

        # Fallback to HTTP
        content = self.fetch_content(f'http://{domain}')
        return content

    def close(self):
        """Close the session"""
        self.session.close()


if __name__ == '__main__':
    print("Content Fetcher - Testing")
    fetcher = ContentFetcher()

    # Test with a known domain
    test_domain = 'example.com'
    print(f"\nFetching content from {test_domain}...")
    content = fetcher.fetch_domain(test_domain)

    if content:
        print(f"✓ Successfully fetched {len(content)} characters")
        print(f"Preview: {content[:200]}...")
    else:
        print("✗ Failed to fetch content")

    fetcher.close()