#!/usr/bin/env python3
"""
Content-Based Feature Extractor
Extracts features from HTML/webpage content for ML classification.

This is where ML is truly justified - analyzing complex content patterns
that simple rules cannot capture.
"""

import re
from typing import Dict, Any, Optional
from urllib.parse import urlparse
from collections import Counter
import math


class ContentFeatureExtractor:
    """
    Extracts features from webpage content (HTML, JavaScript, etc.)
    to determine if a site is ad-related or legitimate.
    """

    # Known ad networks and tracking domains
    AD_NETWORKS = {
        'doubleclick.net', 'googlesyndication.com', 'google-analytics.com',
        'googletagmanager.com', 'adnxs.com', 'adsafeprotected.com',
        'scorecardresearch.com', 'advertising.com', 'facebook.net',
        'taboola.com', 'outbrain.com', 'criteo.com', 'pubmatic.com',
        'openx.net', 'rubiconproject.com', 'indexww.com', 'amazon-adsystem.com',
        'adform.net', 'addthis.com', 'sharethrough.com', 'moatads.com'
    }

    # Ad-related keywords in scripts and HTML
    AD_KEYWORDS = [
        'advertisement', 'adsbygoogle', 'adsense', 'adserver', 'banner',
        'sponsored', 'popup', 'popunder', 'tracking', 'analytics',
        'clickthrough', 'impression', 'conversion', 'retargeting',
        'affiliate', 'monetize', 'ad-slot', 'ad-container'
    ]

    # Suspicious JavaScript patterns
    SUSPICIOUS_JS_PATTERNS = [
        r'window\.open\s*\(',  # Popup windows
        r'document\.write\s*\(',  # Dynamic content injection
        r'eval\s*\(',  # Code obfuscation
        r'setTimeout.*redirect',  # Delayed redirects
        r'location\.href\s*=',  # Redirects
        r'location\.replace',  # Forced navigation
        r'fromCharCode',  # String obfuscation
        r'unescape\s*\(',  # Decoding obfuscated code
    ]

    def extract_features(self, html_content: str, url: str) -> Dict[str, Any]:
        """
        Extract all content-based features from HTML.

        Args:
            html_content: Raw HTML content
            url: Full URL of the page

        Returns:
            Dictionary of extracted features
        """
        if not html_content:
            return self._get_empty_features()

        html_lower = html_content.lower()

        features = {
            # Content structure features
            'html_length': len(html_content),
            'num_scripts': self._count_tags(html_content, 'script'),
            'num_iframes': self._count_tags(html_content, 'iframe'),
            'num_images': self._count_tags(html_content, 'img'),
            'num_links': self._count_tags(html_content, 'a'),
            'num_divs': self._count_tags(html_content, 'div'),
            'num_forms': self._count_tags(html_content, 'form'),

            # Ad network detection
            'has_ad_networks': self._detect_ad_networks(html_lower),
            'num_ad_network_domains': self._count_ad_network_domains(html_lower),
            'ad_network_score': self._calculate_ad_network_score(html_lower),

            # Ad-related keywords
            'num_ad_keywords': self._count_ad_keywords(html_lower),
            'ad_keyword_density': self._calculate_ad_keyword_density(html_lower),

            # JavaScript analysis
            'has_suspicious_js': self._detect_suspicious_js(html_content),
            'num_suspicious_js_patterns': self._count_suspicious_js(html_content),
            'num_external_scripts': self._count_external_scripts(html_content),

            # Tracking and analytics
            'has_google_analytics': 'google-analytics' in html_lower or 'gtag' in html_lower,
            'has_facebook_pixel': 'facebook.net' in html_lower or 'fbq(' in html_lower,
            'has_tracking_pixels': self._detect_tracking_pixels(html_lower),

            # Content ratios
            'script_to_content_ratio': self._calculate_script_ratio(html_content),
            'external_link_ratio': self._calculate_external_link_ratio(html_content, url),
            'iframe_ratio': self._calculate_iframe_ratio(html_content),

            # Redirect detection
            'has_meta_refresh': self._detect_meta_refresh(html_lower),
            'has_js_redirect': self._detect_js_redirect(html_content),
            'num_redirects': self._count_redirects(html_content),

            # Obfuscation detection
            'has_obfuscated_code': self._detect_obfuscation(html_content),
            'obfuscation_score': self._calculate_obfuscation_score(html_content),

            # Content quality
            'text_content_length': self._estimate_text_content(html_content),
            'text_to_html_ratio': self._calculate_text_ratio(html_content),
            'has_meaningful_content': self._has_meaningful_content(html_content),

            # Popup and overlay detection
            'has_popup_code': self._detect_popup_code(html_lower),
            'num_overlay_divs': self._count_overlay_divs(html_lower),

            # Third-party content
            'num_third_party_domains': self._count_third_party_domains(html_content, url),
            'third_party_script_ratio': self._calculate_third_party_ratio(html_content, url),
        }

        return features

    def _count_tags(self, html: str, tag: str) -> int:
        """Count occurrences of an HTML tag"""
        pattern = f'<{tag}[\\s>]'
        return len(re.findall(pattern, html, re.IGNORECASE))

    def _detect_ad_networks(self, html_lower: str) -> bool:
        """Check if any known ad networks are present"""
        return any(network in html_lower for network in self.AD_NETWORKS)

    def _count_ad_network_domains(self, html_lower: str) -> int:
        """Count how many different ad networks are referenced"""
        count = 0
        for network in self.AD_NETWORKS:
            if network in html_lower:
                count += 1
        return count

    def _calculate_ad_network_score(self, html_lower: str) -> float:
        """Calculate a weighted score based on ad network presence"""
        score = 0.0
        for network in self.AD_NETWORKS:
            occurrences = html_lower.count(network)
            score += min(occurrences * 0.1, 1.0)  # Cap per network at 1.0
        return min(score, 10.0)  # Cap total at 10.0

    def _count_ad_keywords(self, html_lower: str) -> int:
        """Count ad-related keywords in HTML"""
        count = 0
        for keyword in self.AD_KEYWORDS:
            count += html_lower.count(keyword)
        return count

    def _calculate_ad_keyword_density(self, html_lower: str) -> float:
        """Calculate density of ad keywords relative to content length"""
        if len(html_lower) == 0:
            return 0.0
        count = self._count_ad_keywords(html_lower)
        return (count / len(html_lower)) * 10000  # Per 10K characters

    def _detect_suspicious_js(self, html: str) -> bool:
        """Detect suspicious JavaScript patterns"""
        for pattern in self.SUSPICIOUS_JS_PATTERNS:
            if re.search(pattern, html, re.IGNORECASE):
                return True
        return False

    def _count_suspicious_js(self, html: str) -> int:
        """Count suspicious JavaScript patterns"""
        count = 0
        for pattern in self.SUSPICIOUS_JS_PATTERNS:
            count += len(re.findall(pattern, html, re.IGNORECASE))
        return count

    def _count_external_scripts(self, html: str) -> int:
        """Count external script tags"""
        pattern = r'<script[^>]+src=["\']https?://[^"\']+["\']'
        return len(re.findall(pattern, html, re.IGNORECASE))

    def _detect_tracking_pixels(self, html_lower: str) -> bool:
        """Detect 1x1 tracking pixels"""
        patterns = [
            r'width=["\']1["\'].*height=["\']1["\']',
            r'height=["\']1["\'].*width=["\']1["\']',
            r'1x1\.gif',
            r'pixel\.gif',
            r'tracking\.gif'
        ]
        return any(re.search(p, html_lower) for p in patterns)

    def _calculate_script_ratio(self, html: str) -> float:
        """Calculate ratio of script content to total content"""
        if len(html) == 0:
            return 0.0
        script_content = re.findall(r'<script[^>]*>.*?</script>', html, re.DOTALL | re.IGNORECASE)
        script_length = sum(len(s) for s in script_content)
        return script_length / len(html)

    def _calculate_external_link_ratio(self, html: str, url: str) -> float:
        """Calculate ratio of external links to total links"""
        domain = urlparse(url).netloc
        all_links = re.findall(r'href=["\']([^"\']+)["\']', html, re.IGNORECASE)
        if not all_links:
            return 0.0

        external = sum(1 for link in all_links
                      if link.startswith('http') and domain not in link)
        return external / len(all_links)

    def _calculate_iframe_ratio(self, html: str) -> float:
        """Calculate iframe content ratio"""
        if len(html) == 0:
            return 0.0
        iframes = re.findall(r'<iframe[^>]*>.*?</iframe>', html, re.DOTALL | re.IGNORECASE)
        iframe_length = sum(len(i) for i in iframes)
        return iframe_length / len(html)

    def _detect_meta_refresh(self, html_lower: str) -> bool:
        """Detect meta refresh redirects"""
        return bool(re.search(r'<meta[^>]+http-equiv=["\']refresh["\']', html_lower))

    def _detect_js_redirect(self, html: str) -> bool:
        """Detect JavaScript-based redirects"""
        patterns = [
            r'window\.location\s*=',
            r'location\.href\s*=',
            r'location\.replace\s*\(',
            r'window\.navigate\s*\('
        ]
        return any(re.search(p, html, re.IGNORECASE) for p in patterns)

    def _count_redirects(self, html: str) -> int:
        """Count total redirect mechanisms"""
        count = 0
        if self._detect_meta_refresh(html.lower()):
            count += 1
        if self._detect_js_redirect(html):
            count += 1
        return count

    def _detect_obfuscation(self, html: str) -> bool:
        """Detect code obfuscation techniques"""
        patterns = [
            r'eval\s*\(',
            r'fromCharCode',
            r'unescape\s*\(',
            r'\\x[0-9a-f]{2}',  # Hex encoding
            r'\\u[0-9a-f]{4}',  # Unicode encoding
        ]
        return any(re.search(p, html, re.IGNORECASE) for p in patterns)

    def _calculate_obfuscation_score(self, html: str) -> float:
        """Calculate obfuscation score"""
        patterns = [r'eval\s*\(', r'fromCharCode', r'unescape\s*\(']
        score = 0
        for pattern in patterns:
            score += len(re.findall(pattern, html, re.IGNORECASE))
        return min(score, 10.0)

    def _estimate_text_content(self, html: str) -> int:
        """Estimate actual text content length (excluding tags)"""
        # Remove script and style tags
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        # Remove all HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return len(text.strip())

    def _calculate_text_ratio(self, html: str) -> float:
        """Calculate text to HTML ratio"""
        if len(html) == 0:
            return 0.0
        text_len = self._estimate_text_content(html)
        return text_len / len(html)

    def _has_meaningful_content(self, html: str) -> bool:
        """Check if page has meaningful text content"""
        text_len = self._estimate_text_content(html)
        return text_len > 500  # At least 500 characters of text

    def _detect_popup_code(self, html_lower: str) -> bool:
        """Detect popup/popunder code"""
        patterns = [
            'window.open',
            'popup',
            'popunder',
            'onclick.*window.open',
        ]
        return any(pattern in html_lower for pattern in patterns)

    def _count_overlay_divs(self, html_lower: str) -> int:
        """Count potential overlay/modal divs"""
        overlay_keywords = ['modal', 'overlay', 'popup', 'lightbox', 'dialog']
        count = 0
        for keyword in overlay_keywords:
            count += html_lower.count(f'class="{keyword}"') + html_lower.count(f"class='{keyword}'")
        return count

    def _count_third_party_domains(self, html: str, url: str) -> int:
        """Count unique third-party domains referenced"""
        domain = urlparse(url).netloc
        # Find all URLs in src, href, etc.
        urls = re.findall(r'(?:src|href)=["\']https?://([^/"\'>]+)', html, re.IGNORECASE)
        third_party = set(u for u in urls if domain not in u)
        return len(third_party)

    def _calculate_third_party_ratio(self, html: str, url: str) -> float:
        """Calculate ratio of third-party scripts"""
        domain = urlparse(url).netloc
        scripts = re.findall(r'<script[^>]+src=["\']https?://([^/"\'>]+)', html, re.IGNORECASE)
        if not scripts:
            return 0.0
        third_party = sum(1 for s in scripts if domain not in s)
        return third_party / len(scripts)

    def _get_empty_features(self) -> Dict[str, Any]:
        """Return empty/default features"""
        return {
            'html_length': 0,
            'num_scripts': 0,
            'num_iframes': 0,
            'num_images': 0,
            'num_links': 0,
            'num_divs': 0,
            'num_forms': 0,
            'has_ad_networks': False,
            'num_ad_network_domains': 0,
            'ad_network_score': 0.0,
            'num_ad_keywords': 0,
            'ad_keyword_density': 0.0,
            'has_suspicious_js': False,
            'num_suspicious_js_patterns': 0,
            'num_external_scripts': 0,
            'has_google_analytics': False,
            'has_facebook_pixel': False,
            'has_tracking_pixels': False,
            'script_to_content_ratio': 0.0,
            'external_link_ratio': 0.0,
            'iframe_ratio': 0.0,
            'has_meta_refresh': False,
            'has_js_redirect': False,
            'num_redirects': 0,
            'has_obfuscated_code': False,
            'obfuscation_score': 0.0,
            'text_content_length': 0,
            'text_to_html_ratio': 0.0,
            'has_meaningful_content': False,
            'has_popup_code': False,
            'num_overlay_divs': 0,
            'num_third_party_domains': 0,
            'third_party_script_ratio': 0.0,
        }


if __name__ == '__main__':
    print("Content Feature Extractor - Loaded")
    print("Extracts 30+ features from HTML content for ML classification")