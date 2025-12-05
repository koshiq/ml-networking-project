#!/usr/bin/env python3
"""
Configuration loader for URL Classifier
Loads settings from .env file using python-dotenv
"""

import os
from pathlib import Path
from typing import Optional

# Try to import python-dotenv
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")


class Config:
    """Configuration manager for URL Classifier"""

    def __init__(self):
        # Load .env file if available
        if DOTENV_AVAILABLE:
            # Look for .env in project root
            env_path = Path(__file__).parent.parent / '.env'
            if env_path.exists():
                load_dotenv(env_path)
                print(f"✓ Loaded configuration from {env_path}")
            else:
                print(f"⚠️  No .env file found at {env_path}")
                print("   Create one from .env.example")

        # Load configuration from environment variables
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.model_name = os.getenv('MODEL_NAME', 'llama-3.1-8b-instant')
        self.groq_api_url = os.getenv('GROQ_API_URL', 'https://api.groq.com/openai/v1/chat/completions')

        # Fetcher settings
        self.fetch_timeout = int(os.getenv('FETCH_TIMEOUT', '10'))
        self.max_retries = int(os.getenv('MAX_RETRIES', '2'))

        # Trie settings
        self.trie_save_path = os.getenv('TRIE_SAVE_PATH', 'data/domain_trie.json')

    def validate(self) -> bool:
        """Check if configuration is valid"""
        if not self.groq_api_key or self.groq_api_key == 'your_groq_api_key_here':
            print("\n⚠️  WARNING: No valid Groq API key configured!")
            print("   Get your free API key at: https://console.groq.com")
            print("   Add it to .env file: GROQ_API_KEY=your_key_here")
            print("   Will use heuristic classification only.\n")
            return False
        return True

    def __repr__(self):
        api_key_display = self.groq_api_key[:10] + "..." if self.groq_api_key else "None"
        return f"""Config(
    groq_api_key={api_key_display},
    model_name={self.model_name},
    groq_api_url={self.groq_api_url},
    fetch_timeout={self.fetch_timeout}s,
    trie_save_path={self.trie_save_path}
)"""


# Global config instance
config = Config()


if __name__ == '__main__':
    print("Configuration Loader Test")
    print("=" * 60)
    print(config)
    print("\nValidation:", "✓ Valid" if config.validate() else "✗ Invalid (will use heuristics)")
