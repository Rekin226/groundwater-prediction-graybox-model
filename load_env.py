#!/usr/bin/env python3
"""
Load environment variables from .env file
Run this before using the paper agent to ensure API keys are available
"""

import os
from pathlib import Path

def load_env():
    """Load .env file into environment variables"""
    env_file = Path(__file__).parent / '.env'
    
    if not env_file.exists():
        print("❌ .env file not found!")
        print(f"   Expected location: {env_file}")
        print("\n📝 Setup instructions:")
        print("   1. Copy .env.example to .env")
        print("   2. Add your Perplexity API key to .env")
        print("   3. Run this script again")
        return False
    
    # Read .env file
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Parse KEY=VALUE
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Set environment variable
                os.environ[key] = value
                
                # Show confirmation (hide actual value for security)
                if 'KEY' in key or 'SECRET' in key or 'PASSWORD' in key:
                    print(f"✅ Loaded: {key} (length: {len(value)})")
                else:
                    print(f"✅ Loaded: {key} = {value}")
    
    # Verify Perplexity API key
    api_key = os.environ.get('PERPLEXITY_API_KEY')
    if not api_key or api_key == 'your-api-key-here':
        print("\n⚠️  PERPLEXITY_API_KEY not configured!")
        print("   Edit .env and replace 'your-api-key-here' with your actual key")
        return False
    
    print(f"\n✅ Environment configured successfully!")
    print(f"   PERPLEXITY_API_KEY: {'*' * (len(api_key) - 8) + api_key[-8:]}")
    
    return True

if __name__ == '__main__':
    import sys
    sys.exit(0 if load_env() else 1)
