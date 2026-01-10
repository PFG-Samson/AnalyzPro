#!/usr/bin/env python3
"""Simple test runner script for the unit tests."""

import sys
import subprocess
from pathlib import Path


def run_tests():
    """Run the test suite."""
    try:
        # Try to run pytest
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/", 
            "-v", 
            "--tb=short",
            "--disable-warnings"
        ], cwd=Path(__file__).parent, check=False, capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except FileNotFoundError:
        print("Error: pytest not found. Please install it with: pip install pytest")
        return False
    except Exception as e:
        print(f"Error running tests: {e}")
        return False


def main():
    """Main function."""
    print("Running unit tests...")
    print("=" * 50)
    
    success = run_tests()
    
    print("=" * 50)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed or there were errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()