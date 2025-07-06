#!/usr/bin/env python3
"""
Test runner script for the long-form fact probe project.
"""

import subprocess
import sys
import os


def main():
    """Run the test suite."""
    print("Running Long-Form Fact Probe Test Suite")
    print("=" * 50)
    
    # Change to the repository root
    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)
    
    # Run pytest with coverage
    cmd = [
        sys.executable, '-m', 'pytest',
        'tests/',
        '--verbose',
        '--tb=short',
        '--cov=long_fact_probes',
        '--cov=scripts',
        '--cov-report=html:htmlcov',
        '--cov-report=term-missing'
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 50)
        print("✅ All tests passed!")
        print("Coverage report available in htmlcov/index.html")
        return 0
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 50)
        print("❌ Some tests failed!")
        return e.returncode
    except FileNotFoundError:
        print("❌ pytest not found. Please install it with: pip install pytest pytest-cov")
        return 1


if __name__ == '__main__':
    sys.exit(main()) 