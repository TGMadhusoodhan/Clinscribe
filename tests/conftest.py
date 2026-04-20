"""
Pytest configuration for ClinScribe tests.
Sets a dummy ANTHROPIC_API_KEY so config.py does not raise on import in structural tests.
Tests that actually call the Anthropic API are guarded by skipif checks in each test file.
"""

import os
os.environ.setdefault("ANTHROPIC_API_KEY", "test-dummy-key-for-structural-tests")
