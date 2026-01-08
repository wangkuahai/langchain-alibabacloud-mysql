"""Smoke test to verify the package can be imported and basic functionality works."""

import sys

try:
    from langchain_alibabacloud_mysql import AlibabaCloudMySQL

    print("✓ Successfully imported AlibabaCloudMySQL")
    print(f"✓ AlibabaCloudMySQL class: {AlibabaCloudMySQL}")
    print("✓ Package import test passed")
    sys.exit(0)
except ImportError as e:
    print(f"✗ Failed to import package: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    sys.exit(1)
