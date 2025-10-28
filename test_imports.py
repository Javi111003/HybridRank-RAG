"""
Test module to verify langchain-community imports work correctly.
"""

import sys


def test_import_unstructured_loaders():
    """Test that unstructured loaders can be imported successfully."""
    try:
        from langchain_community.document_loaders.unstructured import (
            UnstructuredFileLoader,
            UnstructuredAPIFileLoader,
            UnstructuredBaseLoader,
            UnstructuredFileIOLoader,
            UnstructuredAPIFileIOLoader,
        )
        print("✓ Successfully imported all unstructured loaders")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_document_loader_module():
    """Test that the document_loader module can be imported."""
    try:
        import document_loader
        print("✓ Successfully imported document_loader module")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_document_loader_class():
    """Test that the DocumentLoader class can be instantiated."""
    try:
        from document_loader import DocumentLoader
        loader = DocumentLoader("test.txt")
        print(f"✓ Successfully created DocumentLoader instance with file: {loader.file_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to create DocumentLoader: {e}")
        return False


def main():
    """Run all tests."""
    print("Running import tests...\n")
    
    tests = [
        ("Unstructured Loaders Import", test_import_unstructured_loaders),
        ("Document Loader Module Import", test_document_loader_module),
        ("Document Loader Class", test_document_loader_class),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Test: {test_name}")
        result = test_func()
        results.append(result)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    print(f"{'='*50}")
    print(f"Test Results: {passed}/{total} passed")
    print(f"{'='*50}")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())