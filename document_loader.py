"""
Document loader module for HybridRank-RAG.

This module provides document loading functionality using langchain-community's
unstructured document loaders.
"""

from typing import Optional

from langchain_community.document_loaders.unstructured import (
    UnstructuredFileLoader,
    UnstructuredAPIFileLoader,
    UnstructuredBaseLoader,
    UnstructuredFileIOLoader,
    UnstructuredAPIFileIOLoader,
)


class DocumentLoader:
    """
    A wrapper class for loading documents using various unstructured loaders.
    """

    def __init__(self, file_path: str):
        """
        Initialize the DocumentLoader with a file path.

        Args:
            file_path (str): Path to the document to be loaded.
        """
        self.file_path = file_path
        self.loader = None

    def load_with_file_loader(self):
        """
        Load a document using UnstructuredFileLoader.

        Returns:
            List of Document objects.
        """
        self.loader = UnstructuredFileLoader(self.file_path)
        return self.loader.load()

    def load_with_api_loader(self, api_key: Optional[str] = None):
        """
        Load a document using UnstructuredAPIFileLoader.

        Args:
            api_key (Optional[str]): API key for the unstructured API.

        Returns:
            List of Document objects.
        """
        if api_key:
            self.loader = UnstructuredAPIFileLoader(self.file_path, api_key=api_key)
        else:
            self.loader = UnstructuredAPIFileLoader(self.file_path)
        return self.loader.load()


def main():
    """
    Example usage of the DocumentLoader class.
    """
    # Example: Load a document
    # loader = DocumentLoader("path/to/your/document.pdf")
    # documents = loader.load_with_file_loader()
    # print(f"Loaded {len(documents)} document(s)")
    pass


if __name__ == "__main__":
    main()