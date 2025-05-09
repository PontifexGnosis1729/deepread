from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_epub_docs(file_path=None):
    EPUB_DATA_DIR = "/Users/riverhedgehog/code/hackaitx25final/epubdata"
    epub_file = f"{EPUB_DATA_DIR}/superintelligence.epub"

    if file_path is None:
        file_path = epub_file

    smart_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". "],
        chunk_size=1500,
        chunk_overlap=0,
    )

    docs = CustomEpubLoader(file_path).load()
    docs_transformed = smart_splitter.split_documents( docs )

    return docs_transformed


class CustomEpubLoader(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> list[Document]:
        """Load data from the file and return a list of Document objects."""
        # Open the EPUB
        book = epub.read_epub(self.file_path)
        
        # This list will store langchain Document objects
        epub_sections = []

        # The table of contents or the "spine" indicates reading order.
        # For a more robust approach, you can loop through book.get_spine().
        # But we can simply iterate over all items and check if they are documents.
        for item in book.get_items():
            if item.get_type() == ITEM_DOCUMENT:
                # `item.get_content()` returns the raw HTML/XHTML
                content = item.get_content()
                
                # Use BeautifulSoup to parse
                soup = BeautifulSoup(content, 'html.parser')
                
                # Try <title> from the HTML head:
                title_tag = soup.find('title')
                if title_tag and title_tag.get_text(strip=True):
                    section_title = title_tag.get_text(separator=' ', strip=True)
                else:
                    # Fallback: look for an <h1> or <h2> at the top
                    heading = soup.find(['h1', 'h2', 'h3'])
                    section_title = heading.get_text(separator=' ', strip=True) if heading else "Untitled Section"

                # Extract plain text from the body
                # (Depending on your needs, you can be more selective about which elements to include.)
                text_content = soup.get_text(separator='\n', strip=True)

                mdata = {
                    'source': self.file_path,
                    'title': section_title,
                }
                if len(text_content) > 0:
                    epub_sections.append(Document(page_content=text_content, metadata=mdata))

        return epub_sections

# Usage example:
if __name__ == "__main__":
    loader = CustomEpubLoader("path/to/your/file.epub")
    documents = loader.load()
    
    for doc in documents:
        print("Source:", doc.metadata["source"])
        print("Content Preview:", doc.page_content[:200])
