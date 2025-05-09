from ebooklib import epub, ITEM_DOCUMENT
from ebooklib.epub import Link
from bs4 import BeautifulSoup

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


EPUB_DATA_DIR = "/Users/riverhedgehog/code/deepread/data/epub"
DEFAULT_EPUB_FILE = f"{EPUB_DATA_DIR}/crime_and_punishment.epub"

def load_epub_docs(file_path=None):

    if file_path is None:
        file_path = DEFAULT_EPUB_FILE

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
        href_to_title = self._toc_mapping(book.toc)
        
        # This list will store langchain Document objects
        epub_sections: list[Document] = []

        # The table of contents or the "spine" indicates reading order.
        # For a more robust approach, you can loop through book.get_spine().
        # But we can simply iterate over all items and check if they are documents.
        for idref, linear in book.spine:
            item = book.get_item_with_id(idref)
            if item.get_type() == ITEM_DOCUMENT:
                # `item.get_content()` returns the raw HTML/XHTML
                content = item.get_content()
                text_content = self._html_to_text(content)
                section_title = self._fallback_title(content)
                chapter_title = href_to_title.get(item.get_name()) or section_title
                title = self._get_book_title(book)
                

                mdata = {
                    'source': self.file_path,
                    'book_title': title,
                    'chapter_title': chapter_title,
                    'fallback_title': section_title,
                    'linear': linear,
                    'resource_type': 'book',
                }

                if len(text_content) > 0:
                    epub_sections.append(Document(page_content=text_content, metadata=mdata))

        return epub_sections
    

    # ---------- helpers ----------
    @staticmethod
    def _get_book_title(book):
        titles = book.get_metadata('DC', 'title')
        if titles:
            return titles[0][0]
        else:
            return "Untitled Book"
    

    @staticmethod
    def _toc_mapping(toc) -> dict[str, str]:
        """
        Recursively flatten book.toc into {href_without_fragment: title}.
        """
        mapping: dict[str, str] = {}

        def walk(node):
            if isinstance(node, Link):
                mapping[node.href.split("#")[0]] = node.title.strip()
            elif isinstance(node, (list, tuple)):
                for child in node:
                    walk(child)

        walk(toc)
        return mapping


    @staticmethod
    def _html_to_text(raw_html) -> str:
        soup = BeautifulSoup(raw_html, "html.parser")

        # You can drop front-matter or footnotes here if needed, e.g.:
        #   for tag in soup.select("sup.footnote, header, footer"):
        #       tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        return text
    

    @staticmethod
    def _fallback_title(raw_html) -> str:
        """
        Use first non-empty line as a last-chance title.
        """
        soup = BeautifulSoup(raw_html, 'html.parser')
        
        # Try <title> from the HTML head:
        title_tag = soup.find('title')
        if title_tag and title_tag.get_text(strip=True):
            section_title = title_tag.get_text(separator=' ', strip=True)
        else:
            # Fallback: look for an <h1> or <h2> at the top
            heading = soup.find(['h1', 'h2', 'h3'])
            section_title = heading.get_text(separator=' ', strip=True) if heading else "Untitled Section"

        return section_title


# Usage example:
if __name__ == "__main__":
    loader = CustomEpubLoader(DEFAULT_EPUB_FILE)
    documents = loader.load()
    
    for doc in documents:
        print("Meta:", doc.metadata)
        print("Content Preview:", doc.page_content[:200])
