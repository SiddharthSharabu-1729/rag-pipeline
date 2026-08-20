"""Parser factory for dynamic document format resolution."""

from pathlib import Path
from rag.domain.exceptions import DocumentParsingError
from rag.ingestion.parsers.base import BaseParser
from rag.ingestion.parsers.markdown_parser import MarkdownParser
from rag.ingestion.parsers.pdf_parser import PDFParser
from rag.ingestion.parsers.text_parser import TextParser


class ParserFactory:
    """Factory to retrieve the appropriate parser based on file extension."""

    _PARSERS: dict[str, type[BaseParser]] = {
        ".md": MarkdownParser,
        ".markdown": MarkdownParser,
        ".txt": TextParser,
        ".log": TextParser,
        ".json": TextParser,
        ".pdf": PDFParser,
    }

    @classmethod
    def get_parser(cls, filename_or_path: str | Path) -> BaseParser:
        """Return an instantiated parser for the given filename or path."""
        suffix = Path(filename_or_path).suffix.lower()
        parser_cls = cls._PARSERS.get(suffix, TextParser)
        return parser_cls()

