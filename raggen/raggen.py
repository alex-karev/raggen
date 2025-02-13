from typing import Optional, Literal, List, Union
from pathlib import Path
import os
import json
import pandas as pd
import logging
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered
from tqdm import tqdm
from html2text import html2text
import mammoth
import itertools
from dataclasses import asdict
from langchain_core.documents import Document
from .splitter import MarkdownSplitter
from .header_normalizer import HeaderNormalizer
from .cache_manager import CacheManager
from .metadata_manager import MetadataManager
from .models import RAGDocument, RAGInput


# Rag generator
class RAGGen:
    # Initialization
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        chunk_size: int = 256,
        chunk_overlap: int = 30,
        max_heading_level: int = 3,
        preserve_tables: bool = True,
        llm_for_headings: bool = False,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = "gpt-4o",
        template: Optional[str] = None,
        embed_meta: bool = True,
        force_ocr: bool = True,
        strip_existing_ocr: bool = True,
        languages: str = "en",
        field_names: dict = {},
        custom_meta_placement: Literal["before", "after"] = "before",
    ) -> None:
        self.header_normalizer = HeaderNormalizer(
            max_heading_level=max_heading_level,
            llm_for_headings=llm_for_headings,
            base_url=base_url,
            api_key=api_key,
            model=model,
        )
        self.splitter = MarkdownSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            preserve_tables=preserve_tables,
        )
        self.cache = CacheManager(cache_dir=cache_dir)
        self.metadata_manager = MetadataManager(
            template=template,
            custom_meta_placement=custom_meta_placement,
            encoder=self.splitter.encoder,
            field_names=field_names,
        )
        self.field_names = field_names
        self.embed_meta = embed_meta
        self.log = logging.getLogger("RAGGenerator")
        self.pdf_converter_config = ConfigParser(
            {
                "output_format": "markdown",
                "force_ocr": force_ocr,
                "strip_existing_ocr": strip_existing_ocr,
                "languages": languages,
            }
        )
        self.pdf_converter = None

    # Load PDF converter into memory
    def _load_pdf_converter(self) -> PdfConverter:
        if not self.pdf_converter:
            self.pdf_converter = PdfConverter(
                config=self.pdf_converter_config.generate_config_dict(),
                artifact_dict=create_model_dict(),
                processor_list=self.pdf_converter_config.get_processors(),
                renderer=self.pdf_converter_config.get_renderer(),
            )
        return self.pdf_converter

    # Pre-process markdown text
    def _preprocess_markdown_text(self, text: str) -> str:
        processed_text = self.cache.load("process", text=text)
        if not processed_text:
            processed_text = self.header_normalizer(text)
            self.cache.save("process", text=text, data=processed_text)
        return processed_text

    # Split markdown text
    def _split_markdown_text(
        self, text: str, metadata: Optional[dict] = None
    ) -> List[RAGDocument]:
        full_text = text
        full_text += json.dumps(metadata) if metadata and self.embed_meta else ""
        full_text += json.dumps(self.field_names) if self.embed_meta else ""
        documents = self.cache.load("rag", text=full_text)
        if documents:
            return documents
        documents = self.splitter(text)
        if metadata:
            documents = self.metadata_manager.add_metadata(documents, metadata)
        if self.embed_meta:
            documents = self.metadata_manager.embed_metadata(documents)
        self.cache.save("rag", text=full_text, data=documents)
        return documents

    # Convert any type of file to markdown text
    def _convert_file(self, path: str) -> Union[str, None]:
        # Try loading from cache
        text = self.cache.load("convert", path=path)
        if text:
            return text
        # Parse files
        try:
            if not os.path.exists(path):
                raise Exception(f"File not found '{path}'")
            extension = Path(path).suffix.lower()
            # Markdown
            if extension == ".md":
                with open(path, "r") as f:
                    text = f.read()
            # PDF
            elif extension == ".pdf":
                converter = self._load_pdf_converter()
                rendered = converter(path)
                text, _, _ = text_from_rendered(rendered)
            # HTML
            elif extension in [".html", ".htm"]:
                with open(path, "r") as f:
                    text = f.read()
                text = html2text(text)
            # Word
            elif extension in [".doc", ".docx"]:
                with open(path, "rb") as f:
                    text = mammoth.convert_to_markdown(f)
            # Unknown format
            else:
                raise Exception(f"Unsupported format '{extension}' for file '{path}'")
            # Save to cache and return
            self.cache.save("convert", path=path, data=text)
            return text
        # Conversion errors
        except Exception as e:
            self.log.error(e, exc_info=True)
            return None

    # Full document processing pipeline
    def _process(self, path: str, metadata: Optional[dict] = None) -> List[RAGDocument]:
        text = self._convert_file(path)
        if not text:
            return []
        text = self._preprocess_markdown_text(text)
        documents = self._split_markdown_text(text, metadata)
        return documents

    # Process a single file or a set of files
    def __call__(
        self,
        documents: Union[list[Union[RAGInput, str]], RAGInput, str],
        output_format: Literal["doc", "dict", "df", "langchain"] = "doc",
        flatten: bool = False,
    ) -> Union[
        List[RAGDocument],
        List[List[RAGDocument]],
        List[Document],
        List[List[Document]],
        List[dict],
        List[List[dict]],
        pd.DataFrame,
    ]:
        # Standardize input
        output_list = True
        if isinstance(documents, RAGInput):
            documents = [documents]
            output_list = False
        elif isinstance(documents, str):
            documents = [RAGInput(path=documents, metadata={})]
            output_list = False
        data = []
        # Process documents
        for doc in tqdm(documents, desc="Processing documents", total=len(documents)):
            if isinstance(doc, RAGInput) and doc.metadata:
                metadata = doc.metadata
            else:
                metadata = {}
            path = doc if isinstance(doc, str) else doc.path
            splits = self._process(path, metadata)
            if splits:
                data.append(splits)
        # Convert to dict
        if output_format == "dict":
            data = [[asdict(x) for x in doc] for doc in data]
        # Convert to langchain format
        elif output_format == "langchain":
            data = [
                [
                    Document(
                        page_content=x.text, metadata=x.metadata if x.metadata else None
                    )
                    for x in doc
                ]
                for doc in data
            ]
        # Convert to pandas dataframes
        elif output_format == "df":
            data = [pd.DataFrame([asdict(x) for x in doc]) for doc in data]
        # Flattent outputs
        if flatten and output_list:
            if output_format != "df":
                data = list(itertools.chain.from_iterable(data))
            else:
                data = pd.concat(data)
        # Output
        if output_list:
            return data
        elif len(data) != 0:
            return data[0]
        else:
            return None
