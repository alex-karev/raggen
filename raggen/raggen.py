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
from .splitter import MarkdownSplitter
from .header_normalizer import HeaderNormalizer
from .cache_manager import CacheManager
from .metadata_manager import MetadataManager


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
            field_names=field_names
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

    # Process markdown texts
    def _process_markdown_text(
        self, text: str, metadata: Optional[dict] = None
    ) -> List[dict]:
        full_text = text
        full_text += json.dumps(metadata) if metadata and self.embed_meta else ""
        full_text += json.dumps(self.field_names) if self.embed_meta else ""
        documents = self.cache.load("rag", text=full_text)
        if documents:
            return documents
        processed_text = self.cache.load("process", text=text)
        if not processed_text:
            processed_text = self.header_normalizer(text)
            self.cache.save("process", text=text, data=processed_text)
        documents = self.splitter(processed_text)
        if metadata:
            documents = self.metadata_manager.add_metadata(documents, metadata)
        if self.embed_meta:
            documents = self.metadata_manager.embed_metadata(documents)
        self.cache.save("rag", text=full_text, data=documents)
        return documents

    # Process markdown files
    def _process_markdown_file(
        self, path: str, metadata: Optional[dict] = None
    ) -> List[dict]:
        with open(path, "r") as f:
            text = f.read()
        splits = self._process_markdown_text(text, metadata)
        return splits

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

    # Process PDF
    def _process_pdf_file(
        self, path: str, metadata: Optional[dict] = None
    ) -> List[dict]:
        text = self.cache.load("convert", path=path)
        if not text:
            converter = self._load_pdf_converter()
            rendered = converter(path)
            text, _, _ = text_from_rendered(rendered)
            self.cache.save("convert", path=path, data=text)
        splits = self._process_markdown_text(text, metadata)
        return splits

    # Process HTML
    def _process_html_file(
        self, path: str, metadata: Optional[dict] = None
    ) -> List[dict]:
        text = self.cache.load("convert", path=path)
        if not text:
            with open(path, "r") as f:
                text = f.read()
            text = html2text(text)
            self.cache.save("convert", path=path, data=text)
        splits = self._process_markdown_text(text, metadata)
        return splits

    # Process word
    def _process_docx_file(
        self, path: str, metadata: Optional[dict] = None
    ) -> List[dict]:
        text = self.cache.load("convert", path=path)
        if not text:
            with open(path, "rb") as f:
                text = mammoth.convert_to_markdown(f)
            self.cache.save("convert", path=path, data=text)
        splits = self._process_markdown_text(text, metadata)
        return splits

    # Process any type of file
    def _process(self, path: str, metadata: Optional[dict] = None) -> List[dict]:
        try:
            if not os.path.exists(path):
                raise Exception(f"File not found '{path}'")
            extension = Path(path).suffix.lower()
            if extension == ".md":
                documents = self._process_markdown_file(path, metadata)
            elif extension == ".pdf":
                documents = self._process_pdf_file(path, metadata)
            elif extension in [".html", ".htm"]:
                documents = self._process_html_file(path, metadata)
            elif extension in [".doc", ".docx"]:
                documents = self._process_docx_file(path, metadata)
            else:
                raise Exception(f"Unsupported format '{extension}' for file '{path}'")
            return documents
        except Exception as e:
            self.log.error(e, exc_info=True)
            return []

    # Process a single file or a set of files
    def __call__(
        self, documents: Union[list[Union[dict, str]], dict, str]
    ) -> Union[List[dict], List[List[dict]], None]:
        output_list = True
        if isinstance(documents, dict):
            documents = [documents]
            output_list = False
        elif isinstance(documents, str):
            documents = [{"path": documents, "metadata": {}}]
            output_list = False
        data = []
        for doc in tqdm(documents, desc="Processing documents", total=len(documents)):
            if isinstance(doc, dict) and "metadata" in doc:
                metadata = doc["metadata"]
            else:
                metadata = {}
            path = doc if isinstance(doc, str) else doc["path"]
            splits = self._process(path, metadata)
            if splits:
                data.append(splits)
        if output_list:
            return data
        elif len(data) != 0:
            return data[0]
        else:
            return None

    # Generate rag dataset
    def generate_dataset(
        self, documents: Union[list[Union[dict, str]], dict, str]
    ) -> pd.DataFrame:
        doc_data = self.__call__(documents)
        if not doc_data:
            return None
        data = []
        for index, doc in enumerate(doc_data):
            for chunk in doc:
                chunk["domain"] = index
            data.extend(doc)
        return pd.DataFrame(data)
