from typing import Optional, Literal, List, Union, Tuple
from pathlib import Path
import hashlib
import os
import json
import pandas as pd
from jinja2 import Template, Environment, PackageLoader
import logging
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered
from tqdm import tqdm
from html2text import html2text
import mammoth
from .splitter import MarkdownSplitter

# Define constants
DEFAULT_TEMPLATE = Environment(
    loader=PackageLoader("raggen", "templates")
).get_template("meta_embed")


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
        custom_meta_placement: Literal["before", "after"] = "before",
    ) -> None:
        self.splitter = MarkdownSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_heading_level=max_heading_level,
            preserve_tables=preserve_tables,
            llm_for_headings=llm_for_headings,
            base_url=base_url,
            api_key=api_key,
            model=model,
        )
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_dir = Path(cache_dir)
        self.template = Template(template) if template else DEFAULT_TEMPLATE
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
        self.custom_meta_placement = custom_meta_placement

    # Get text checksum
    def _get_text_hash(self, text: str) -> str:
        text_hash = hashlib.blake2b()
        text_hash.update(text.encode("utf8"))
        return text_hash.hexdigest()

    # Get file checksum
    def _get_file_hash(self, path: str) -> str:
        with open(path, "rb") as f:
            file_hash = hashlib.blake2b()
            while chunk := f.read(8192):
                file_hash.update(chunk)
        return file_hash.hexdigest()

    # Embed metadata to text fragments
    def _embed_metadata(self, documents: List[dict]) -> List[dict]:
        for doc in documents:
            if not "metadata" in doc or not doc["metadata"]:
                return documents
            new_text = self.template.render(text=doc["text"], metadata=doc["metadata"])
            doc["text"] = new_text.strip()
            if "length" in doc:
                doc["length"] = len(self.splitter.encoder.encode(doc["text"]))
        return documents

    # Process markdown texts
    def _process_markdown_text(
        self, text: str, metadata: Optional[dict] = None
    ) -> List[dict]:
        if self.cache_dir:
            filename = text
            if self.embed_meta:
                filename += json.dumps(metadata)
            filename = self._get_text_hash(filename)
            filename = f"{filename}.json"
            cache_path = self.cache_dir / filename
            if os.path.exists(cache_path):
                return json.load(open(cache_path, "r"))
        documents = self.splitter(text)
        if self.embed_meta:
            if metadata:
                for doc in documents:
                    if self.custom_meta_placement == "before":
                        new_meta = metadata.copy()
                        new_meta.update(doc["metadata"])
                        doc["metadata"] = new_meta
                    else:
                        doc["metadata"].update(metadata)
            documents = self._embed_metadata(documents)
        if self.cache_dir:
            json.dump(documents, open(cache_path, "w"))
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

    # Try loading any pre-processed file from cache
    def _load_markdown_from_cache(self, path: str) -> Tuple[str | None, str | None]:
        if self.cache_dir:
            filename = f"{self._get_file_hash(path)}.md"
            cache_path = self.cache_dir / filename
            if os.path.exists(cache_path):
                with open(cache_path, "r") as f:
                    return filename, f.read()
            return filename, None
        return None, None

    # Save pre-processed markdown to cache
    def _save_markdown_to_cache(self, filename: str, text: str) -> None:
        if self.cache_dir:
            cache_path = self.cache_dir / filename
            with open(cache_path, "w") as f:
                f.write(text)

    # Process PDF
    def _process_pdf_file(
        self, path: str, metadata: Optional[dict] = None
    ) -> List[dict]:
        filename, text = self._load_markdown_from_cache(path)
        if not text:
            converter = self._load_pdf_converter()
            rendered = converter(path)
            text, _, _ = text_from_rendered(rendered)
            self._save_markdown_to_cache(filename, text)
        splits = self._process_markdown_text(text, metadata)
        return splits

    # Process HTML
    def _process_html_file(
        self, path: str, metadata: Optional[dict] = None
    ) -> List[dict]:
        filename, text = self._load_markdown_from_cache(path)
        if not text:
            with open(path, "r") as f:
                text = f.read()
            text = html2text(text)
            self._save_markdown_to_cache(filename, text)
        splits = self._process_markdown_text(text, metadata)
        return splits

    # Process word
    def _process_docx_file(
        self, path: str, metadata: Optional[dict] = None
    ) -> List[dict]:
        filename, text = self._load_markdown_from_cache(path)
        if not text:
            with open(path, "rb") as f:
                text = mammoth.convert_to_markdown(f)
            self._save_markdown_to_cache(filename, text)
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
        for doc in doc_data:
            data.extend(doc)
        return pd.DataFrame(data)
