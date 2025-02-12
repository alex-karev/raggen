from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List
from tenacity import retry, stop_after_attempt
import json
from typing import Optional
import tiktoken
import re

# Table finder reges
TABLE_PATTERN = re.compile(r"(\n\n(\|.+\|\n)+\n)")
SPLITTER_PROMPT = "Given the following headers with incorrect levels, adjust them to the correct hierarchical structure. Do not rearrange the headers. Output results exclusively in json format.\n{format_instructions}\n{query}\n"


# Header formats for llm
class HeaderFormat(BaseModel):
    text: str = Field(description="header text")
    level: int = Field(description="header level")


class HeadersFormat(BaseModel):
    headers: List[HeaderFormat] = Field(description="array of headers")


# Markdown text splitter
class MarkdownSplitter:
    # Initialization
    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 30,
        preserve_tables: bool = True,
        max_heading_level: int = 3,
        llm_for_headings: bool = False,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = "gpt-4o",
    ) -> None:
        headers_to_split = [
            ("#", "Section"),
            ("##", "Subsection"),
            ("###", "Paragraph"),
        ]
        self.preserve_tables = preserve_tables
        self.max_heading_level = max_heading_level
        self.llm_for_headings = llm_for_headings
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split, strip_headers=True
        )
        self.encoder = tiktoken.encoding_for_model("gpt-4o")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lambda text: len(self.encoder.encode(text)),
        )
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    # Use LLM to normalize markdown headers
    @retry(stop=stop_after_attempt(3))
    def _normalize_headers_with_llm(self, headers: dict) -> dict:
        client = ChatOpenAI(
            openai_api_base=self.base_url,
            openai_api_key=self.api_key,
            model_name=self.model,
        )
        parser = JsonOutputParser(pydantic_object=HeadersFormat)
        prompt = PromptTemplate(
            template=SPLITTER_PROMPT,
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        proc_chain = prompt | client | parser
        corrected_headers = proc_chain.invoke(
            {"query": json.dumps(headers, ensure_ascii=False)}
        )
        assert len(headers) == len(corrected_headers)
        return corrected_headers

    # Normalize markdown headers manually
    def _normalize_headers_simple(self, headers: dict) -> dict:
        corrected_headers = [
            {"text": x["text"], "level": min(x["level"], self.max_heading_level)}
            for x in headers["headers"]
        ]
        return {"headers": corrected_headers}

    # Normalize markdown headers
    def normalize_headers(self, text: str) -> str:
        # Normalize headers
        headers_texts = [
            line for line in text.split("\n") if len(line) > 0 and line[0] == "#"
        ]
        headers = [
            {"text": x.replace("#", "").strip(), "level": x.count("#")}
            for x in headers_texts
        ]
        headers = {"headers": headers}
        # Try different tec techniques
        corrected_headers = None
        if self.llm_for_headings:
            try:
                corrected_headers = self._normalize_headers_with_llm(headers)
            except Exception as e:
                print(f"Failed normalizing headers: {e}")
        if not corrected_headers:
            corrected_headers = self._normalize_headers_simple(headers)
        # Replace headers in text
        correct_texts = [
            f'{"#"*x["level"]} {x["text"]}' for x in corrected_headers["headers"]
        ]
        for i in range(len(headers_texts)):
            text = text.replace(f"\n{headers_texts[i]}\n", f"\n{correct_texts[i]}\n")
        return text

    # Extract tables and replace them with placeholders
    def extract_tables(self, text: str) -> str:
        tables = TABLE_PATTERN.findall(text)
        tables = [x[0].strip() for x in tables]
        for i, table in enumerate(tables):
            text = text.replace(table, f"{{table_{i}}}")
        return text, tables

    # Restore extracted tables in all splits
    def _restore_tables(self, splits: list[dict], tables: list[str]):
        table_tags = [f"{{table_{i}}}" for i in range(len(tables))]
        for split in splits:
            for i, tag in enumerate(table_tags):
                if tag in split["text"]:
                    split["text"] = split["text"].replace(tag, f"\n{tables[i]}\n")

    # Split markdown document
    def __call__(self, text: str) -> list[dict]:
        text = self.normalize_headers(text)
        if self.preserve_tables:
            text, tables = self.extract_tables(text)
        md_header_splits = self.markdown_splitter.split_text(text)
        splits = self.text_splitter.split_documents(md_header_splits)
        splits = [
            {
                "metadata": x.metadata,
                "text": x.page_content,
                "length": len(self.encoder.encode(x.page_content)),
            }
            for x in splits
        ]
        if self.preserve_tables:
            self._restore_tables(splits, tables)
        return splits
