from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
import tiktoken
import re

# Table finder reges
TABLE_PATTERN = re.compile(r"(\n\n(\|.+\|\n)+\n)")


# Markdown text splitter
class MarkdownSplitter:
    # Initialization
    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 30,
        preserve_tables: bool = True,
    ) -> None:
        headers_to_split = [
            ("#", "section"),
            ("##", "subsection"),
            ("###", "paragraph"),
        ]
        self.preserve_tables = preserve_tables
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split, strip_headers=True
        )
        self.encoder = tiktoken.encoding_for_model("gpt-4o")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lambda text: len(self.encoder.encode(text)),
        )

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
