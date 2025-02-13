from typing import Optional, List
import json
from tenacity import retry, stop_after_attempt
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# LLM prompt for normalizing headers
SPLITTER_PROMPT = "Given the following headers with incorrect levels, adjust them to the correct hierarchical structure. Do not rearrange the headers. Output results exclusively in json format.\n{format_instructions}\n{query}\n"


# Header formats for llm
class HeaderFormat(BaseModel):
    text: str = Field(description="header text")
    level: int = Field(description="header level")


class HeadersFormat(BaseModel):
    headers: List[HeaderFormat] = Field(description="array of headers")


# Class for normalizing markdown headers
class HeaderNormalizer:
    def __init__(
        self,
        max_heading_level: int = 3,
        llm_for_headings: bool = False,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = "gpt-4o",
    ):
        self.llm_for_headings = llm_for_headings
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.max_heading_level = max_heading_level

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
    def __call__(self, text: str) -> str:
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
