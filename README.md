# RAGGen - RAG Dataset Generator

A universal tool for converting files into high-quality RAG datasets.

## Features

- Supports a variety of commonly used formats for storing textual data.
- Robust PDF parsing with OCR utilizing [marker](https://github.com/VikParuchuri/marker) library.
- Preserves headers, supports fixing header levels using LLM (OpenAI API).
- Respects tables, does not split them into chunks.
- Supports embedding metadata directly into chunk text.
- Supports adding custom metadata for each input.
- Multiple outputs formats including pandas DataFrames and Langchain documents.
- Checksum-based result caching.

## Supported formats

- PDF (via [marker](https://github.com/VikParuchuri/marker)).
- Word (via [mammoth](https://pypi.org/project/mammoth/)).
- HTML (via [html2text](https://pypi.org/project/html2text/)).
- Markdown 

## Installation

```bash
pip install raggen
```

## Usage

```python
from raggen import RAGGen, RAGInput

# Initialize RAGGen
gen = RAGGen(cache_dir="cache")

# Define inputs
inputs = ["sample1.pdf", "sample2.html", "sample3.md"]

# Input with custom metadata
inputs.append(RAGInput(
    path = "sample4.docx",
    metadata = {"title": "Doc title"}
))

# Generate RAG dataset as list
data = gen(inputs, output_format="df", flatten=True)
```

## TODO

- [ ] Save images
- [ ] Add proper documentation
- [ ] Add more usage examples
- [ ] Add txt support
- [ ] Add Power Point support
- [ ] Add Excel support
- [ ] Add support for reading files from buffer

## Contribution

Feel free to fork this repo and make pull requests.

If you like my work, please, support me:

BTC: 32F3zAnQQGwZzsG7R35rPUS269Xz11cZ8B

## Lisense

Free to use under Apache-2.0. See LICENSE for more information.
