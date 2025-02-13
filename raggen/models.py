from dataclasses import dataclass
from typing import Optional

# Rag input anbstraction
@dataclass
class RAGInput:
    path: str
    metadata: Optional[dict] = None

# Rag document abstraction
@dataclass
class RAGDocument:
    text: str
    length: int
    metadata: Optional[dict] = None

