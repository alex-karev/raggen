from typing import Literal, Optional, List, Any
from jinja2 import Template, Environment, PackageLoader
from .models import RAGDocument

# Define constants
DEFAULT_TEMPLATE = Environment(
    loader=PackageLoader("raggen", "templates")
).get_template("meta_embed")


# Abstraction for dealing with metadata related operations
class MetadataManager:
    def __init__(
        self,
        encoder: Optional[Any] = None,
        template: Optional[str] = None,
        custom_meta_placement: Literal["before", "after"] = "before",
        field_names: Optional[dict] = None,
    ):
        self.encoder = encoder
        self.template = Template(template) if template else DEFAULT_TEMPLATE
        self.custom_meta_placement = custom_meta_placement
        self.field_names = (
            {"section": "Section", "subsection": "Subsection", "paragraph": "Paragraph"}
            if not field_names
            else field_names
        )

    # Embed metadata to text fragments
    def embed_metadata(self, documents: List[RAGDocument]) -> List[RAGDocument]:
        for doc in documents:
            if not doc.metadata:
                continue
            metadata = {
                self.field_names[key] if key in self.field_names else key: value
                for key, value in doc.metadata.items()
            }
            new_text = self.template.render(text=doc.text, metadata=metadata)
            doc.text = new_text.strip()
            if doc.length != None and self.encoder:
                doc.length = len(self.encoder.encode(doc.text))
        return documents

    # Add custom metadata
    def add_metadata(self, documents: List[RAGDocument], metadata: dict) -> List[RAGDocument]:
        for doc in documents:
            if self.custom_meta_placement == "before":
                new_meta = metadata.copy()
                new_meta.update(doc.metadata)
                doc.metadata = new_meta
            else:
                doc.metadata.update(metadata)
        return documents
