from typing import Optional, Union, Literal
import os
from pathlib import Path
import hashlib
import json


# Abstraction for managing cache
class CacheManager:
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_dir = Path(cache_dir)
            os.makedirs(self.cache_dir / "convert", exist_ok=True)
            os.makedirs(self.cache_dir / "process", exist_ok=True)
            os.makedirs(self.cache_dir / "rag", exist_ok=True)

    # Get text checksum
    def _get_text_hash(self, text: str) -> str:
        text_hash = hashlib.blake2b(digest_size=32)
        text_hash.update(text.encode("utf8"))
        return text_hash.hexdigest()

    # Get file checksum
    def _get_file_hash(self, path: str) -> str:
        with open(path, "rb") as f:
            file_hash = hashlib.blake2b(digest_size=32)
            while chunk := f.read(8192):
                file_hash.update(chunk)
        return file_hash.hexdigest()

    # Load from cache
    def load(
        self,
        cache_type: Literal["convert", "process", "rag"],
        *args,
        path: Optional[str] = None,
        text: Optional[str] = None,
    ) -> Union[str, dict, None]:
        if not self.cache_dir or not (path or text):
            return None
        filename = self._get_file_hash(path) if path else self._get_text_hash(text)
        format = "json" if cache_type == "rag" else "md"
        cache_path = self.cache_dir / cache_type / f"{filename}.{format}"
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                output = f.read()
            if format == "json":
                return json.loads(output)
            return output

    # Save to cache
    def save(
        self,
        cache_type: Literal["convert", "process", "rag"],
        *args,
        path: Optional[str] = None,
        text: Optional[str] = None,
        data: Union[str, list],
    ) -> None:
        if not self.cache_dir or not (path or text):
            return None
        filename = self._get_file_hash(path) if path else self._get_text_hash(text)
        format = "json" if cache_type == "rag" else "md"
        cache_path = self.cache_dir / cache_type / f"{filename}.{format}"
        with open(cache_path, "w") as f:
            if format == "json":
                f.write(json.dumps(data))
            else:
                f.write(data)
