import hashlib
import json
import os.path
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from unstructured_ingest.interfaces import (
    EmbeddingConfig,
)
from unstructured_ingest.logger import logger
from unstructured_ingest.pipeline.interfaces import ReformatNode


@dataclass
class Embedder(ReformatNode):
    embedder_config: EmbeddingConfig

    def initialize(self):
        logger.info(
            f"Running embedding node. Embedding config: {self.embedder_config.to_json()}]",
        )
        super().initialize()

    def create_hash(self) -> str:
        hash_dict = self.embedder_config.to_dict()
        return hashlib.sha256(json.dumps(hash_dict, sort_keys=True).encode()).hexdigest()[:32]

    def run(self, elements_json: str) -> Optional[str]:
        try:
            elements_json_filename = os.path.basename(elements_json)
            filename_ext = os.path.basename(elements_json_filename)
            filename = os.path.splitext(filename_ext)[0]
            hashed_filename = hashlib.sha256(
                f"{self.create_hash()}{filename}".encode(),
            ).hexdigest()[:32]
            json_filename = f"{hashed_filename}.json"
            json_path = (Path(self.get_path()) / json_filename).resolve()
            self.pipeline_context.ingest_docs_map[hashed_filename] = (
                self.pipeline_context.ingest_docs_map[filename]
            )
            if (
                not self.pipeline_context.reprocess
                and json_path.is_file()
                and json_path.stat().st_size
            ):
                logger.debug(f"file exists: {json_path}, skipping embedding")
                return str(json_path)
            with open(elements_json) as f:
                elements = json.load(f)
            embedder = self.embedder_config.get_embedder()
            element_dicts = embedder.embed_documents(elements=elements)
            with open(json_path, "w", encoding="utf8") as output_f:
                logger.info(f"writing embeddings content to {json_path}")
                json.dump(element_dicts, output_f, ensure_ascii=False, indent=2)
            return str(json_path)
        except Exception as e:
            if self.pipeline_context.raise_on_error:
                raise
            logger.error(f"failed to embed content from file {elements_json}, {e}", exc_info=True)
            return None

    def get_path(self) -> Path:
        return (Path(self.pipeline_context.work_dir) / "embedded.py").resolve()
