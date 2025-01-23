from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from pydantic import Field

from unstructured_ingest.embed.interfaces import (
    BaseEmbeddingEncoder,
    EmbeddingConfig,
)

from ollama import Client

if TYPE_CHECKING:
    from ollama import Client


class OllamaEmbeddingConfig(EmbeddingConfig):

    embedder_model_name: str = Field(default="mistral", alias="model_name")
    ollama_client: ClassVar[Client] = Client(host="http://ie-ollama:11434")

    def get_client(self) -> "Client":
        resp = self.ollama_client.pull(model=self.embedder_model_name)  # , stream=True
        print(resp.status)
        return self.ollama_client


@dataclass #(config=MyConfig)
class OllamaEmbeddingEncoder(BaseEmbeddingEncoder):
    config: OllamaEmbeddingConfig

    def embed_query(self, query: str) -> list[float]:
        return self._embed_documents(texts=[query])[0]

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        client = self.config.get_client()
        response = client.embed(model=self.config.embedder_model_name, input=texts)
        return response.embeddings

    def embed_documents(self, elements: list[dict]) -> list[dict]:
        embeddings = self._embed_documents([e.get("text", "") for e in elements])
        elements_with_embeddings = self._add_embeddings_to_elements(elements, embeddings)
        return elements_with_embeddings
