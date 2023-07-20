from __future__ import annotations

from typing import Any

from chromadb.api.types import CollectionMetadata
from chromadb.api.types import Include
from pydantic import BaseModel


class AddEmbedding(BaseModel):  # type: ignore
    # Pydantic doesn't handle Union types cleanly like Embeddings which has
    # Union[int, float] so we use Any here to ensure data is parsed
    # to its original type.
    embeddings: list[Any] | None = None
    metadatas: list[dict[Any, Any]] | None = None
    documents: list[str] | None = None
    ids: list[str]
    increment_index: bool = True


class UpdateEmbedding(BaseModel):  # type: ignore
    embeddings: list[Any] | None = None
    metadatas: list[dict[Any, Any]] | None = None
    documents: list[str] | None = None
    ids: list[str]
    increment_index: bool = True


class QueryEmbedding(BaseModel):  # type: ignore
    # TODO: Pydantic doesn't bode well with recursive types so we use generic Dicts
    # for Where and WhereDocument. This is not ideal, but it works for now since
    # there is a lot of downstream validation.
    where: dict[Any, Any] | None = {}
    where_document: dict[Any, Any] | None = {}
    query_embeddings: list[Any]
    n_results: int = 10
    include: Include = ["metadatas", "documents", "distances"]


class GetEmbedding(BaseModel):  # type: ignore
    ids: list[str] | None = None
    where: dict[Any, Any] | None = None
    where_document: dict[Any, Any] | None = None
    sort: str | None = None
    limit: int | None = None
    offset: int | None = None
    include: Include = ["metadatas", "documents"]


class RawSql(BaseModel):  # type: ignore
    raw_sql: str


class DeleteEmbedding(BaseModel):  # type: ignore
    ids: list[str] | None = None
    where: dict[Any, Any] | None = None
    where_document: dict[Any, Any] | None = None


class CreateCollection(BaseModel):  # type: ignore
    name: str
    metadata: CollectionMetadata | None = None
    get_or_create: bool = False


class UpdateCollection(BaseModel):  # type: ignore
    new_name: str | None = None
    new_metadata: CollectionMetadata | None = None
