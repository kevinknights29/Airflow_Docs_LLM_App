from __future__ import annotations

from abc import abstractmethod
from typing import Sequence
from uuid import UUID

from chromadb.api.types import Documents
from chromadb.api.types import Embeddings
from chromadb.api.types import IDs
from chromadb.api.types import Metadata
from chromadb.api.types import Metadatas
from chromadb.api.types import Where
from chromadb.api.types import WhereDocument
from chromadb.config import Component


class DB(Component):
    @abstractmethod
    def create_collection(
        self,
        name: str,
        metadata: Metadata | None = None,
        get_or_create: bool = False,
    ) -> Sequence:  # type: ignore
        pass

    @abstractmethod
    def get_collection(self, name: str) -> Sequence:  # type: ignore
        pass

    @abstractmethod
    def list_collections(self) -> Sequence:  # type: ignore
        pass

    @abstractmethod
    def update_collection(
        self,
        id: UUID,
        new_name: str | None = None,
        new_metadata: Metadata | None = None,
    ) -> None:
        pass

    @abstractmethod
    def delete_collection(self, name: str) -> None:
        pass

    @abstractmethod
    def get_collection_uuid_from_name(self, collection_name: str) -> UUID:
        pass

    @abstractmethod
    def add(
        self,
        collection_uuid: UUID,
        embeddings: Embeddings,
        metadatas: Metadatas | None,
        documents: Documents | None,
        ids: list[str],
    ) -> list[UUID]:
        pass

    @abstractmethod
    def add_incremental(
        self,
        collection_uuid: UUID,
        ids: list[UUID],
        embeddings: Embeddings,
    ) -> None:
        pass

    @abstractmethod
    def get(
        self,
        where: Where = {},
        collection_name: str | None = None,
        collection_uuid: UUID | None = None,
        ids: IDs | None = None,
        sort: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
        where_document: WhereDocument = {},
        columns: list[str] | None = None,
    ) -> Sequence:  # type: ignore
        pass

    @abstractmethod
    def update(
        self,
        collection_uuid: UUID,
        ids: IDs,
        embeddings: Embeddings | None = None,
        metadatas: Metadatas | None = None,
        documents: Documents | None = None,
    ) -> bool:
        pass

    @abstractmethod
    def count(self, collection_id: UUID) -> int:
        pass

    @abstractmethod
    def delete(
        self,
        where: Where = {},
        collection_uuid: UUID | None = None,
        ids: IDs | None = None,
        where_document: WhereDocument = {},
    ) -> list[str]:
        pass

    @abstractmethod
    def get_nearest_neighbors(
        self,
        collection_uuid: UUID,
        where: Where = {},
        embeddings: Embeddings | None = None,
        n_results: int = 10,
        where_document: WhereDocument = {},
    ) -> tuple[list[list[UUID]], list[list[float]]]:
        pass

    @abstractmethod
    def get_by_ids(
        self,
        uuids: list[UUID],
        columns: list[str] | None = None,
    ) -> Sequence:  # type: ignore
        pass

    @abstractmethod
    def raw_sql(self, raw_sql):  # type: ignore
        pass

    @abstractmethod
    def create_index(self, collection_uuid: UUID):  # type: ignore
        pass
