from __future__ import annotations

from abc import abstractmethod
from typing import Sequence
from uuid import UUID

from chromadb.config import Component
from chromadb.types import Collection
from chromadb.types import OptionalArgument
from chromadb.types import Segment
from chromadb.types import SegmentScope
from chromadb.types import Unspecified
from chromadb.types import UpdateMetadata


class SysDB(Component):
    """Data interface for Chroma's System database"""

    @abstractmethod
    def create_segment(self, segment: Segment) -> None:
        """Create a new segment in the System database. Raises DuplicateError if the ID
        already exists."""
        pass

    @abstractmethod
    def delete_segment(self, id: UUID) -> None:
        """Create a new segment in the System database."""
        pass

    @abstractmethod
    def get_segments(
        self,
        id: UUID | None = None,
        type: str | None = None,
        scope: SegmentScope | None = None,
        topic: str | None = None,
        collection: UUID | None = None,
    ) -> Sequence[Segment]:
        """Find segments by id, type, scope, topic or collection."""
        pass

    @abstractmethod
    def update_segment(
        self,
        id: UUID,
        topic: OptionalArgument[str | None] = Unspecified(),
        collection: OptionalArgument[UUID | None] = Unspecified(),
        metadata: OptionalArgument[UpdateMetadata | None] = Unspecified(),
    ) -> None:
        """Update a segment. Unspecified fields will be left unchanged. For the
        metadata, keys with None values will be removed and keys not present in the
        UpdateMetadata dict will be left unchanged."""
        pass

    @abstractmethod
    def create_collection(self, collection: Collection) -> None:
        """Create a new topic"""
        pass

    @abstractmethod
    def delete_collection(self, id: UUID) -> None:
        """Delete a topic and all associated segments from the SysDB"""
        pass

    @abstractmethod
    def get_collections(
        self,
        id: UUID | None = None,
        topic: str | None = None,
        name: str | None = None,
    ) -> Sequence[Collection]:
        """Find collections by id, topic or name"""
        pass

    @abstractmethod
    def update_collection(
        self,
        id: UUID,
        topic: OptionalArgument[str] = Unspecified(),
        name: OptionalArgument[str] = Unspecified(),
        dimension: OptionalArgument[int | None] = Unspecified(),
        metadata: OptionalArgument[UpdateMetadata | None] = Unspecified(),
    ) -> None:
        """Update a collection. Unspecified fields will be left unchanged. For metadata,
        keys with None values will be removed and keys not present in the UpdateMetadata
        dict will be left unchanged."""
        pass
