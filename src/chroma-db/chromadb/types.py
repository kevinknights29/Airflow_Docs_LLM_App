from __future__ import annotations

from enum import Enum
from typing import Dict
from typing import List
from typing import Mapping
from typing import Sequence
from typing import Union
from uuid import UUID

from typing_extensions import Literal
from typing_extensions import TypedDict
from typing_extensions import TypeVar

Metadata = Mapping[str, Union[str, int, float]]
UpdateMetadata = Mapping[str, Union[int, float, str, None]]

# Namespaced Names are mechanically just strings, but we use this type to indicate that
# the intent is for the value to be globally unique and semantically meaningful.
NamespacedName = str


class ScalarEncoding(Enum):
    FLOAT32 = "FLOAT32"
    INT32 = "INT32"


class SegmentScope(Enum):
    VECTOR = "VECTOR"
    METADATA = "METADATA"


class Collection(TypedDict):
    id: UUID
    name: str
    topic: str
    metadata: Metadata | None
    dimension: int | None


class Segment(TypedDict):
    id: UUID
    type: NamespacedName
    scope: SegmentScope
    # If a segment has a topic, it implies that this segment is a consumer of the topic
    # and indexes the contents of the topic.
    topic: str | None
    # If a segment has a collection, it implies that this segment implements the full
    # collection and can be used to service queries (for it's given scope.)
    collection: UUID | None
    metadata: Metadata | None


# SeqID can be one of three types of value in our current and future plans:
# 1. A Pulsar MessageID encoded as a 192-bit integer
# 2. A Pulsar MessageIndex (a 64-bit integer)
# 3. A SQL RowID (a 64-bit integer)

# All three of these types can be expressed as a Python int, so that is the type we
# use in the internal Python API. However, care should be taken that the larger 192-bit
# values are stored correctly when persisting to DBs.
SeqId = int


class Operation(Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    UPSERT = "UPSERT"
    DELETE = "DELETE"


Vector = Union[Sequence[float], Sequence[int]]


class VectorEmbeddingRecord(TypedDict):
    id: str
    seq_id: SeqId
    embedding: Vector


class MetadataEmbeddingRecord(TypedDict):
    id: str
    seq_id: SeqId
    metadata: Metadata | None


class EmbeddingRecord(TypedDict):
    id: str
    seq_id: SeqId
    embedding: Vector | None
    encoding: ScalarEncoding | None
    metadata: UpdateMetadata | None
    operation: Operation


class SubmitEmbeddingRecord(TypedDict):
    id: str
    embedding: Vector | None
    encoding: ScalarEncoding | None
    metadata: UpdateMetadata | None
    operation: Operation


class VectorQuery(TypedDict):
    """A KNN/ANN query"""

    vectors: Sequence[Vector]
    k: int
    allowed_ids: Sequence[str] | None
    include_embeddings: bool
    options: dict[str, str | int | float] | None


class VectorQueryResult(TypedDict):
    """A KNN/ANN query result"""

    id: str
    seq_id: SeqId
    distance: float
    embedding: Vector | None


# Metadata Query Grammar
LiteralValue = Union[str, int, float]
LogicalOperator = Union[Literal["$and"], Literal["$or"]]
WhereOperator = Union[
    Literal["$gt"],
    Literal["$gte"],
    Literal["$lt"],
    Literal["$lte"],
    Literal["$ne"],
    Literal["$eq"],
]
OperatorExpression = Dict[Union[WhereOperator, LogicalOperator], LiteralValue]

Where = Dict[
    Union[str, LogicalOperator],
    Union[LiteralValue, OperatorExpression, List["Where"]],
]

WhereDocumentOperator = Union[Literal["$contains"], LogicalOperator]
WhereDocument = Dict[WhereDocumentOperator, Union[str, List["WhereDocument"]]]


class Unspecified:
    """A sentinel value used to indicate that a value should not be updated"""

    _instance: Unspecified | None = None

    def __new__(cls) -> Unspecified:
        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance


T = TypeVar("T")
OptionalArgument = Union[T, Unspecified]
