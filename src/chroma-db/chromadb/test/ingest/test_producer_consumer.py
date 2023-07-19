from __future__ import annotations

import os
import shutil
import tempfile
from asyncio import Event
from asyncio import TimeoutError
from asyncio import wait_for
from itertools import count
from typing import Callable
from typing import Generator
from typing import Iterator
from typing import Sequence

import pytest
from chromadb.config import Settings
from chromadb.config import System
from chromadb.db.impl.sqlite import SqliteDB
from chromadb.ingest import Consumer
from chromadb.ingest import Producer
from chromadb.types import EmbeddingRecord
from chromadb.types import Operation
from chromadb.types import ScalarEncoding
from chromadb.types import SubmitEmbeddingRecord
from pytest import approx
from pytest import FixtureRequest


def sqlite() -> Generator[tuple[Producer, Consumer], None, None]:
    """Fixture generator for sqlite Producer + Consumer"""
    system = System(Settings(allow_reset=True))
    db = system.require(SqliteDB)
    system.start()
    yield db, db
    system.stop()


def sqlite_persistent() -> Generator[tuple[Producer, Consumer], None, None]:
    """Fixture generator for sqlite_persistent Producer + Consumer"""
    save_path = tempfile.mkdtemp()
    system = System(
        Settings(allow_reset=True, is_persistent=True, persist_directory=save_path),
    )
    db = system.require(SqliteDB)
    system.start()
    yield db, db
    system.stop()
    if os.path.exists(save_path):
        shutil.rmtree(save_path)


def fixtures() -> list[Callable[[], Generator[tuple[Producer, Consumer], None, None]]]:
    return [sqlite, sqlite_persistent]


@pytest.fixture(scope="module", params=fixtures())
def producer_consumer(
    request: FixtureRequest,
) -> Generator[tuple[Producer, Consumer], None, None]:
    yield next(request.param())


@pytest.fixture(scope="module")
def sample_embeddings() -> Iterator[SubmitEmbeddingRecord]:
    def create_record(i: int) -> SubmitEmbeddingRecord:
        vector = [i + i * 0.1, i + 1 + i * 0.1]
        metadata: dict[str, str | int | float] | None
        if i % 2 == 0:
            metadata = None
        else:
            metadata = {"str_key": f"value_{i}", "int_key": i, "float_key": i + i * 0.1}

        record = SubmitEmbeddingRecord(
            id=f"embedding_{i}",
            embedding=vector,
            encoding=ScalarEncoding.FLOAT32,
            metadata=metadata,
            operation=Operation.ADD,
        )
        return record

    return (create_record(i) for i in count())


class CapturingConsumeFn:
    embeddings: list[EmbeddingRecord]
    waiters: list[tuple[int, Event]]

    def __init__(self) -> None:
        self.embeddings = []
        self.waiters = []

    def __call__(self, embeddings: Sequence[EmbeddingRecord]) -> None:
        self.embeddings.extend(embeddings)
        for n, event in self.waiters:
            if len(self.embeddings) >= n:
                event.set()

    async def get(self, n: int) -> Sequence[EmbeddingRecord]:
        "Wait until at least N embeddings are available, then return all embeddings"
        if len(self.embeddings) >= n:
            return self.embeddings[:n]
        else:
            event = Event()
            self.waiters.append((n, event))
            # timeout so we don't hang forever on failure
            await wait_for(event.wait(), 10)
            return self.embeddings[:n]


def assert_approx_equal(a: Sequence[float], b: Sequence[float]) -> None:
    for i, j in zip(a, b):
        assert approx(i) == approx(j)


def assert_records_match(
    inserted_records: Sequence[SubmitEmbeddingRecord],
    consumed_records: Sequence[EmbeddingRecord],
) -> None:
    """Given a list of inserted and consumed records, make sure they match"""
    assert len(consumed_records) == len(inserted_records)
    for inserted, consumed in zip(inserted_records, consumed_records):
        assert inserted["id"] == consumed["id"]
        assert inserted["operation"] == consumed["operation"]
        assert inserted["encoding"] == consumed["encoding"]
        assert inserted["metadata"] == consumed["metadata"]

        if inserted["embedding"] is not None:
            assert consumed["embedding"] is not None
            assert_approx_equal(inserted["embedding"], consumed["embedding"])


@pytest.mark.asyncio
async def test_backfill(
    producer_consumer: tuple[Producer, Consumer],
    sample_embeddings: Iterator[SubmitEmbeddingRecord],
) -> None:
    producer, consumer = producer_consumer
    producer.reset_state()

    embeddings = [next(sample_embeddings) for _ in range(3)]

    producer.create_topic("test_topic")
    for e in embeddings:
        producer.submit_embedding("test_topic", e)

    consume_fn = CapturingConsumeFn()
    consumer.subscribe("test_topic", consume_fn, start=consumer.min_seqid())

    recieved = await consume_fn.get(3)
    assert_records_match(embeddings, recieved)


@pytest.mark.asyncio
async def test_notifications(
    producer_consumer: tuple[Producer, Consumer],
    sample_embeddings: Iterator[SubmitEmbeddingRecord],
) -> None:
    producer, consumer = producer_consumer
    producer.reset_state()
    producer.create_topic("test_topic")

    embeddings: list[SubmitEmbeddingRecord] = []

    consume_fn = CapturingConsumeFn()

    consumer.subscribe("test_topic", consume_fn, start=consumer.min_seqid())

    for i in range(10):
        e = next(sample_embeddings)
        embeddings.append(e)
        producer.submit_embedding("test_topic", e)
        received = await consume_fn.get(i + 1)
        assert_records_match(embeddings, received)


@pytest.mark.asyncio
async def test_multiple_topics(
    producer_consumer: tuple[Producer, Consumer],
    sample_embeddings: Iterator[SubmitEmbeddingRecord],
) -> None:
    producer, consumer = producer_consumer
    producer.reset_state()
    producer.create_topic("test_topic_1")
    producer.create_topic("test_topic_2")

    embeddings_1: list[SubmitEmbeddingRecord] = []
    embeddings_2: list[SubmitEmbeddingRecord] = []

    consume_fn_1 = CapturingConsumeFn()
    consume_fn_2 = CapturingConsumeFn()

    consumer.subscribe("test_topic_1", consume_fn_1, start=consumer.min_seqid())
    consumer.subscribe("test_topic_2", consume_fn_2, start=consumer.min_seqid())

    for i in range(10):
        e_1 = next(sample_embeddings)
        embeddings_1.append(e_1)
        producer.submit_embedding("test_topic_1", e_1)
        results_2 = await consume_fn_1.get(i + 1)
        assert_records_match(embeddings_1, results_2)

        e_2 = next(sample_embeddings)
        embeddings_2.append(e_2)
        producer.submit_embedding("test_topic_2", e_2)
        results_2 = await consume_fn_2.get(i + 1)
        assert_records_match(embeddings_2, results_2)


@pytest.mark.asyncio
async def test_start_seq_id(
    producer_consumer: tuple[Producer, Consumer],
    sample_embeddings: Iterator[SubmitEmbeddingRecord],
) -> None:
    producer, consumer = producer_consumer
    producer.reset_state()
    producer.create_topic("test_topic")

    consume_fn_1 = CapturingConsumeFn()
    consume_fn_2 = CapturingConsumeFn()

    consumer.subscribe("test_topic", consume_fn_1, start=consumer.min_seqid())

    embeddings = []
    for _ in range(5):
        e = next(sample_embeddings)
        embeddings.append(e)
        producer.submit_embedding("test_topic", e)

    results_1 = await consume_fn_1.get(5)
    assert_records_match(embeddings, results_1)

    start = consume_fn_1.embeddings[-1]["seq_id"]
    consumer.subscribe("test_topic", consume_fn_2, start=start)
    for _ in range(5):
        e = next(sample_embeddings)
        embeddings.append(e)
        producer.submit_embedding("test_topic", e)

    results_2 = await consume_fn_2.get(5)
    assert_records_match(embeddings[-5:], results_2)


@pytest.mark.asyncio
async def test_end_seq_id(
    producer_consumer: tuple[Producer, Consumer],
    sample_embeddings: Iterator[SubmitEmbeddingRecord],
) -> None:
    producer, consumer = producer_consumer
    producer.reset_state()
    producer.create_topic("test_topic")

    consume_fn_1 = CapturingConsumeFn()
    consume_fn_2 = CapturingConsumeFn()

    consumer.subscribe("test_topic", consume_fn_1, start=consumer.min_seqid())

    embeddings = []
    for _ in range(10):
        e = next(sample_embeddings)
        embeddings.append(e)
        producer.submit_embedding("test_topic", e)

    results_1 = await consume_fn_1.get(10)
    assert_records_match(embeddings, results_1)

    end = consume_fn_1.embeddings[-5]["seq_id"]
    consumer.subscribe("test_topic", consume_fn_2, start=consumer.min_seqid(), end=end)

    results_2 = await consume_fn_2.get(6)
    assert_records_match(embeddings[:6], results_2)

    # Should never produce a 7th
    with pytest.raises(TimeoutError):
        _ = await wait_for(consume_fn_2.get(7), timeout=1)
