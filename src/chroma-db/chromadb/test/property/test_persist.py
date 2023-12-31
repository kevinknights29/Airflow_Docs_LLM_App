from __future__ import annotations

import logging
import multiprocessing
import os
import shutil
import tempfile
from multiprocessing.connection import Connection
from typing import Callable
from typing import Generator

import chromadb
import chromadb.test.property.invariants as invariants
import chromadb.test.property.strategies as strategies
import hypothesis.strategies as st
import pytest
from chromadb.api import API
from chromadb.config import Settings
from chromadb.test.property.test_embeddings import (
    collection_st as embedding_collection_st,
)
from chromadb.test.property.test_embeddings import EmbeddingStateMachine
from chromadb.test.property.test_embeddings import EmbeddingStateMachineStates
from chromadb.test.property.test_embeddings import trace
from hypothesis import given
from hypothesis.stateful import initialize
from hypothesis.stateful import precondition
from hypothesis.stateful import rule
from hypothesis.stateful import run_state_machine_as_test

CreatePersistAPI = Callable[[], API]

configurations = [
    Settings(
        chroma_api_impl="chromadb.api.segment.SegmentAPI",
        chroma_sysdb_impl="chromadb.db.impl.sqlite.SqliteDB",
        chroma_producer_impl="chromadb.db.impl.sqlite.SqliteDB",
        chroma_consumer_impl="chromadb.db.impl.sqlite.SqliteDB",
        chroma_segment_manager_impl="chromadb.segment.impl.manager.local.LocalSegmentManager",
        allow_reset=True,
        is_persistent=True,
        persist_directory=tempfile.gettempdir() + "/tests",
    ),
]


@pytest.fixture(scope="module", params=configurations)
def settings(request: pytest.FixtureRequest) -> Generator[Settings, None, None]:
    configuration = request.param
    save_path = configuration.persist_directory
    # Create if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    yield configuration
    # Remove if it exists
    if os.path.exists(save_path):
        shutil.rmtree(save_path)


collection_st = st.shared(
    strategies.collections(with_hnsw_params=True, with_persistent_hnsw_params=True),
    key="coll",
)


@given(
    collection_strategy=collection_st,
    embeddings_strategy=strategies.recordsets(collection_st),
)
def test_persist(
    settings: Settings,
    collection_strategy: strategies.Collection,
    embeddings_strategy: strategies.RecordSet,
) -> None:
    api_1 = chromadb.Client(settings)
    api_1.reset()
    coll = api_1.create_collection(
        name=collection_strategy.name,
        metadata=collection_strategy.metadata,
        embedding_function=collection_strategy.embedding_function,
    )

    if not invariants.is_metadata_valid(invariants.wrap_all(embeddings_strategy)):
        with pytest.raises(Exception):
            coll.add(**embeddings_strategy)
        return

    coll.add(**embeddings_strategy)

    invariants.count(coll, embeddings_strategy)
    invariants.metadatas_match(coll, embeddings_strategy)
    invariants.documents_match(coll, embeddings_strategy)
    invariants.ids_match(coll, embeddings_strategy)
    invariants.ann_accuracy(
        coll,
        embeddings_strategy,
        embedding_function=collection_strategy.embedding_function,
    )

    del api_1

    api_2 = chromadb.Client(settings)
    coll = api_2.get_collection(
        name=collection_strategy.name,
        embedding_function=collection_strategy.embedding_function,
    )
    invariants.count(coll, embeddings_strategy)
    invariants.metadatas_match(coll, embeddings_strategy)
    invariants.documents_match(coll, embeddings_strategy)
    invariants.ids_match(coll, embeddings_strategy)
    invariants.ann_accuracy(
        coll,
        embeddings_strategy,
        embedding_function=collection_strategy.embedding_function,
    )


def load_and_check(
    settings: Settings,
    collection_name: str,
    record_set: strategies.RecordSet,
    conn: Connection,
) -> None:
    try:
        api = chromadb.Client(settings)
        coll = api.get_collection(
            name=collection_name,
            embedding_function=strategies.not_implemented_embedding_function(),
        )
        invariants.count(coll, record_set)
        invariants.metadatas_match(coll, record_set)
        invariants.documents_match(coll, record_set)
        invariants.ids_match(coll, record_set)
        invariants.ann_accuracy(coll, record_set)
    except Exception as e:
        conn.send(e)
        raise e


class PersistEmbeddingsStateMachineStates(EmbeddingStateMachineStates):
    persist = "persist"


class PersistEmbeddingsStateMachine(EmbeddingStateMachine):
    def __init__(self, api: API, settings: Settings):
        self.api = api
        self.settings = settings
        self.last_persist_delay = 10
        self.api.reset()
        super().__init__(self.api)

    @initialize(collection=embedding_collection_st, batch_size=st.integers(min_value=3, max_value=2000), sync_threshold=st.integers(min_value=3, max_value=2000))  # type: ignore
    def initialize(
        self,
        collection: strategies.Collection,
        batch_size: int,
        sync_threshold: int,
    ):
        self.api.reset()
        self.collection = self.api.create_collection(
            name=collection.name,
            metadata=collection.metadata,
            embedding_function=collection.embedding_function,
        )
        self.embedding_function = collection.embedding_function
        trace("init")
        self.on_state_change(EmbeddingStateMachineStates.initialize)

        self.record_set_state = strategies.StateMachineRecordSet(
            ids=[],
            metadatas=[],
            documents=[],
            embeddings=[],
        )

    @precondition(
        lambda self: len(self.record_set_state["ids"]) >= 1
        and self.last_persist_delay <= 0,
    )
    @rule()
    def persist(self) -> None:
        self.on_state_change(PersistEmbeddingsStateMachineStates.persist)
        collection_name = self.collection.name
        # Create a new process and then inside the process run the invariants
        # TODO: Once we switch off of duckdb and onto sqlite we can remove this
        ctx = multiprocessing.get_context("spawn")
        conn1, conn2 = multiprocessing.Pipe()
        p = ctx.Process(
            target=load_and_check,
            args=(self.settings, collection_name, self.record_set_state, conn2),
        )
        p.start()
        p.join()

        if conn1.poll():
            e = conn1.recv()
            raise e

    def on_state_change(self, new_state: str) -> None:
        if new_state == PersistEmbeddingsStateMachineStates.persist:
            self.last_persist_delay = 10
        else:
            self.last_persist_delay -= 1

    def teardown(self) -> None:
        self.api.reset()


def test_persist_embeddings_state(
    caplog: pytest.LogCaptureFixture,
    settings: Settings,
) -> None:
    caplog.set_level(logging.ERROR)
    api = chromadb.Client(settings)
    run_state_machine_as_test(
        lambda: PersistEmbeddingsStateMachine(settings=settings, api=api),
    )  # type: ignore
