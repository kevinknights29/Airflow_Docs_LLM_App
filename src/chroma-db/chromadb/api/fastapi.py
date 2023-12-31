from __future__ import annotations

import json
from typing import cast
from typing import Sequence
from uuid import UUID

import chromadb.errors as errors
import chromadb.utils.embedding_functions as ef
import pandas as pd
import requests
from chromadb.api import API
from chromadb.api.models.Collection import Collection
from chromadb.api.types import CollectionMetadata
from chromadb.api.types import Documents
from chromadb.api.types import EmbeddingFunction
from chromadb.api.types import Embeddings
from chromadb.api.types import GetResult
from chromadb.api.types import IDs
from chromadb.api.types import Include
from chromadb.api.types import Metadatas
from chromadb.api.types import QueryResult
from chromadb.api.types import Where
from chromadb.api.types import WhereDocument
from chromadb.config import Settings
from chromadb.config import System
from chromadb.telemetry import Telemetry
from overrides import override


class FastAPI(API):
    _settings: Settings

    def __init__(self, system: System):
        super().__init__(system)
        url_prefix = "https" if system.settings.chroma_server_ssl_enabled else "http"
        system.settings.require("chroma_server_host")
        system.settings.require("chroma_server_http_port")

        self._telemetry_client = self.require(Telemetry)
        self._settings = system.settings

        port_suffix = (
            f":{system.settings.chroma_server_http_port}"
            if system.settings.chroma_server_http_port
            else ""
        )
        self._api_url = (
            f"{url_prefix}://{system.settings.chroma_server_host}{port_suffix}/api/v1"
        )

        self._header = system.settings.chroma_server_headers
        self._session = requests.Session()
        if self._header is not None:
            self._session.headers.update(self._header)

    @override
    def heartbeat(self) -> int:
        """Returns the current server time in nanoseconds to check if the server is alive"""
        resp = self._session.get(self._api_url)
        raise_chroma_error(resp)
        return int(resp.json()["nanosecond heartbeat"])

    @override
    def list_collections(self) -> Sequence[Collection]:
        """Returns a list of all collections"""
        resp = self._session.get(self._api_url + "/collections")
        raise_chroma_error(resp)
        json_collections = resp.json()
        collections = []
        for json_collection in json_collections:
            collections.append(Collection(self, **json_collection))

        return collections

    @override
    def create_collection(
        self,
        name: str,
        metadata: CollectionMetadata | None = None,
        embedding_function: EmbeddingFunction | None = ef.DefaultEmbeddingFunction(),
        get_or_create: bool = False,
    ) -> Collection:
        """Creates a collection"""
        resp = self._session.post(
            self._api_url + "/collections",
            data=json.dumps(
                {"name": name, "metadata": metadata, "get_or_create": get_or_create},
            ),
        )
        raise_chroma_error(resp)
        resp_json = resp.json()
        return Collection(
            client=self,
            id=resp_json["id"],
            name=resp_json["name"],
            embedding_function=embedding_function,
            metadata=resp_json["metadata"],
        )

    @override
    def get_collection(
        self,
        name: str,
        embedding_function: EmbeddingFunction | None = ef.DefaultEmbeddingFunction(),
    ) -> Collection:
        """Returns a collection"""
        resp = self._session.get(self._api_url + "/collections/" + name)
        raise_chroma_error(resp)
        resp_json = resp.json()
        return Collection(
            client=self,
            name=resp_json["name"],
            id=resp_json["id"],
            embedding_function=embedding_function,
            metadata=resp_json["metadata"],
        )

    @override
    def get_or_create_collection(
        self,
        name: str,
        metadata: CollectionMetadata | None = None,
        embedding_function: EmbeddingFunction | None = ef.DefaultEmbeddingFunction(),
    ) -> Collection:
        return self.create_collection(
            name,
            metadata,
            embedding_function,
            get_or_create=True,
        )

    @override
    def _modify(
        self,
        id: UUID,
        new_name: str | None = None,
        new_metadata: CollectionMetadata | None = None,
    ) -> None:
        """Updates a collection"""
        resp = self._session.put(
            self._api_url + "/collections/" + str(id),
            data=json.dumps({"new_metadata": new_metadata, "new_name": new_name}),
        )
        raise_chroma_error(resp)

    @override
    def delete_collection(self, name: str) -> None:
        """Deletes a collection"""
        resp = self._session.delete(self._api_url + "/collections/" + name)
        raise_chroma_error(resp)

    @override
    def _count(self, collection_id: UUID) -> int:
        """Returns the number of embeddings in the database"""
        resp = self._session.get(
            self._api_url + "/collections/" + str(collection_id) + "/count",
        )
        raise_chroma_error(resp)
        return cast(int, resp.json())

    @override
    def _peek(self, collection_id: UUID, n: int = 10) -> GetResult:
        return self._get(
            collection_id,
            limit=n,
            include=["embeddings", "documents", "metadatas"],
        )

    @override
    def _get(
        self,
        collection_id: UUID,
        ids: IDs | None = None,
        where: Where | None = {},
        sort: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
        page: int | None = None,
        page_size: int | None = None,
        where_document: WhereDocument | None = {},
        include: Include = ["metadatas", "documents"],
    ) -> GetResult:
        if page and page_size:
            offset = (page - 1) * page_size
            limit = page_size

        resp = self._session.post(
            self._api_url + "/collections/" + str(collection_id) + "/get",
            data=json.dumps(
                {
                    "ids": ids,
                    "where": where,
                    "sort": sort,
                    "limit": limit,
                    "offset": offset,
                    "where_document": where_document,
                    "include": include,
                },
            ),
        )

        raise_chroma_error(resp)
        body = resp.json()
        return GetResult(
            ids=body["ids"],
            embeddings=body.get("embeddings", None),
            metadatas=body.get("metadatas", None),
            documents=body.get("documents", None),
        )

    @override
    def _delete(
        self,
        collection_id: UUID,
        ids: IDs | None = None,
        where: Where | None = {},
        where_document: WhereDocument | None = {},
    ) -> IDs:
        """Deletes embeddings from the database"""
        resp = self._session.post(
            self._api_url + "/collections/" + str(collection_id) + "/delete",
            data=json.dumps(
                {"where": where, "ids": ids, "where_document": where_document},
            ),
        )

        raise_chroma_error(resp)
        return cast(IDs, resp.json())

    @override
    def _add(
        self,
        ids: IDs,
        collection_id: UUID,
        embeddings: Embeddings,
        metadatas: Metadatas | None = None,
        documents: Documents | None = None,
        increment_index: bool = True,
    ) -> bool:
        """
        Adds a batch of embeddings to the database
        - pass in column oriented data lists
        - by default, the index is progressively built up as you add more data. If for ingestion performance reasons you want to disable this, set increment_index to False
        -   and then manually create the index yourself with collection.create_index()
        """
        resp = self._session.post(
            self._api_url + "/collections/" + str(collection_id) + "/add",
            data=json.dumps(
                {
                    "ids": ids,
                    "embeddings": embeddings,
                    "metadatas": metadatas,
                    "documents": documents,
                    "increment_index": increment_index,
                },
            ),
        )

        raise_chroma_error(resp)
        return True

    @override
    def _update(
        self,
        collection_id: UUID,
        ids: IDs,
        embeddings: Embeddings | None = None,
        metadatas: Metadatas | None = None,
        documents: Documents | None = None,
    ) -> bool:
        """
        Updates a batch of embeddings in the database
        - pass in column oriented data lists
        """
        resp = self._session.post(
            self._api_url + "/collections/" + str(collection_id) + "/update",
            data=json.dumps(
                {
                    "ids": ids,
                    "embeddings": embeddings,
                    "metadatas": metadatas,
                    "documents": documents,
                },
            ),
        )

        resp.raise_for_status()
        return True

    @override
    def _upsert(
        self,
        collection_id: UUID,
        ids: IDs,
        embeddings: Embeddings,
        metadatas: Metadatas | None = None,
        documents: Documents | None = None,
        increment_index: bool = True,
    ) -> bool:
        """
        Upserts a batch of embeddings in the database
        - pass in column oriented data lists
        """
        resp = self._session.post(
            self._api_url + "/collections/" + str(collection_id) + "/upsert",
            data=json.dumps(
                {
                    "ids": ids,
                    "embeddings": embeddings,
                    "metadatas": metadatas,
                    "documents": documents,
                    "increment_index": increment_index,
                },
            ),
        )

        resp.raise_for_status()
        return True

    @override
    def _query(
        self,
        collection_id: UUID,
        query_embeddings: Embeddings,
        n_results: int = 10,
        where: Where | None = {},
        where_document: WhereDocument | None = {},
        include: Include = ["metadatas", "documents", "distances"],
    ) -> QueryResult:
        """Gets the nearest neighbors of a single embedding"""
        resp = self._session.post(
            self._api_url + "/collections/" + str(collection_id) + "/query",
            data=json.dumps(
                {
                    "query_embeddings": query_embeddings,
                    "n_results": n_results,
                    "where": where,
                    "where_document": where_document,
                    "include": include,
                },
            ),
        )

        raise_chroma_error(resp)
        body = resp.json()

        return QueryResult(
            ids=body["ids"],
            distances=body.get("distances", None),
            embeddings=body.get("embeddings", None),
            metadatas=body.get("metadatas", None),
            documents=body.get("documents", None),
        )

    @override
    def reset(self) -> bool:
        """Resets the database"""
        resp = self._session.post(self._api_url + "/reset")
        raise_chroma_error(resp)
        return cast(bool, resp.json())

    @override
    def raw_sql(self, sql: str) -> pd.DataFrame:
        """Runs a raw SQL query against the database"""
        resp = self._session.post(
            self._api_url + "/raw_sql",
            data=json.dumps({"raw_sql": sql}),
        )
        raise_chroma_error(resp)
        return pd.DataFrame.from_dict(resp.json())

    @override
    def create_index(self, collection_name: str) -> bool:
        """Soon deprecated"""
        resp = self._session.post(
            self._api_url + "/collections/" + collection_name + "/create_index",
        )
        raise_chroma_error(resp)
        return cast(bool, resp.json())

    @override
    def get_version(self) -> str:
        """Returns the version of the server"""
        resp = self._session.get(self._api_url + "/version")
        raise_chroma_error(resp)
        return cast(str, resp.json())

    @override
    def get_settings(self) -> Settings:
        """Returns the settings of the client"""
        return self._settings


def raise_chroma_error(resp: requests.Response) -> None:
    """Raises an error if the response is not ok, using a ChromaError if possible"""
    if resp.ok:
        return

    chroma_error = None
    try:
        body = resp.json()
        if "error" in body:
            if body["error"] in errors.error_types:
                chroma_error = errors.error_types[body["error"]](body["message"])

    except BaseException:
        pass

    if chroma_error:
        raise chroma_error

    try:
        resp.raise_for_status()
    except requests.HTTPError:
        raise (Exception(resp.text))
