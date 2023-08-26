from __future__ import annotations

import chainlit as cl
import chromadb
from chainlit_app.common import config
from chainlit_app.llm import text_generation
from langchain.chains import RetrievalQA
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

# Create or load a vector store from the database
# client = chromadb.PersistentClient(
#     path=os.path.join(
#         ROOT,
#         config()["db"]["dir"],
#     ),
# )

# Example setup of the client to connect to your chroma server
client = chromadb.HttpClient(host="server", port=8000)

# Define the embedding function
embeddings = SentenceTransformerEmbeddings(
    model_name=config()["db"]["embeddings_model"],
)

# Initialize the vector store
vector_db = Chroma(
    client=client,
    collection_name=config()["db"]["collection"],
    embedding_function=embeddings,
)
print(f"Documents Loaded: {vector_db._collection.count()}")


@cl.on_chat_start
def main():
    # Create a chain that uses the Chroma vector store
    chain = RetrievalQA.from_chain_type(
        text_generation._llm_init(),
        chain_type="refine",
        retriever=vector_db.as_retriever(search_type="mmr"),
        return_source_documents=True,
    )

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    chain = cl.user_session.get("chain")  # type: RetrievalQA

    # Call the chain synchronously in a different thread
    res = await cl.make_async(chain)(
        message,
        callbacks=[cl.LangchainCallbackHandler()],
    )

    # Post process the response
    answer = res["result"]
    source_documents = res["source_documents"]
    source_elements = []

    if source_documents:
        found_sources = []
        # Get the metadata and texts from vector database
        sources = [doc.metadata["source"] for doc in source_documents]
        texts = [doc.page_content for doc in source_documents]

        # Add the sources to the message
        for index, source in enumerate(sources):
            source_name = source.strip().replace(".", "")
            text = texts[index]
            found_sources.append(source_name)
            # Create the text element referenced in the message
            source_elements.append(cl.Text(content=text, name=source_name))

        if sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=source_elements).send()
