from __future__ import annotations

import chainlit as cl
import chromadb
from chainlit_app.common import config
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from transformers import pipeline

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


@cl.langchain_factory(use_async=False)
async def init():
    # Model setup
    model = config()["model"]
    text_gen_pipeline = pipeline(
        model=model,
        model_kwargs={
            "device_map": "auto",
            "load_in_8bit": False,
            "temperature": 0.1,
            "top_p": 1.0,
            "max_length": 1024,
        },
        max_new_tokens=2048,
        # device=torch.device('mps'), # remove this line if you don't have an M1+ Mac and uncomment the line 45
        # torch_dtype=torch.float16,
    )
    llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

    # Create a message to let the user know that the system is loading
    msg = cl.Message(content="Init started! This may take a while...")
    await msg.send()

    # Create a chain that uses the Chroma vector store
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="refine",
        retriever=vector_db.as_retriever(search_type="mmr"),
    )

    # Let the user know that the system is ready
    # Create a message to let the user know that the system is loading
    msg = cl.Message(content="Init finished... You can now ask questions!")
    await msg.send()

    return chain


@cl.langchain_postprocess
async def process_response(res):
    answer = res["answer"]
    sources = res["sources"].strip()
    source_elements = []

    # Get the metadata and texts from the user session
    metadatas = cl.user_session.get("metadatas")
    all_sources = [m["source"] for m in metadatas]
    texts = cl.user_session.get("texts")

    if sources:
        found_sources = []

        # Add the sources to the message
        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            # Get the index of the source
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = texts[index]
            found_sources.append(source_name)
            # Create the text element referenced in the message
            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=source_elements).send()
