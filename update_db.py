from dotenv import load_dotenv

load_dotenv()

from langchain.document_loaders import UnstructuredMarkdownLoader, DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma


persist_directory = "db"
embedding = OpenAIEmbeddings(client=None)


# TODO: support upserting?
def run():
    print("indexing docs...")
    loader = DirectoryLoader(
        ".",
        glob="**/*.md",
        use_multithreading=True,
        show_progress=True,
        loader_cls=UnstructuredMarkdownLoader,
    )
    docs = loader.load()
    ids = [doc.metadata["source"] for doc in docs]

    vectordb = Chroma.from_documents(
        collection_name="langchain",
        documents=docs,
        ids=ids,
        embedding=embedding,
        persist_directory=persist_directory,
    )
    vectordb.persist()

    print("indexing done")


if __name__ == "__main__":
    run()
