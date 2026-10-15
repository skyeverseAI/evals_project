import os
import re
import glob

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()  # loads OPENAI_API_KEY from .env

DATA_DIR = "data_en"
DB_DIR = "chroma_store"


# 1. LOAD ---- read each transcript, throw away the VTT timestamps
def load_transcripts():

    docs = []
    for path in glob.glob(f"{DATA_DIR}/*.vtt"):
        lines = []
        for line in open(path,encoding="utf-8"):
            line = line.strip()
            if (not line or line == "WEBVTT" or "-->" in line or line.isdigit()):
                continue
            lines.append(line)
        text = " ".join(lines)

        lecture = re.search(r"Lecture-(\d+)", path).group(1)

        docs.append(Document(page_content=text, metadata={"lecture": lecture}))

    return docs


# 2. BUILD ---- chunk, embed once, and keep it on disk so we don't re-embed
def load_store():
    embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-large",        # OpenRouter's name for the model
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    check_embedding_ctx_length=False,             # send plain text, not token IDs
    )

    if os.path.exists(DB_DIR):
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    docs = load_transcripts()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=100,
    ).split_documents(docs)

    return Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)


def build_retriever():
    return load_store().as_retriever(search_kwargs={"k": 5})


# 3. TRY IT ---- python src/retriever.py
if __name__ == "__main__":

    retriever = build_retriever()

    results = retriever.invoke("what is regression testing?")
    
    for r in results:
        print(f"[Lecture {r.metadata['lecture']}] {r.page_content[:150]}...\n")