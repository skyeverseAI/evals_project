"""
export_chroma_chunks.py — export all stored chunks from the Chroma store to JSON.

Assumes your retriever code lives in src/retriever.py and exposes load_store()
(which opens the persisted Chroma store without rebuilding).

Run from the project root:
    python export_chroma_chunks.py

Produces chunks_dump.json — every chunk grouped by lecture, in reading order —
so you (or an LLM) can scan them to pick ideal_context for the faithfulness dataset.
"""

import json

from src.retriever import load_store, load_transcripts

OUT_PATH = "chunks_dump.json"


# 1. LOAD every stored chunk (no re-embedding).
#    include=["documents","metadatas"] is required — by default Chroma
#    returns ids + metadata but NOT the chunk text.
store = load_store()
data = store._collection.get(include=["documents", "metadatas"])


# 2. PLACE each chunk: find where it starts in its lecture's full text,
#    so chunks can be put back in the order they were spoken.
full_text = {d.metadata["lecture"]: d.page_content for d in load_transcripts()}

by_lecture = {}
not_found = 0
for chunk_text, meta in zip(data["documents"], data["metadatas"]):
    n = meta["lecture"]
    position = full_text[n].find(chunk_text)
    if position == -1:
        not_found += 1
        continue
    by_lecture.setdefault(n, []).append((position, chunk_text))


# 3. ORDER lectures 1..19 and chunks within each lecture by position
dump = []
for n in sorted(by_lecture, key=int):
    ordered = [text for _, text in sorted(by_lecture[n])]
    chunks = [
        {"chunk": i, "text": text}
        for i, text in enumerate(ordered, start=1)
    ]
    dump.append({"lecture": int(n), "num_chunks": len(chunks), "chunks": chunks})

with open(OUT_PATH, "w") as f:
    json.dump(dump, f, indent=2, ensure_ascii=False)


print(f"Dumped {sum(l['num_chunks'] for l in dump)} chunks to {OUT_PATH}")
if not_found:
    print(f"  WARNING: {not_found} chunks not found in the transcripts — rebuild chroma_store/")
for l in dump:
    print(f"  lecture {l['lecture']}: {l['num_chunks']} chunks")
