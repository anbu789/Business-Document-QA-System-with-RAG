import chromadb
client = chromadb.PersistentClient(path="chroma_store")
col = client.get_collection("documents")
print(col.count())  # Should show number of chunks