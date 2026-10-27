from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai import OpenAI
import os
import chromadb


MAX_DISTANCE = 0.75
CHROMA_DIR = Path("data/chroma")
COLLECTION_NAME = "agent_knowledge"
KNOWLEDGE_DIR = Path("data/knowledge")
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
load_dotenv(Path(__file__).with_name(".env"))
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise RuntimeError("DASHSCOPE_API_KEY environment variable is not set.")

client = OpenAI(
    api_key=api_key,
    base_url="https://ws-tvd06scn7rvvfzug.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
)

MODEL = "qwen3.8-flash"



def load_documents() ->list[dict]:
    """
    Load documents from the knowledge directory.
    Each document is expected to be a text file.
    Returns a list of dictionaries with 'filename' and 'content'.
    """
    documents = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue  # Skip empty files
        documents.append({
            "filename": path.name,
            "text": text,
        })
    return documents


def split_documents(documents: list[dict]) -> list[dict]:
    """
    Split documents into chunks with overlap.
    Returns a list of dictionaries with 'filename' and 'chunk'.
    """
    chunks = []
    for document in documents:
        text = document["text"]
        start = 0
        chunk_id = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk_text = text[start:end]
            chunks.append({
                "filename": document["filename"],
                "chunk_id": chunk_id,
                "chunk": chunk_text,
            })
            if end >= len(text):
                break
            start = end - CHUNK_OVERLAP
            chunk_id += 1
    return chunks




def create_embedding(text: list[str]) -> list[list[float]]:
    """
    Create an embedding for the given text using the OpenAI API.
    Returns a list of floats representing the embedding vector.
    """
    response = client.embeddings.create(
        model="text-embedding-v4",
        input=text
    )
    return [item.embedding for item in response.data]


def get_collection():
    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )
    return chroma_client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

def store_chunks(collection, chunks: list[dict]):
   batch_size = 10
   for start in range(0, len(chunks), batch_size):
       batch = chunks[start:start + batch_size]
       ids = [
            f'{item["filename"]}#chunk-{item["chunk_id"]}'
            for item in batch
        ]
       
       texts = [item["chunk"] for item in batch]
       embeddings = create_embedding(texts)
       metadata = [{"filename": item["filename"], "chunk_id": item["chunk_id"]} for item in batch]
       collection.upsert(
           ids=ids,
           documents=texts,
           embeddings=embeddings,
           metadatas=metadata
       )

def search_knowledge(collection, query: str, top_k: int = 3) -> list[dict]:
    """
    Search the knowledge base for relevant chunks based on the query.
    Returns a list of dictionaries with 'filename', 'chunk_id', and 'content'.
    """
    query_embedding = create_embedding([query])[0]
    response = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    results = []
    for document, metadata, distance in zip(
        response["documents"][0],
        response["metadatas"][0],
        response["distances"][0]
    ):
        if distance > MAX_DISTANCE:
            continue  # Skip results that are too far away
        results.append({
            "filename": metadata["filename"],
            "chunk_id": metadata["chunk_id"],
            "text": document,
            "distance": distance
        })
    return results
    

def answer_with_rag(question: str, results: list[dict]) -> str:
    context = "\n\n".join(
        (
            f"[来源：{item['filename']}#chunk-{item['chunk_id']}]\n"
            f"{item['text']}"
        )
        for item in results
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是知识库助手。只能根据提供的资料回答。"
                    "资料不足时明确说“资料中未找到依据”。"
                    "回答末尾列出实际使用的来源。"
                ),
            },
            {
                "role": "user",
                "content": f"资料：\n{context}\n\n问题：{question}",
            },
        ],
    )

    return response.choices[0].message.content or "未收到模型回答。"



documents = load_documents()
chunks = split_documents(documents)

print(f"文档数量：{len(documents)}")
print(f"切分后 chunk 数量：{len(chunks)}")


collection = get_collection()

if collection.count() == 0:
    store_chunks(collection, chunks)

print(f"当前知识库 chunk 数量：{collection.count()}")

question = input("请输入查询问题：").strip()
results = search_knowledge(collection, question)
if not question:
    print("问题不能为空。")
    raise SystemExit

for item in results:
    print(f"\n来源：{item['filename']}#chunk-{item['chunk_id']}")
    print(f"距离：{item['distance']:.4f}")
    print(item["text"])

answer = answer_with_rag(question, results)

print("\n模型回答：")
print(answer)