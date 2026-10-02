import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

class RAGEngine:
    def __init__(self):
        # Using a small and efficient Hugging Face embedding model locally
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=80
        )
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            collection_name=f"rag_session_{uuid.uuid4().hex}"
        )
        
    def index_documents(self, documents: list[dict]):
        """
        documents format: [{'filename': str, 'content': str}]
        """
        docs_to_index = []
        for doc_info in documents:
            filename = doc_info['filename']
            content = doc_info['content']
            
            chunks = self.text_splitter.split_text(content)
            for i, chunk in enumerate(chunks):
                # Hybrid context assembly with source attribution
                context_chunk = f"[Source: {filename}, Page/Section: {i+1}]:\n{chunk}"
                docs_to_index.append(
                    Document(
                        page_content=context_chunk,
                        metadata={"source": filename, "section": i+1}
                    )
                )
                
        if docs_to_index:
            self.vectorstore.add_documents(docs_to_index)
            return True
        return False

    def retrieve(self, query: str, k: int = 5) -> str:
        results = self.vectorstore.similarity_search(query, k=k)
        if not results:
            return ""
            
        context_parts = []
        for doc in results:
            context_parts.append(doc.page_content)
            
        return "\n\n".join(context_parts)
