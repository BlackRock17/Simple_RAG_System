# AI Interview Demo - Simple RAG System
# Created for Metasim Junior AI Engineer Interview

import os
from typing import List, Dict
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.schema import BaseRetriever
import numpy as np


class SimpleRAGSystem:
    """
    Simple RAG (Retrieval-Augmented Generation) System

    RAG works in 3 steps:
    1. RETRIEVE - find relevant documents through similarity search
    2. AUGMENT - add context to our query
    3. GENERATE - generate response (we'll simulate this here)
    """

    def __init__(self):
        print("🚀 Initializing RAG system...")

        # Using HuggingFace embeddings (free, local)
        # Embeddings convert text into numerical vectors
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # ChromaDB is a vector database for storing embeddings
        self.vectorstore = None

        # Text splitter divides large documents into smaller chunks
        self.text_splitter = CharacterTextSplitter(
            chunk_size=200,  # Smaller chunks for better precision
            chunk_overlap=20,  # Smaller overlap
            separator=". "  # Split on sentences
        )

        print("✅ RAG system ready!")

    def load_documents(self, texts: List[str]) -> None:
        """
        Loads documents into vector database

        The process is:
        1. Split texts into chunks
        2. Create embeddings for each chunk
        3. Store in ChromaDB for searching
        """
        print(f"📚 Loading {len(texts)} documents...")

        # Create Document objects with unique IDs
        documents = []
        for i, text in enumerate(texts):
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)

            for j, chunk in enumerate(chunks):
                documents.append(Document(
                    page_content=chunk,
                    metadata={
                        "doc_id": i,
                        "chunk_id": j,
                        "source": f"document_{i}_chunk_{j}"
                    }
                ))

        # Create vector store with embeddings
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=None  # Don't persist to avoid cache issues
        )

        print(f"✅ Loaded {len(documents)} chunks into vector database")

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        Searches for the most relevant documents for a given query

        How it works:
        1. Convert query to embedding
        2. Calculate cosine similarity with all documents
        3. Return top-k most relevant ones
        """
        if not self.vectorstore:
            raise ValueError("No documents loaded! Use load_documents() first.")

        print(f"🔍 Searching for: '{query}'")

        # Similarity search in ChromaDB
        results = self.vectorstore.similarity_search(query, k=k)

        print(f"📄 Found {len(results)} relevant documents")
        return results

    def retrieve_and_rank(self, query: str, k: int = 3) -> List[Dict]:
        """
        Searches and ranks documents with additional information
        """
        results = self.similarity_search(query, k=k)

        # Add additional information for each result
        ranked_results = []
        for i, doc in enumerate(results):
            ranked_results.append({
                "rank": i + 1,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": f"High (top {i + 1})"  # Simulated score
            })

        return ranked_results

    def generate_augmented_response(self, query: str, context_docs: List[Document]) -> str:
        """
        Simulates AI response generation with context

        In real RAG systems, this would use an LLM (GPT, Claude, etc.)
        For the demo, we'll simulate an intelligent response
        """
        context = "\n\n".join([doc.page_content for doc in context_docs])

        # Simulated AI response (in reality, would use LLM)
        response = f"""
🤖 RAG GENERATED RESPONSE:

Based on the information found in the documents, here's an answer to your question: "{query}"

CONTEXT FROM DOCUMENTS:
{context[:500]}...

CONCLUSION:
Using Retrieval-Augmented Generation, I found relevant information 
and can provide a more accurate and informed response than using only my pre-trained knowledge.
        """

        return response

    def full_rag_pipeline(self, query: str) -> Dict:
        """
        Complete RAG pipeline: Retrieve → Augment → Generate
        """
        print(f"\n🎯 Starting RAG pipeline for: '{query}'")
        print("=" * 60)

        # STEP 1: RETRIEVE
        print("1️⃣ RETRIEVE - Searching for relevant documents...")
        relevant_docs = self.similarity_search(query, k=3)

        # STEP 2: AUGMENT
        print("2️⃣ AUGMENT - Adding context...")
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # STEP 3: GENERATE
        print("3️⃣ GENERATE - Generating response...")
        response = self.generate_augmented_response(query, relevant_docs)

        return {
            "query": query,
            "retrieved_docs": [doc.page_content for doc in relevant_docs],
            "context": context,
            "response": response,
            "num_docs_used": len(relevant_docs)
        }


def demo_ai_knowledge_base():
    """
    Демо функция за интервюто - показва RAG система с AI знания
    """
    print("🎯 AI INTERVIEW DEMO - RAG SYSTEM")
    print("=" * 50)

    # Създаваме RAG системата
    rag = SimpleRAGSystem()

    # Sample AI knowledge base documents (relevant for the interview)
    ai_documents = [
        """
        Large Language Models (LLMs) are AI models trained on massive amounts of text data.
        They can generate text, answer questions, and perform various natural language processing tasks.
        Examples of LLMs include GPT-4, Claude, PaLM, and Llama. LLMs use transformer architecture
        and attention mechanisms to understand and generate human-like text.
        """,

        """
        RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval
        with text generation. RAG works in three steps: First, Retrieve relevant documents from a database.
        Second, Augment the query by adding this information as context. Third, Generate an answer
        using an LLM based on the provided context. RAG helps LLMs provide more accurate and up-to-date responses.
        """,

        """
        Vector Databases are specialized databases for storing embeddings and performing similarity search.
        Embeddings are numerical representations of text that capture semantic meaning.
        Popular vector databases include ChromaDB, Pinecone, Weaviate, and Milvus.
        They enable similarity search using cosine similarity, dot product, or Euclidean distance.
        """,

        """
        LangChain is a framework for developing applications with Large Language Models.
        LangChain provides tools for prompt management, chains for sequential operations,
        and agents for autonomous behavior. It supports multiple LLM providers like OpenAI, Anthropic, and Hugging Face.
        LangChain simplifies building complex AI applications through a modular approach.
        """,

        """
        Prompt Engineering is the art of crafting effective prompts for LLMs to get better results.
        Techniques include few-shot learning, chain-of-thought reasoning, and role-based prompting.
        Good prompts can significantly improve the quality and accuracy of LLM responses.
        Prompt engineering is a crucial skill for working with LLMs in production environments.
        """
    ]

    # Зареждаме документите
    rag.load_documents(ai_documents)

    # Demo questions for the interview
    demo_questions = [
        "What is RAG and how does it work?",
        "What are the advantages of vector databases?",
        "How does LangChain help when working with LLMs?",
        "What is prompt engineering?"
    ]

    print("\n🎤 INTERVIEW DEMONSTRATION:")
    print("=" * 40)

    for question in demo_questions:
        print(f"\n❓ Question: {question}")
        print("-" * 50)

        # Show only the retrieve step for brevity
        relevant_docs = rag.similarity_search(question, k=3)

        print("📄 Found relevant documents:")
        for i, doc in enumerate(relevant_docs, 1):
            source = doc.metadata.get('source', f'doc_{i}')
            preview = doc.page_content[:80].replace('\n', ' ')
            print(f"{i}. [{source}] {preview}...")

        print("\n" + "=" * 60)

    print("\n🚀 Demo complete! This shows how RAG finds relevant information for each question.")

    return rag


def interactive_demo():
    """
    Interactive demo for testing
    """
    print("🎯 INTERACTIVE RAG DEMO")
    print("Type 'quit' to exit\n")

    # Load the data
    rag = demo_ai_knowledge_base()

    while True:
        query = input("\n❓ Your question: ").strip()

        if query.lower() in ['quit', 'exit']:
            print("👋 Goodbye!")
            break

        if not query:
            continue

        try:
            # Full RAG pipeline
            result = rag.full_rag_pipeline(query)
            print(result["response"])

        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("🎯 METASIM AI INTERVIEW DEMO")
    print("=" * 30)
    print("This code demonstrates:")
    print("✅ RAG (Retrieval-Augmented Generation)")
    print("✅ Vector Databases (ChromaDB)")
    print("✅ Embeddings (HuggingFace)")
    print("✅ LangChain integration")
    print("✅ Similarity Search")
    print("\n" + "=" * 30)

    # Избор на режим
    print("\nChoose mode:")
    print("1. Automatic demo (for interview)")
    print("2. Interactive mode")

    choice = input("\nYour choice (1 or 2): ").strip()

    if choice == "1":
        demo_ai_knowledge_base()
    elif choice == "2":
        interactive_demo()
    else:
        print("Starting automatic demo...")
        demo_ai_knowledge_base()
