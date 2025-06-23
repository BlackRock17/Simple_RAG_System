# Interactive RAG Learning System
# Designed for interview preparation - понятно обяснение на всяка стъпка

import os
import json
from typing import List, Dict, Any
from datetime import datetime

# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.schema import BaseRetriever

# For similarity calculations
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class InteractiveRAGSystem:
    """
    Interactive RAG System за учебни цели

    Показва как работи всяка стъпка:
    1. Document Processing (как се обработват документите)
    2. Embedding Creation (как се създават векторите)
    3. Vector Storage (как се съхраняват в база данни)
    4. Similarity Search (как се търси сходство)
    5. Context Augmentation (как се добавя контекст)
    6. Response Generation (как се генерира отговор)
    """

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.documents = []
        self.document_chunks = []
        self.embeddings_model = None
        self.vectorstore = None

        if self.verbose:
            print("🚀 Инициализиране на RAG система...")
            print("=" * 60)

    def step1_setup_embeddings(self):
        """
        СТЪПКА 1: Настройка на embedding модел

        Embeddings са начинът да превръщаме текст в числа (вектори),
        за да може компютърът да разбира семантиката
        """
        if self.verbose:
            print("📊 СТЪПКА 1: Настройка на Embedding модел")
            print("-" * 40)
            print("Embedding моделът превръща текст в числови вектори")
            print("Използваме 'sentence-transformers/all-MiniLM-L6-v2' - малък но ефективен модел")

        # Малък и бърз embedding модел
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},  # Използваме CPU за по-бърза настройка
            encode_kwargs={'normalize_embeddings': False}
        )

        if self.verbose:
            print("✅ Embedding модел готов!")
            print()

    def step2_load_documents(self, texts: List[str]):
        """
        СТЪПКА 2: Зареждане и обработка на документи

        Тук разделяме документите на по-малки части (chunks),
        за да може да намираме по-точна информация
        """
        if self.verbose:
            print("📚 СТЪПКА 2: Обработка на документи")
            print("-" * 40)
            print(f"Зареждаме {len(texts)} документа...")

        # Текст сплитър - разделя големи текстове на по-малки части
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,  # Размер на всеки chunk
            chunk_overlap=50,  # Припокриване между chunks
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

        self.documents = texts
        self.document_chunks = []

        for doc_id, text in enumerate(texts):
            # Разделяме документа на chunks
            chunks = text_splitter.split_text(text)

            for chunk_id, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "source": f"document_{doc_id}",
                        "total_chunks": len(chunks)
                    }
                )
                self.document_chunks.append(doc)

        if self.verbose:
            print(f"✅ Създадени {len(self.document_chunks)} chunks от {len(texts)} документа")
            print("Примерен chunk:")
            print(f"  Content: {self.document_chunks[0].page_content[:100]}...")
            print(f"  Metadata: {self.document_chunks[0].metadata}")
            print()

    def step3_create_vectorstore(self):
        """
        СТЪПКА 3: Създаване на vector database

        Тук превръщаме всички текстове в вектори и ги съхраняваме
        в ChromaDB за бърза търсене
        """
        if self.verbose:
            print("🗄️ СТЪПКА 3: Създаване на Vector Database")
            print("-" * 40)
            print("Превръщаме всички chunks в embedding вектори...")

        # Създаваме ChromaDB vectorstore
        self.vectorstore = Chroma.from_documents(
            documents=self.document_chunks,
            embedding=self.embeddings_model,
            persist_directory=None  # Не записваме на диска
        )

        if self.verbose:
            print("✅ Vector database създадена!")
            print(f"Съхранени {len(self.document_chunks)} embedding вектора")
            print()

    def step4_similarity_search_explained(self, query: str, k: int = 3):
        """
        СТЪПКА 4: Similarity Search с обяснение

        Показва как работи търсенето на сходство стъпка по стъпка
        """
        if self.verbose:
            print("🔍 СТЪПКА 4: Similarity Search")
            print("-" * 40)
            print(f"Търсим за: '{query}'")
            print()

        # Извършваме similarity search
        results = self.vectorstore.similarity_search_with_score(query, k=k)

        if self.verbose:
            print("Процес на търсене:")
            print("1. Преобразуваме query-то в embedding вектор")
            print("2. Сравняваме с всички съхранени вектори")
            print("3. Връщаме най-сходните резултати")
            print()

            print("Резултати:")
            for i, (doc, score) in enumerate(results, 1):
                print(f"{i}. Score: {score:.4f}")
                print(f"   Source: {doc.metadata['source']}")
                print(f"   Content: {doc.page_content[:100]}...")
                print()

        return [doc for doc, score in results]

    def step5_demonstrate_embeddings(self, query: str):
        """
        СТЪПКА 5: Демонстрация на embeddings

        Показва как изглеждат векторите и как се изчислява сходството
        """
        if self.verbose:
            print("🧮 СТЪПКА 5: Демонстрация на Embeddings")
            print("-" * 40)

        # Създаваме embedding на query-то
        query_embedding = self.embeddings_model.embed_query(query)

        if self.verbose:
            print(f"Query: '{query}'")
            print(f"Embedding размер: {len(query_embedding)} числа")
            print(f"Първите 5 числа: {query_embedding[:5]}")
            print()

            # Показваме как се изчислява сходство с първия document
            if self.document_chunks:
                first_doc_embedding = self.embeddings_model.embed_query(
                    self.document_chunks[0].page_content
                )

                # Cosine similarity
                similarity = cosine_similarity(
                    [query_embedding],
                    [first_doc_embedding]
                )[0][0]

                print(f"Сходство с първия document: {similarity:.4f}")
                print(f"Document: {self.document_chunks[0].page_content[:100]}...")
                print()

    def step6_context_augmentation(self, query: str, relevant_docs: List[Document]):
        """
        СТЪПКА 6: Context Augmentation

        Показва как се комбинира query-то с намерените документи
        """
        if self.verbose:
            print("🔗 СТЪПКА 6: Context Augmentation")
            print("-" * 40)

        # Създаваме контекст от намерените документи
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Създаваме augmented prompt
        augmented_prompt = f"""
Контекст от документи:
{context}

Въпрос: {query}

Моля, отговори на въпроса въз основа на предоставения контекст.
"""

        if self.verbose:
            print("Комбиниране на query + context:")
            print(f"Оригинален въпрос: {query}")
            print(f"Добавен контекст от {len(relevant_docs)} документа")
            print(f"Общ размер на prompt: {len(augmented_prompt)} символа")
            print()
            print("Augmented Prompt:")
            print(augmented_prompt[:300] + "..." if len(augmented_prompt) > 300 else augmented_prompt)
            print()

        return augmented_prompt

    def step7_generate_response(self, augmented_prompt: str):
        """
        СТЪПКА 7: Response Generation (симулация)

        В реална система тук би се използвал LLM като GPT, Claude, etc.
        За демо целите ще симулираме отговор
        """
        if self.verbose:
            print("🤖 СТЪПКА 7: Response Generation")
            print("-" * 40)
            print("В реална система тук би се използвал LLM (GPT, Claude, Llama, etc.)")
            print("За демонстрация ще симулираме 'умен' отговор")
            print()

        # Симулиран отговор
        doc_count = len(augmented_prompt.split('Въпрос:')[0].split('\n\n'))
        response = f"""
                    [СИМУЛИРАН LLM ОТГОВОР]
                    
                    Въз основа на предоставената информация в контекста, мога да отговоря на въпроса.
                    
                    Информацията е взета от {doc_count} документа 
                    и предоставя релевантни детайли.
                    
                    (В реална система тук би се генерирал точен отговор от LLM модел)
                    """

        if self.verbose:
            print("Генериран отговор:")
            print(response)
            print()

        return response

    def full_rag_pipeline(self, query: str):
        """
        Пълен RAG pipeline със всички стъпки
        """
        print("🎯 ПЪЛЕН RAG PIPELINE")
        print("=" * 60)
        print(f"Query: '{query}'")
        print("=" * 60)
        print()

        # Стъпка по стъпка
        relevant_docs = self.step4_similarity_search_explained(query)
        self.step5_demonstrate_embeddings(query)
        augmented_prompt = self.step6_context_augmentation(query, relevant_docs)
        response = self.step7_generate_response(augmented_prompt)

        return {
            "query": query,
            "relevant_documents": [doc.page_content for doc in relevant_docs],
            "augmented_prompt": augmented_prompt,
            "response": response
        }


def interactive_learning_demo():
    """
    Интерактивно демо за учене на RAG концепции
    """
    print("🎓 ИНТЕРАКТИВНО RAG ДЕМО ЗА УЧЕНЕ")
    print("=" * 60)
    print("Това демо ще ти покаже как работи RAG стъпка по стъпка")
    print("Всяка стъпка ще бъде обяснена подробно")
    print("=" * 60)
    print()

    # Създаваме RAG система
    rag = InteractiveRAGSystem(verbose=True)

    # Примерни документи за AI/ML интервю
    sample_documents = [
        """
        Retrieval-Augmented Generation (RAG) е техника за подобряване на Large Language Models.
        RAG работи в три основни стъпки: първо търси релевантни документи в база данни,
        после добавя тези документи като контекст към въпроса, и накрая генерира отговор
        въз основа на предоставената информация. Това позволява на модела да използва 
        актуална и специфична информация, която не е била част от тренировъчните му данни.
        """,

        """
        Vector databases са специализирани бази данни за съхранение на embedding вектори.
        Те позволяват бързо търсене на сходство между векторите чрез алгоритми като
        cosine similarity, dot product или euclidean distance. Популярни vector databases
        включват ChromaDB, Pinecone, Weaviate и Milvus. Тези бази данни са ключови за
        RAG системи, защото позволяват ефективно намиране на най-релевантните документи.
        """,

        """
        LangChain е framework за разработка на приложения с Large Language Models.
        Той предоставя готови компоненти за создаване на chains, agents и tools.
        LangChain поддържа множество LLM доставчици като OpenAI, Anthropic, Hugging Face.
        Основните компоненти включват: Prompts, Chains, Agents, Memory, Tools и Callbacks.
        Това прави разработката на AI приложения по-лесна и по-структурирана.
        """,

        """
        Embeddings са числови представления на текст, които улавят семантичното значение.
        Те се създават от специализирани модели като sentence-transformers. Качествени
        embeddings позволяват на компютъра да разбира, че "кола" и "автомобил" са сходни,
        въпреки че са различни думи. За създаване на embeddings се използват модели като
        BERT, Sentence-BERT, OpenAI embeddings или Hugging Face transformers.
        """,

        """
        Prompt Engineering е изкуството на създаване на ефективни prompts за LLM модели.
        Техники включват few-shot learning (даване на примери), chain-of-thought reasoning
        (стъпка по стъпка мислене), и role-based prompting (задаване на роля).
        Добрите prompts могат значително да подобрят качеството на отговорите от LLM.
        Prompt engineering е критична умения за работа с AI модели в продукция.
        """
    ]

    # Стъпка по стъпка настройка
    rag.step1_setup_embeddings()
    rag.step2_load_documents(sample_documents)
    rag.step3_create_vectorstore()

    # Примерни въпроси за демонстрация
    demo_questions = [
        "Как работи RAG?",
        "Какво представляват vector databases?",
        "Какви са основните компоненти на LangChain?",
        "Какво са embeddings?",
        "Какви техники се използват в prompt engineering?"
    ]

    print("🎯 ДЕМОНСТРАЦИЯ С ПРИМЕРНИ ВЪПРОСИ")
    print("=" * 60)

    for i, question in enumerate(demo_questions, 1):
        print(f"\n📝 ДЕМО ВЪПРОС {i}: {question}")
        print("-" * 80)

        # Показваме само retrieval частта за краткост
        relevant_docs = rag.step4_similarity_search_explained(question, k=2)

        # Малка пауза за четене
        input("Натисни Enter за следващия въпрос...")

    print("\n🎯 ИНТЕРАКТИВНА ЧАСТ")
    print("=" * 60)
    print("Сега можеш да задаваш свои въпроси!")
    print("Напиши 'exit' за изход")
    print()

    while True:
        user_question = input("❓ Твоят въпрос: ").strip()

        if user_question.lower() in ['exit', 'quit', 'изход']:
            print("👋 Успех!")
            break

        if not user_question:
            continue

        try:
            result = rag.full_rag_pipeline(user_question)
            print("\n" + "=" * 80)
            input("Натисни Enter за следващ въпрос...")
        except Exception as e:
            print(f"❌ Грешка: {e}")


if __name__ == "__main__":
    # Стартираме интерактивното демо
    interactive_learning_demo()
