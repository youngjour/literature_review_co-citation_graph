import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def load_vector_store(vectorstore_path, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Load the FAISS vector store.
    """
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vector_store = FAISS.load_local(vectorstore_path, embedding_model)
    print("Vector store loaded successfully.")
    return vector_store

def create_llama_pipeline(llm_model_name="decapoda-research/llama-7b-hf"):
    """
    Load the LLaMA model and tokenizer to create a pipeline for answering queries.
    """
    print(f"Loading LLaMA model: {llm_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    model = AutoModelForCausalLM.from_pretrained(llm_model_name, device_map="auto", torch_dtype="auto")
    llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    print("LLaMA model loaded successfully.")
    return llm_pipeline

def create_rag_pipeline(vector_store, llm_pipeline):
    """
    Create a Retrieval-Augmented Generation (RAG) pipeline.
    """
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_pipeline,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )
    print("RAG pipeline created successfully.")
    return qa_chain

def main(vectorstore_path, llm_model_name="decapoda-research/llama-7b-hf", embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Main function to create and run the LLaMA-based question-answering system.
    """
    # Load FAISS vector store
    vector_store = load_vector_store(vectorstore_path, embedding_model_name)

    # Load LLaMA model pipeline
    llm_pipeline = create_llama_pipeline(llm_model_name)

    # Create RAG pipeline
    rag_pipeline = create_rag_pipeline(vector_store, llm_pipeline)

    # Interactive Q&A
    print("\nAsk your questions! Type 'exit' to quit.")
    while True:
        query = input("\nYour Query: ")
        if query.lower() == "exit":
            print("Exiting the system. Goodbye!")
            break

        try:
            result = rag_pipeline.run(query)
            answer = result['answer']
            sources = result['source_documents']
            print(f"\nAnswer: {answer}")
            print("\nSources:")
            for doc in sources:
                print(f"- {doc.metadata.get('source', 'Unknown Source')}")
        except Exception as e:
            print(f"Error processing query: {e}")

if __name__ == "__main__":
    VECTORSTORE_PATH = "vector_store"  # Directory containing FAISS vector store
    LLM_MODEL_NAME = "decapoda-research/llama-7b-hf"  # Replace with your LLaMA model
    main(VECTORSTORE_PATH, llm_model_name=LLM_MODEL_NAME)
