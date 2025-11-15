import os
import sys
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
#from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain_classic.chains.retrieval_qa.base import RetrievalQA


def main():
    # Loading the speech text
    loader = TextLoader('speech.txt')
    documents = loader.load()

    #  Splitting text into manageable chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=250,
        chunk_overlap=30
    )
    docs = text_splitter.split_documents(documents)

    # Creating embeddings and store in ChromaDB vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(docs, embeddings)

    # Set up Ollama LLM (Mistral 7B)
    #llm = Ollama(model="Mistral 7B") due to low ram i have used llama2

    llm = OllamaLLM(model="llama2") # due to low ram i have used llama2


    # Created RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )


    print("AmbedkarGPT is ready! Ask any question about the speech.")
    while True:
        query = input("\nYour question (type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        result = qa_chain.invoke(query)
        print("\nAnswer:\n", result["result"])
        print("\nRelevant Context:\n", "\n".join([doc.page_content for doc in result["source_documents"]]))

if __name__ == "__main__":
    main()



