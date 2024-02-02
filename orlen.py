# Vector database
from langchain_community.vectorstores.chroma import Chroma

# Embeddings + Cache do nich
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_openai.embeddings.base import OpenAIEmbeddings

# Document loader
from langchain_community.document_loaders.pdf import PDFMinerLoader

# Spliting text
from langchain.text_splitter import CharacterTextSplitter

# Compression of retireved material
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# LLMs
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.cache import InMemoryCache

# Tracking token usage
from langchain_community.callbacks import get_openai_callback

# Checking the database presence
import os

# LLM here
llm = ChatOpenAI(verbose=True,
             temperature=0.0,
             max_tokens=1000,
             model="gpt-4-turbo-preview")
cache = InMemoryCache()
memory = ConversationBufferMemory(llm=llm, memory_key='chat_history', return_messages=True, cache=cache)


# retrieval compression & chain setup
def compressor_setup(database_connection):
    compressor = LLMChainExtractor.from_llm(llm=llm)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=database_connection.as_retriever())
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=compression_retriever, memory=memory)


def text_split_and_load():
    text_split = CharacterTextSplitter()
    loader = PDFMinerLoader(r"./orlen_raport.pdf")
    document = loader.load_and_split(text_splitter=text_split, verbose=False)
    return document


def database_operations():
    cwd = os.getcwd().__str__()
    os.chdir(cwd)
    store = LocalFileStore("./cache/")
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, store, namespace=embeddings.model)
    if "orlen.db" not in os.listdir():
        embedding_database = Chroma.from_documents(documents=text_split_and_load(), embedding=cached_embeddings, persist_directory="./orlen.db")
        embedding_database.persist()
    else:
        pass
    return Chroma(persist_directory="./orlen.db", embedding_function=cached_embeddings)


def ask_and_receive(query):
    combined_chain = compressor_setup(database_connection=database_connection)
    return combined_chain(f"{query}")['answer']


if __name__ == "__main__":
    
    database_connection = database_operations()

    with get_openai_callback() as cb:
        while True:
            print("\nTo exit, write 'exit'\n")
            query = input("Ask us anythong about Orlen's last report:\n")
            if query == "Exit".lower():
                print("\nVery well, farewell\nSome technical info for you:\n")
                print(cb)
                break
            else:
                answer = ask_and_receive(query)
                print(answer)
