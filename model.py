from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import LlamaCpp
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from loguru import logger
from dotenv import load_dotenv
import os

load_dotenv()
DB_FAISS_PATH = "vectorstore/faiss"

prompt_template = """Use the following pieces of information to answer the question.
If you cannot formulate an answer based on the given context,
just say that you don't know, don't try to make up an answer under any circumstance.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful and polite answer:
"""


# Define function to set custom prompt
def set_custom_prompt():
    logger.debug("Into def set_custom_prompt")
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    logger.info(f"prompt: {prompt}")
    return prompt


# Load the LLM model
def load_llm(useLocalLLM):
    logger.debug("Into def load_llm")
    if useLocalLLM:
        llm = LlamaCpp(
            model_path=os.getenv("MODEL_PATH"),
            max_tokens=2048,  # maximum number of new tokens to be generated. It controls length of response.
            n_ctx=2048,
            temperature=0,
            n_gpu_layers=4,
        )
    # option to use openAI model instead, API key required
    else:
        llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-4",
            temperature=0,
        )

    return llm


# Define a function to RetrievalQA chain using the specified LLM, prompt, and vector store (db).
def retrieval_qa_chain(llm, prompt, db):
    """
    llm: large language model used for text generation.
    prompt: A structured prompt template to guide the generation process.
    db: Vector store (database) instance, used for retrieval.
    return:
        Returns the configured RetrievalQA chain.
    """
    logger.debug("Into def retrieval_qa_chain")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 'stuff' documents to pass to llm
        retriever=db.as_retriever(search_kwargs={"k": 2}),  # no.of results
        return_source_documents=True, 
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain


# QA Model Function
def qa_bot():
    logger.debug("Into def qa_bot")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", 
        model_kwargs={"device": "cpu"},
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings) 
    llm = load_llm(useLocalLLM=True)
    qa_prompt = set_custom_prompt()  
    qa = retrieval_qa_chain(
        llm, qa_prompt, db
    )
    logger.info(f"final_result: {qa}")
    return qa

# output function
def final_result(query):
    logger.debug("Into def final_result")
    qa_result = qa_bot()  # Calling definition func_qa_bot
    response = qa_result({"query": query})
    logger.info(f"func_final_result: {response}")
    return response