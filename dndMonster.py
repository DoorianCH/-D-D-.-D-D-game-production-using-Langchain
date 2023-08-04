# 라이브러리 불러오기
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.docstore.document import Document
import os
import re
import gradio as gr

os.environ["OPENAI_API_KEY"] = "your api key" # 환경변수에 OPENAI_API_KEY를 설정

def openAiGPT(message, chat_history) :
    
    # pdf 불러오기
    loader = PyPDFLoader("dndCreatear.pdf")     # pdf 이름 입력 ex dndCreater.pdf
    documents = loader.load()
    documents

    # pdf 내용 전처리하기
    documents_pro = []      # 저장 리스트

    for page in documents:
        text = page.page_content
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())    # "116\n제12장 -> 116 제12장
        text = re.sub(r"\n\s*\n", "\n\n", text)     # 60ft\n \n근력\n \n민첩 -> 60ft\n\n근력\n\n민첩
        text = re.sub(r"\n\s*", " ", text)      # 비행 60ft\n\n근력\n\n민첩 -> 비행 60ft 근력 민첩
        documents_pro.append(text)      # documents_pro 에 추가

    # chunk splite, list -> document
    doc_chunks = []

    for line in documents_pro:      # documents_pro의 각 문서를 반복
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,    # 최대청크의 최대 길이를 3000자로 제한(문서에 따라 다르게 기준을 잡음)
            separators=["\n\n", "\n", ".", ""],     #  텍스트를 청크로 분할하는데 기준이 되는 문자 목록
            chunk_overlap=0,    # 청크 사이의 중첩, 현재는 없음
        )
        chunks = text_splitter.split_text(line)
        for i, chunk in enumerate(chunks):      # 분리된 청크를 document형식으로 변환
            doc = Document(
                page_content=chunk, metadata={"page": i, "source": "dndCreatear.pdf"}   # 페이지 메타데이터 정보, 정보의 근원이 되는 pdf
            )
            doc_chunks.append(doc)      # doc_chunks 리스트에 추가

    
    # ChromaDB에 임베딩
    embeddings = OpenAIEmbeddings()     
    vector_store = Chroma.from_documents(doc_chunks, embeddings)    # doc_chunks의 정보를 OpenAIEmbeddings()로 임베딩, doc_chunks의 내용을 수행
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})       # 검색을 진행할 때, kwargs 갯수 설정(현재 2, 0이하인 경우 수행 x)


    from langchain.prompts.chat import (
        ChatPromptTemplate,      
        SystemMessagePromptTemplate,       
        HumanMessagePromptTemplate,        
    )
    
    ### 프롬프트 조절
    system_template="""To answer the question at the end, use the following context. 
    If you don't know the answer, just say you don't know and don't try to make up an answer.
    you tell me the exact information and figures of the monster.
    I want you to act as Monster expert.

    you only answer in Korean

    {summaries}

    """

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),     # 시스템 사전 설정
        HumanMessagePromptTemplate.from_template("{question}")      # 내 질문 설정
    ]

    prompt = ChatPromptTemplate.from_messages(messages)     # prompt 변수에 저장

    chain_type_kwargs = {"prompt": prompt}

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)  # 챗지피티 모델 설정(현재, gpt-3.5-turbo)

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever = retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        reduce_k_below_max_tokens=True  # 토큰 제한 해제
)
    
    # myAsk = """겨울 늑대에 대해 알려줘""" # 질문 내용
    result = chain(message)
    gpt_message = result['answer']
    print(gpt_message)

    chat_history.append((message, gpt_message))
    return "", chat_history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])
    
    msg.submit(openAiGPT, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()