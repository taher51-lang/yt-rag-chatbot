from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpointEmbeddings,HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from dotenv import load_dotenv
import streamlit as st
load_dotenv()
if "vectorStore" not in st.session_state:
    st.session_state.vectorStore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
with st.sidebar:
    url = st.text_input("Enter the URL of the youtube video")
    if st.button("Process Video"):
        with st.spinner("Processing..."):
            # Phase 1: Scraping
            video_id = url.split("v=")[-1]
            yt_api = YouTubeTranscriptApi()
            transcriptList = yt_api.list(video_id)
            try:
                transcript = transcriptList.find_transcript(["en"])
            except:
                transcript = next(iter(transcriptList))            
            fetchObjects = transcript.fetch()
            text = ""
            for snippet in fetchObjects:
                text +=snippet.text + " "
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunk_list = splitter.create_documents([text])            
            embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)
            # Storing in Session State
            st.session_state.vectorStore = FAISS.from_documents(chunk_list, embedding=embeddings,distance_strategy="COSINE")
            st.success("Video ready!")
            st.rerun() # Refreshing to clear the landing page

if st.session_state.vectorStore is None:
    st.markdown("# 🎥 YouTube RAG\nPaste a URL to get started.")
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="font-size: 3rem; color: #FF4B4B;">📺 VideoMind AI</h1>
        <h3 style="color: #555;">Turn YouTube transcripts into your personal knowledge base.</h3>
        <p style="font-size: 1.1rem; color: #888;">Powered by <b>Mistral 7B</b> & <b>Vector Search</b></p>
    </div>
    <hr>
""", unsafe_allow_html=True)
else:
        st.markdown("### Video Player")
        st.video(url)
        st.markdown("---")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        if prompt := st.chat_input("Ask a question"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            llm_ = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            task="text-generation"
            )
            retriever_ = st.session_state.vectorStore.as_retriever(search_type="similarity",search_kwargs={"k":4})
            MISTRAL_QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""[INST] You are an AI assistant. Rewrite the following user question 
            into exactly 2 different versions to help find relevant video transcript segments.
            Use specific technical keywords from the context.
            Original question: {question} [/INST]""",
            )
            multiRetriever = MultiQueryRetriever.from_llm(
                retriever=retriever_,
                llm = ChatHuggingFace(
                    llm = llm_
                ),
                prompt=MISTRAL_QUERY_PROMPT
            )
            prompt = PromptTemplate(
            template='''You are a helpful AI assistant, answer the question {question} on the basis of following context
             {context} . If the question are not related to the context, say that the video does not cover this topic''',
            input_variables=["context","question"]
                )
            def format(retrivedDocs):
                cn = "\n\n".join(doc.page_content for doc in retrivedDocs)
                return cn
            questionEnhancer = PromptTemplate(
                template='''You are Question Enhancer AI, Enhace the following question into meaningful query with proper grammar and semantic meaning,
                {question}''',
                input_variables=["question"]
            )
            llm = ChatHuggingFace(llm=llm_)
            parser = StrOutputParser()
            parallelChain = RunnableParallel({
            "context": multiRetriever | RunnableLambda(format),
            "question":questionEnhancer|llm|parser
                }
            )
            main_chain = parallelChain | prompt | llm | parser 
            result = main_chain.invoke(prompt)
            # result = chain.invoke(prompt)
            st.session_state.messages.append({"role": "assistant", "content":result})
            st.rerun()
