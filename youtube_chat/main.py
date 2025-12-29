from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpointEmbeddings
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv

# Phase 1 of RAG Implementation(Aquiring Knowledge Source)
yt_api = YouTubeTranscriptApi()
transcriptList = yt_api.list(video_id="3pLT-InFaIY")
try:
    transcript = transcriptList.find_transcript(["en"])
except:
    transcript = next(iter(transcriptList))
    # transcript = first_available.translate("en")
fetchObjects = transcript.fetch()
text = ""
for snippet in fetchObjects:
    text+=snippet.text
# Phase 2 Text Splitting using Recursive Text Splitter
splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 200)
chunk_list = splitter.create_documents([text])
load_dotenv()
llm = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(chunk_list,embedding=llm)
# Phase 4  Retriever
retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":6})
# Injecting the context and query into the llm
multiRetriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash-lite"
)
)
prompt = PromptTemplate(
    template='''You are a Helpful assistant
    ONLY from the given Context : {context}, answer the following questions:{question}
    Give a brief explanation about the topic of question while staying relevant to context,
    If the question is not related to context, SAY I DONT KNOW. ''',
    input_variables=["context","question"]
)
def format(retrivedDocs):
    cn = "\n\n".join(doc.page_content for doc in retrivedDocs)
    return cn
parallelChain = RunnableParallel({
    "context": multiRetriever | RunnableLambda(format),
    "question":RunnablePassthrough(),
}
)
llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash-lite"
)
parser = StrOutputParser()
main_chain = parallelChain | prompt | llm | parser 
result = main_chain.invoke("which site is used to trnaslate language")
print(result)