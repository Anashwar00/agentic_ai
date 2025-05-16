from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import Tool
from langchain_tavily import TavilySearch
from langchain_chroma import Chroma
from langgraph.graph import StateGraph

load_dotenv()
search=TavilySearch()
# create embeddings
embedding=OpenAIEmbeddings()
# #create db
vectorstore=Chroma(collection_name="agentic_bot",embedding_function=embedding,persist_directory="./db/chroma")
retrive=vectorstore.as_retriever()


search_tool=Tool(
    name="WebSearch",
    func=search.run,
    description="Search the web for recent and factual information"
)
llm=ChatOpenAI(model="gpt-4o",temperature=0.0)

def agent_bot(state):
    user_inp=state["user_inp"]
    prompt=f"""You are a assistent with access to tools
    user asked:{user_inp}
    """
    docs=retrive.get_relevant_documents(user_inp)
    memory=[doc.metadata["response"] for doc in docs]
    search_res=search_tool.run(user_inp)
    print(search_res)
    
    res=llm.invoke(f"Use this {memory} following previous conversation memory,and based on search result {search_res},answer the question:{user_inp} ")
    vectorstore.add_texts([user_inp],metadatas=[{"response":res.content}])
    # vectorstore.add_texts([res],metadatas=[{"response":res.content}])
    return {"result":res}

graph=StateGraph(dict)
graph.add_node("agent",agent_bot)
graph.set_entry_point("agent")
graph.set_finish_point("agent")
app=graph.compile()

while True:
    query=input("enter your query:")
    if query in ["exit","quit"]:
        break
    
    res=app.invoke({"user_inp":query})
    
    print(res['result'].content)



    
