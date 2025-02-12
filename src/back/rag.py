import pandas as pd
import os
from dotenv import load_dotenv
from langgraph.graph import MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool

# Үүсгэсэн vector store-г файлаас унших
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vector_store = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
langchain_tracing = os.getenv("LANGCHAIN_TRACING_V2")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

PROJECT_ID = "ai-last-project-2024"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}
import vertexai

vertexai.init(project=PROJECT_ID, location=LOCATION)

from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(model="gemini-1.5-pro")


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Classify the query using keyword matching."""
    query = query.lower()
    retrieved_docs = vector_store.similarity_search(query, k=5)

    # Serialize the results
    serialized = "\n\n".join(
        f"Source: {doc.metadata.source}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs
graph_builder = StateGraph(MessagesState)

# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    # return state['messages']
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an mongolian assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise, and use mongolian. if the result contains metadata, give the source"
        
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}

memory = MemorySaver()
config = {"configurable": {"thread_id": "llama000"}}

graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile(checkpointer=memory)

def invoke_llama(input_message: str):
    results = graph.stream(
        {"messages": [{"role": "user", "content": input_message}]},
        stream_mode="values",
        config = config
    )
    for item in results:
        last_item = item
    return last_item