from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, SystemMessage
from langchain.retrievers import BaseRetriever
from langchain.llms import OpenAI
from typing import List, Dict, Any

class HistoryAwareRetriever:
    def __init__(self, llm: OpenAI, retriever: BaseRetriever, prompt: ChatPromptTemplate):
        self.llm_chain = LLMChain(llm=llm, prompt=prompt)
        self.retriever = retriever

    def get_relevant_documents(self, chat_history: List[Dict[str, Any]], user_input: str):
        # Construct chat history and input to contextualize the question
        formatted_chat_history = [
            HumanMessage(content=item["human"]) if item["role"] == "user" else SystemMessage(content=item["content"])
            for item in chat_history
        ]

        # Generate a standalone question
        standalone_question = self.llm_chain.run(chat_history=formatted_chat_history, input=user_input)

        # Retrieve documents using the reformulated question
        return self.retriever.get_relevant_documents(standalone_question)