from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Prompt Templates
class prompt_templates:
    def final_rag_template():
        prompt = """
            You are a meeting facilitator.
            This user will ask you a questions about the conversation of the meeting.
            Use the  context to answer the question.
            If you don't know the answer, just say you don't know.
            Keep the answer complete and concise.
            Context: {context}
            Here are some background questions and answers that will help you find answers from the context: {qa_pairs}
            Question: {question}
        """
        
        return ChatPromptTemplate.from_template(prompt)
    
    def final_rag_template_with_memory():
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a meeting facilitator. This user will ask you a questions about the conversation of the meeting. Use the  context to answer the question. If you don't know the answer, just say you don't know. Keep the answer complete and concise. Context: {context}"),
                ("system", "Here are some background questions and answers that will help you find answers from the context: {qa_pairs}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

        return prompt
    
    def decomposition_template():
        prompt = """
            Below is the conversation context followed by the user's question.
            Use the context to help decompose the question into smaller, more specific subquestions.
            If you see unknown terms, do not rephrase them.
        
            Question: {question}
            Subquestions:
        """

        return ChatPromptTemplate.from_template(prompt)
    
    def qa_template():
        prompt = """
            Answer the question in the following context:\n{context}\n\nQuestion: {subquestion}
        """

        return ChatPromptTemplate.from_template(prompt)
    
    def conversation_history_template():
        prompt = """
            Conversation History: {memory}\nUser Query: {query}\nResponse: {initial_response}\nRefined Answer:
        """

        return ChatPromptTemplate.from_template(prompt)

    def contextualize_q_system_template():
        prompt = """
            Given a chat history and the latest user question which might reference context in the chat history, \
            formulate a standalone question which can be understood without the chat history. Do NOT answer the question, \
            just reformulate it if needed and otherwise return it as is.
            """

        return ChatPromptTemplate.from_messages(
            [
                ("system", prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ]
        )