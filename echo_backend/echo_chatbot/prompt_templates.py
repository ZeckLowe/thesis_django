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
                ("system", "You are a meeting facilitator. This user will ask you a questions about the conversation of the meeting. Use the  context to answer the question. If you don't know the answer, just say you don't know. ALWAYS refer your answer to the context and keep the answer complete and concise. Context: {context}"),
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
    
    def summary_template():
        prompt = """
            You are a summarization assistant. Generate a concise summary of the following text. Do not include label.
            Incorporate the following details into your summary: {texts}, {date}, {meeting_title}
        """

        return ChatPromptTemplate.from_template(prompt)