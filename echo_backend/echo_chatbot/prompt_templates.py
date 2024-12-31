from langchain_core.prompts import ChatPromptTemplate

# Prompt Templates
class prompt_templates:
    def final_rag_template():
        prompt = """
            You are a meeting facilitator.
            This user will ask you a questions about the conversation of the meeting.
            Use following piece of context to answer the question.
            If you don't know the answer, just say you don't know.
            Keep the answer complete and concise.
            Context: {context}
            Here are some background questions and answers that will help you answer the question: {qa_pairs}
            Question: {question}
        """
        
        return ChatPromptTemplate.from_template(prompt)
    
    def decomposition_template():
        prompt = """
            Break the following user question into smaller, more specific questions.
            Provide these subquestions separated by newlines. 
            Do not rephrase if you see unknown terms.
            Question: {question}
            subquestions:
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