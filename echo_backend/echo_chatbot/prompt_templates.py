from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Prompt Templates
class prompt_templates:
    def final_rag_template():
        prompt = """
            You are a meeting facilitator.
            This user will ask you a questions about the conversation of the meeting.
            Use the  context to answer the question.
            If you don't know the answer, just say you don't know.
            Keep the answer complete and concise, and make it a paragraph.
            Don't use any '*' in your answer.
            Context: {context}
            Here are some background questions and answers that will help you find answers from the context: {qa_pairs}
            Question: {question}
        """
        
        return ChatPromptTemplate.from_template(prompt)
    
    # def final_rag_template_with_memory():
    #     prompt = ChatPromptTemplate.from_messages(
    #         [
    #             ("system", "You are a meeting facilitator. This user will ask you a questions about the conversation of the meeting. Use the context to answer the question. If you don't know the answer, just say you don't know. ALWAYS refer your answer to the context and keep the answer complete and concise and in paragraph. Don't use any font effect. Context: {context}, Meeting date: {text_date}, Meeting Title: {text_title}"),
    #             MessagesPlaceholder("chat_history"),
    #             ("system", "Here are some background questions and answers that will help you find answers from the context: {qa_pairs}"),
    #             ("human", "{question}"),
    #         ]
    #     )

    #     return prompt
    
    def final_rag_template_with_memory():
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful and concise meeting facilitator. "
                    "The user will ask questions about the content of a meeting. "
                    "Use only the provided context to answer. If the answer is not found in the context, say you don't know. "
                    "Always keep your answer relevant, complete, and in paragraph form. Do not use any font effects. "
                    "Context: {context} | Meeting Date: {text_date} | Meeting Title: {text_title}"
                ),
                (
                    "system",
                    "You may use the following previous chat interactions only if they are relevant to the context and current question. "
                ),
                MessagesPlaceholder("chat_history"),
                (
                    "system",
                    "Here are some background Q&A that may help extract answers from the context: {qa_pairs}"
                ),
                (
                    "human", "{question}"
                )
            ]
        )

        return prompt
    
    # def decomposition_template():
    #     prompt = """
    #         Below is the conversation context followed by the user's question.
    #         Use the context to help decompose the question into not more than three smaller, more specific subquestions.
    #         If you see unknown terms, do not rephrase them.
        
    #         Question: {question}
    #         Subquestions:
    #     """

    #     return ChatPromptTemplate.from_template(prompt)

    def decomposition_template():
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Below is the conversation context, followed by prior related interactions and the user's current question."
                    "Use the context and relevant chat history to help break down the question into no more than three specific subquestions."
                    "Do not rephrase unknown terms. Focus on clarity and specificity."

                    "Previous Chat History:"
                ),
                MessagesPlaceholder("chat_history"),
                (
                    "human",
                    "Question: {question}"
                ),
                (
                    "system",
                    "Subquestions:"
                )
            ]
        )

        return prompt
    
    def qa_template():
        prompt = """
            In paragraph, answer the question in the following context:\n{context}\n\nQuestion: {subquestion} 
        """

        return ChatPromptTemplate.from_template(prompt)
    
    def summary_template():
        prompt = """
            You are a summarization assistant. Generate a concise summary of the following text. Do not include label.
            Incorporate the following details into your summary: Transcript:{texts}, Date of the Meeting:{date}, Meeting Title:{meeting_title}
        """

        return ChatPromptTemplate.from_template(prompt)

    def followup_template():
        prompt = """
            You are a meeting facilitator. The user's query was ambiguous for you to answer. Generate a follow up question to aid you to answer the user's query.
            Use the previous query and the list of recorded meeting titles and its summaries for context.
            Previous query: {question}
            Meeting list: {meeting_list}
            """
        
        return ChatPromptTemplate.from_template(prompt)