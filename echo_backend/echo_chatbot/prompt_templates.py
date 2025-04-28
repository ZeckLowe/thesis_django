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
                    "You may use the following previous chat interactions only if they are relevant to the context and current query. "
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

    # def decomposition_template():
    #     prompt = ChatPromptTemplate.from_messages(
    #         [
    #             (
    #                 "system",
    #                 "Below is the conversation context, followed by prior related interactions and the user's current question."
    #                 "Use the context and chat history to decide whether the current question consists of multiple parts or distinct sub-questions."
    #                 "If the question is clearly a single, standalone query, do not decompose it and return the original query."
    #                 "If the question contains multiple inquiries or distinct components, break it down into no more than three specific sub-questions for clarity."
    #                 "Do not rephrase unknown terms. Focus on clarity and specificity."

    #                 "Previous Chat History:"
    #             ),
    #             MessagesPlaceholder("chat_history"),
    #             (
    #                 "human",
    #                 "Question: {question}"
    #             ),
    #             (
    #                 "system",
    #                 "Subquestions:"
    #             )
    #         ]
    #     )

    #     return prompt
    def decomposition_template():
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Below is the conversation context, followed by prior related interactions and the user's current question. "
                    "Analyze the conversation flow to determine which questions need to be addressed first. "
                    "Pay special attention to follow-up questions as they often reference or build upon previous inquiries. "
                    "Consider these factors when analyzing the current question and chat history:"
                    "\n\n1. Recency - More recent questions generally take priority"
                    "\n2. Explicit follow-ups - Direct follow-up questions should be linked to their parent questions"
                    "\n3. Unanswered aspects - Identify parts of previous questions that weren't fully addressed"
                    "\n4. Context shifts - Note when the user changes topic or direction"
                    "\n\nPrevious Chat History:"
                ),
                MessagesPlaceholder("chat_history"),
                (
                    "human",
                    "Question: {question}"
                ),
                (
                    "system",
                    "First, identify if the current question is a follow-up to a previous question or a new line of inquiry. "
                    "Then, determine if decomposition is needed:"
                    "\n- If it's a simple follow-up that references a previous question, return both the parent question and the follow-up for context"
                    "\n- If it's a complex question with multiple parts, decompose it into specific sub-questions (no more than three)"
                    "\n- If it's a single, standalone query, return the original question"
                    "\n\nOutput format: Return only the prioritized question(s) or sub-questions without explanations."
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
            Give the list of meetings as a guide for the user to choose from.
            Always keep your answer relevant, complete and concise, and in paragraph form. Do not use any font effects.
            Previous query: {question}
            Meeting list: {meeting_list}
            """

        return ChatPromptTemplate.from_template(prompt)