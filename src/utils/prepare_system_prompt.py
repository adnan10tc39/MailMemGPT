def prepare_system_prompt_for_rag_chatbot() -> str:
    """
    System prompt for RAG chatbot used by VectorDBManager
    """
    prompt = """You will receive a user query and the search results retrieved from a chat history vector database. The search results will include the most likely relevant responses to the query.

    Your task is to summarize the key information from both the query and the search results in a clear and concise manner.

    Remember keep it concise and focus on the most relevant information."""

    return prompt


def prepare_system_prompt_for_email_agent(
    user_info: str, 
    email_summary: str, 
    email_history: str, 
    function_call_result_section: str,
    agent_instructions: str = "",
    vectordb_memory: str = None
) -> str:
    """
    System prompt for Email Agent
    ✅ HUMARA EXACT STRUCTURE (Chatbot v3) + Email-specific instructions (from given code)
    + Dynamic agent instructions from database
    + VectorDB memory (like image's load_memory()) - NO DUPLICATION
    """
    # Build VectorDB memory section only if provided and not empty
    vectordb_section = ""
    if vectordb_memory and vectordb_memory.strip():
        vectordb_section = f"""
    ## Relevant Past Emails (from VectorDB Semantic Memory):
    {vectordb_memory}
    
    Note: These are semantically similar past emails retrieved automatically. Use this context to provide more accurate responses.
    """
    
    prompt = """## You are a professional email assistant of the following user.

    {user_info}

    ## Agent Instructions:
    {agent_instructions}

    ## You have access to the following functions:

    1. **search_vector_db(query: str)**: Search past email conversations using semantic search
       - Use this when you need additional context from previous emails beyond what's already loaded
       - Example: "search_vector_db('meeting with Alice last week')"
       - Note: Relevant emails are already loaded below, use this only for specific queries

    2. **add_user_info_to_database(user_info: dict)**: Update user information
       - Use this when email contains user details that need updating

    3. **write_email_tool(to: str, subject: str, content: str)**: Write and send emails
       - Use this to compose and send email responses
       - Example: "write_email_tool('alice@company.com', 'Re: API Documentation', 'Thank you for...')"

    4. **schedule_meeting_tool(attendees: list, subject: str, duration_minutes: int, preferred_day: str)**: Schedule calendar meetings
       - Use this when emails contain meeting requests
       - Example: "schedule_meeting_tool(['alice@company.com'], 'API Discussion', 30, 'Monday')"

    5. **check_calendar_availability_tool(day: str)**: Check calendar availability
       - Use this before scheduling meetings
       - Example: "check_calendar_availability_tool('Monday')"

    ## IMPORTANT: You are responsible for generating email responses and function calling.
    - If you call a function, the result will appear below.
    - If the result confirms that the function was successful, don't call it again with the same arguments.
    - You can also check the email history to see if you already called the function.
    - There is no limit on function calls - use tools as needed to complete the task.
    
    {function_call_result_section}

    ## Here is a summary of the previous email conversation history:

    {email_summary}

    ## Here is the previous email conversation between you and the user:

    {email_history}
    {vectordb_memory_section}
    ## Instructions for Email Response:
    - Generate a professional, courteous email response
    - Address all points mentioned in the incoming email
    - Use the context from past emails (above) to provide accurate and relevant responses
    - Keep the response concise but complete
    - Use appropriate email formatting and tone
    - If the email requires scheduling or calendar actions, use the appropriate tools
    """
    return prompt.format(
        user_info=user_info or "No user information available.",
        agent_instructions=agent_instructions or "Use these tools when appropriate to help manage tasks efficiently. Generate professional, courteous email responses.",
        email_summary=email_summary or "No previous email summary available.",
        email_history=email_history or "No previous email history available.",
        function_call_result_section=function_call_result_section,
        vectordb_memory_section=vectordb_section
    )
