import logging
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

from backend.config import settings

logger = logging.getLogger(__name__)

def _get_guardrail_llm() -> ChatGroq:
    return ChatGroq(
        api_key=settings.groq_api_key,
        model_name=settings.guardrail_model,
        temperature=0.0,
    )

def check_input(query: str) -> tuple[bool, str]:
    """
    Checks if the input query is safe using the guardrail model.
    Returns (is_safe, reason)
    """
    if not settings.enable_guardrails:
        return True, ""
        
    try:
        llm = _get_guardrail_llm()
        # For Llama Guard, we pass the user's message as a standard chat.
        response = llm.invoke([HumanMessage(content=query)])
        result = response.content.strip()
        
        if result.lower().startswith("unsafe"):
            reason = result.split("\n")[1] if "\n" in result else "Unsafe content detected"
            logger.warning(f"Guardrail flagged input as unsafe: {reason}")
            return False, reason
            
        return True, ""
    except Exception as e:
        logger.error(f"Error evaluating input guardrail: {e}")
        # Fail open on error
        return True, ""

def check_output(query: str, response_text: str) -> tuple[bool, str]:
    """
    Checks if the output response is safe using the guardrail model.
    Returns (is_safe, reason)
    """
    if not settings.enable_guardrails:
        return True, ""
        
    try:
        llm = _get_guardrail_llm()
        # Llama Guard can check assistant responses by providing the conversation history
        messages = [
            HumanMessage(content=query),
            AIMessage(content=response_text)
        ]
        
        response = llm.invoke(messages)
        result = response.content.strip()
        
        if result.lower().startswith("unsafe"):
            reason = result.split("\n")[1] if "\n" in result else "Unsafe content detected in output"
            logger.warning(f"Guardrail flagged output as unsafe: {reason}")
            return False, reason
            
        return True, ""
    except Exception as e:
        logger.error(f"Error evaluating output guardrail: {e}")
        # Fail open on error
        return True, ""
