from typing import Optional, Dict, Any, List, Type
from pydantic import BaseModel, ValidationError
from agents.helpers import validate_router
from agents.llm.llm_models import client_llm


def _recover_from_failed_generation(
    error: Exception, constraine_model: Type[BaseModel]
) -> Optional[BaseModel]:
    """
    Some providers (e.g. Groq, on forced tool-calling) reject a response with
    a `tool_use_failed` error when the model writes valid schema-shaped JSON
    as plain content instead of a native tool call. The SDK still surfaces
    that JSON via `error.failed_generation` on the raised exception's `body`
    -- recover it instead of throwing away a perfectly good answer.
    """
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return None

    error_obj = body.get("error")
    if not isinstance(error_obj, dict):
        return None

    failed_generation = error_obj.get("failed_generation")
    if not failed_generation:
        return None

    try:
        return constraine_model.model_validate_json(failed_generation)
    except (ValidationError, TypeError, ValueError):
        return None


class FallBack:
    def __init__(
        self,
        llm_ollama: Optional[str] = None,
        llm_gpt: Optional[str] = None,
        llm_gemini: Optional[str] = None,
        llm_groq: Optional[str] = None,
    ):
        # Map the router names to their specific model strings
        self.llms: Dict[str, str] = {}
        
        if llm_ollama:
            self.llms["ollama"] = llm_ollama
        if llm_gpt:
            self.llms["gpt"] = llm_gpt
        if llm_gemini:
            self.llms["gemini"] = llm_gemini
        if llm_groq:
            self.llms["groq"] = llm_groq
            

    def invoke(self, message: Any, fallback_order: List[str]) -> str:
        """
        Entry point 1: Regular text generation.
        Attempts to invoke models in the provided sequence.
        Falls back to the next router if one fails.

        `message` may be a callable(router) -> messages instead of a fixed
        value, letting a caller shrink the payload for a specific router
        (e.g. a tight-TPM one) while other routers still get the full prompt.
        """
        errors = []

        for router in fallback_order:
            try:
                print(f"Attempting regular generation using: {router}...")
                router = validate_router(router)

                if router not in self.llms:
                    raise ValueError(f"No model string configured for '{router}' during initialization.")

                model_name = self.llms[router]
                llm = client_llm.get_model(router, model_name)

                resolved_message = message(router) if callable(message) else message
                response = llm.invoke(resolved_message)
                return response.content
                
            except Exception as e:
                print(f"Router '{router}' failed: {e}")
                errors.append(f"{router} error: {str(e)}")
                continue
                
        # If the loop finishes without returning, all models failed
        raise RuntimeError(f"All fallback models failed. Details: {errors}")

    def constrained_invoke(
        self,
        message: Any,
        fallback_order: List[str],
        constraine_model: Optional[BaseModel] = None,
        method: Optional[str] = None,
        groq_reasoning_effort: Optional[str] = None,
    ) -> dict:
        """
        Entry point 2: Constrained/structured generation.
        Forces the output to match the Pydantic schema, trying models in sequence.
        Returns the parsed attributes as a dictionary.

        `method` is forwarded to `with_structured_output` (e.g. "json_schema")
        when the caller's schema/model combination needs a specific structured
        output strategy instead of the provider default ("function_calling").

        `groq_reasoning_effort` (e.g. "low") is forwarded only when the router
        being attempted is "groq" -- gpt-oss models spend part of their token
        budget on a hidden reasoning channel before writing the final answer,
        and a short/simple schema doesn't need much of it. Left unset, that
        channel can eat the whole budget on an unlucky generation and leave
        nothing valid for the schema/tool-call validator to parse.
        """
        if not constraine_model:
            raise ValueError("Cannot perform constrained invoke: 'constraine_model' was not provided.")

        errors = []

        for router in fallback_order:
            try:
                print(f"Attempting constrained generation using: {router}...")
                router = validate_router(router)

                if router not in self.llms:
                    raise ValueError(f"No model string configured for '{router}' during initialization.")

                model_name = self.llms[router]
                model_kwargs = {"reasoning_effort": groq_reasoning_effort} if router == "groq" and groq_reasoning_effort else {}
                llm = client_llm.get_model(router, model_name, **model_kwargs)

                # Bind the Pydantic schema to the LLM
                structured_kwargs = {"method": method} if method else {}
                structured_llm = llm.with_structured_output(constraine_model, **structured_kwargs)
                pydantic_response = structured_llm.invoke(message)
                
                # Return as a dictionary
                return pydantic_response.model_dump()
                
            except Exception as e:
                recovered = _recover_from_failed_generation(e, constraine_model)
                if recovered is not None:
                    print(f"Router '{router}' rejected the tool call but its failed_generation parsed cleanly; recovering.")
                    return recovered.model_dump()

                print(f"Constrained router '{router}' failed: {e}")
                errors.append(f"{router} error: {str(e)}")
                continue

        # If the loop finishes without returning, all models failed
        raise RuntimeError(f"All fallback models failed to generate valid constrained output. Details: {errors}")