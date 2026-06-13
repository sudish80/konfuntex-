import json
import time
import logging
from typing import Optional, Callable
from config.settings import settings

logger = logging.getLogger(__name__)


class LLMResponseError(Exception):
    """Raised when LLM response is malformed or missing required fields."""
    pass


class LLMAPIError(Exception):
    """Raised when the LLM API call fails after all retries."""
    pass


class LLMClient:
    MAX_RETRIES = 3
    BASE_DELAY = 2.0
    MAX_DELAY = 30.0

    def __init__(self):
        self.provider = settings.llm_provider
        self.model = settings.llm_model
        self.temperature = settings.agent_temperature
        self.client = self._build_client()

    def _build_client(self):
        if self.provider == "openai":
            from openai import OpenAI
            kwargs = {"api_key": settings.openai_api_key}
            if settings.openai_base_url:
                kwargs["base_url"] = settings.openai_base_url
            return OpenAI(**kwargs)
        elif self.provider == "anthropic":
            import anthropic
            return anthropic.Anthropic(api_key=settings.anthropic_api_key)
        elif self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=settings.gemini_api_key)
            return genai
        elif self.provider == "local":
            from openai import OpenAI
            return OpenAI(base_url=settings.local_llm_endpoint, api_key="not-needed")
        raise ValueError(f"Unknown provider: {self.provider}")

    def _call_with_retry(self, fn, *args, **kwargs) -> dict:
        last_error = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(f"LLM API call failed (attempt {attempt}/{self.MAX_RETRIES}): {e}")
                if attempt < self.MAX_RETRIES:
                    delay = min(self.BASE_DELAY * (2 ** (attempt - 1)), self.MAX_DELAY)
                    time.sleep(delay)
        raise LLMAPIError(f"LLM API call failed after {self.MAX_RETRIES} attempts: {last_error}")

    def _validate_response(self, response: dict) -> dict:
        if "role" not in response:
            raise LLMResponseError("Response missing 'role' field")
        content = response.get("content")
        tool_calls = response.get("tool_calls")
        if not content and not tool_calls:
            raise LLMResponseError("Response has no content and no tool_calls")
        if tool_calls:
            for tc in tool_calls:
                if "function" not in tc or "name" not in tc.get("function", {}):
                    raise LLMResponseError(f"Malformed tool_call: {tc}")
                try:
                    json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, KeyError) as e:
                    raise LLMResponseError(f"Tool call arguments not valid JSON: {e}")
        return response

    def _call_openai(self, messages: list, tools: list = None, tool_choice: str = None) -> dict:
        def _do_call():
            kwargs = dict(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
            if tools:
                kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
            resp = self.client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            result = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                result["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                    }
                    for tc in msg.tool_calls
                ]
            return result
        return self._call_with_retry(_do_call)

    def _call_anthropic(self, messages: list, tools: list = None) -> dict:
        def _do_call():
            system = None
            msgs = list(messages)
            if msgs and msgs[0].get("role") == "system":
                system = msgs.pop(0)["content"]
            api_kwargs = dict(model=self.model, max_tokens=4096, temperature=self.temperature)
            if system:
                api_kwargs["system"] = system
            if tools:
                api_kwargs["tools"] = tools
            anthropic_msgs = [m for m in msgs if m["role"] != "system"]
            api_kwargs["messages"] = anthropic_msgs
            resp = self.client.messages.create(**api_kwargs)
            result = {"role": "assistant", "content": ""}
            text_parts = [b.text for b in resp.content if hasattr(b, "text")]
            if text_parts:
                result["content"] = "\n".join(text_parts)
            if resp.stop_reason == "tool_use":
                tool_blocks = [b for b in resp.content if b.type == "tool_use"]
                result["tool_calls"] = [
                    {"id": tb.id, "type": "function",
                     "function": {"name": tb.name, "arguments": json.dumps(tb.input)}}
                    for tb in tool_blocks
                ]
            return result
        return self._call_with_retry(_do_call)

    def _call_gemini(self, messages: list) -> dict:
        def _do_call():
            model = self.client.GenerativeModel(self.model)
            gemini_history = []
            for m in messages:
                if m["role"] in ("user", "assistant"):
                    gemini_history.append({
                        "role": "user" if m["role"] == "user" else "model",
                        "parts": [m["content"]]
                    })
            resp = model.generate_content(gemini_history)
            return {"role": "assistant", "content": resp.text}
        return self._call_with_retry(_do_call)

    def chat(self, messages: list, tools: list = None, tool_choice: str = None) -> dict:
        try:
            if self.provider in ("openai", "local"):
                response = self._call_openai(messages, tools, tool_choice)
            elif self.provider == "anthropic":
                response = self._call_anthropic(messages, tools)
            elif self.provider == "gemini":
                response = self._call_gemini(messages)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

            return self._validate_response(response)
        except LLMResponseError as e:
            logger.error(f"Invalid LLM response: {e}")
            return {"role": "assistant", "content": f"[Error: LLM returned malformed response - {e}]"}
        except LLMAPIError as e:
            logger.error(f"LLM API error: {e}")
            return {"role": "assistant", "content": f"[Error: API call failed - {e}]"}
        except Exception as e:
            logger.error(f"Unexpected LLM error: {e}")
            return {"role": "assistant", "content": f"[Error: Unexpected error - {e}]"}

    def extract_json_from_response(self, content: str) -> Optional[dict]:
        if not content:
            return None
        import re
        json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", content)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        brace_match = re.search(r"(\{[\s\S]*\})", content)
        if brace_match:
            try:
                return json.loads(brace_match.group(1))
            except json.JSONDecodeError:
                pass
        return None

    def safe_json_chat(self, messages: list, parser: str = "auto") -> Optional[dict]:
        response = self.chat(messages)
        content = response.get("content", "")
        if parser == "json_only":
            return self.extract_json_from_response(content)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return self.extract_json_from_response(content)

    def chat_stream(self, messages: list, on_chunk: Callable[[str], None] = None) -> str:
        def _do_stream():
            if self.provider in ("openai", "local"):
                stream = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    temperature=self.temperature, stream=True
                )
                full = ""
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    full += delta
                    if on_chunk:
                        on_chunk(delta)
                return full
            else:
                resp = self.chat(messages)
                text = resp.get("content", "")
                if on_chunk:
                    on_chunk(text)
                return text
        try:
            return self._call_with_retry(_do_stream)
        except Exception as e:
            error_msg = f"[Stream error: {e}]"
            if on_chunk:
                on_chunk(error_msg)
            return error_msg
