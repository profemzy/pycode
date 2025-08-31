"""
Production-grade web search tools for the chatbot.

Goals:
- Provide a consistent, robust search interface for multiple providers.
- Prioritize providers (Tavily if API key present,
  otherwise DuckDuckGo).
- Implement retries, timeouts, a simple in-memory cache, and clear logging.
- Keep function signatures compatible with the rest of the app:
    - create_tavily_tool() -> Optional[Tool]
    - create_duckduckgo_tool() -> Tool
    - get_search_tools() -> list[Tool]

Notes:
- This module minimizes external dependencies while leveraging
  existing LangChain utilities when available.
- Environment variables:
    - TAVILY_API_KEY (optional)
    - SEARCH_TIMEOUT (seconds, optional, default 6)
    - SEARCH_RETRIES (int, optional, default 2)
    - SEARCH_CACHE_TTL (seconds, optional, default 300)
"""
from __future__ import annotations

import os
import time
import hashlib
import logging
from typing import Optional, Callable, Any, Dict, List

from dotenv import load_dotenv
from langchain_core.tools import Tool

try:
    from utils import validate_search_query, InputValidationError
except ImportError:
    # Fallback if utils module is not available
    def validate_search_query(query: str) -> str:
        return query.strip() if query else ""
    
    class InputValidationError(ValueError):
        pass

# Optional imports for providers; handle absence gracefully
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
except Exception:
    DuckDuckGoSearchRun = None
    DuckDuckGoSearchAPIWrapper = None

try:
    from langchain_tavily import TavilySearch
except Exception:
    TavilySearch = None

load_dotenv()

# Configuration
SEARCH_TIMEOUT = float(os.getenv("SEARCH_TIMEOUT", "6"))
SEARCH_RETRIES = int(os.getenv("SEARCH_RETRIES", "2"))
SEARCH_CACHE_TTL = int(os.getenv("SEARCH_CACHE_TTL", "300"))
MAX_RESULTS = int(os.getenv("SEARCH_MAX_RESULTS", "5"))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Simple in-memory cache with TTL
class SimpleCache:
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    def _now(self) -> float:
        return time.time()

    def _hash(self, key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        h = self._hash(key)
        entry = self._store.get(h)
        if not entry:
            return None
        if entry["expires_at"] < self._now():
            del self._store[h]
            return None
        return entry["value"]

    def set(self, key: str, value: Any, ttl: int = SEARCH_CACHE_TTL) -> None:
        h = self._hash(key)
        self._store[h] = {
            "value": value,
            "expires_at": self._now() + ttl,
        }


_cache = SimpleCache()


# Utility: retry helper with simple backoff
def retry_call(
    func: Callable[..., Any],
    *args,
    retries: int = SEARCH_RETRIES,
    backoff: float = 0.5,
    **kwargs,
) -> Any:
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exc = e
            wait = backoff * (2 ** attempt)
            logger.error(
                "Search attempt %d failed: %s. Retrying in %.2fs",
                attempt + 1,
                e,
                wait,
            )
            time.sleep(wait)
    logger.error(
        "All %d attempts failed for search: %s", retries + 1, last_exc
    )
    raise last_exc


# Provider: DuckDuckGo
def create_duckduckgo_tool() -> Tool:
    """Create a DuckDuckGo-based search tool.

    This always returns a Tool. If the langchain_community utilities
    are missing, a safe dummy tool is returned explaining the issue.
    """
    description = (
        "Search the web for CURRENT information. Use for time-sensitive "
        "queries and recent events. Returns a short summary with source "
        "attribution."
    )

    if DuckDuckGoSearchRun is None or DuckDuckGoSearchAPIWrapper is None:
        def duck_missing(query: str) -> str:
            return (
                "DuckDuckGo search is not available because the required "
                "langchain_community utilities are not installed. Please "
                "install the dependency to enable DuckDuckGo search."
            )

        return Tool(
            name="web_search",
            description=description,
            func=duck_missing,
        )

    # Configure wrapper and runner
    wrapper = DuckDuckGoSearchAPIWrapper(
        max_results=MAX_RESULTS, region="wt-wt", safesearch="moderate"
    )

    def _run(query: str) -> str:
        try:
            # Validate and sanitize the search query
            validated_query = validate_search_query(query)
        except InputValidationError as e:
            logger.warning("Invalid search query: %s", e)
            return f"Invalid search query: {e}"
        except Exception as e:
            logger.error("Query validation error: %s", e)
            return "Search query validation failed."

        cache_key = f"duck:{validated_query}"
        cached = _cache.get(cache_key)
        if cached:
            logger.info("DuckDuckGo cache hit for query: %s", query)
            return cached

        def _do():
            # Use the high-level DuckDuckGo run interface
            search = DuckDuckGoSearchRun(
                api_wrapper=wrapper, timeout=SEARCH_TIMEOUT
            )
            return search.run(validated_query)

        try:
            raw = retry_call(_do)
            if not raw or not str(raw).strip():
                result_text = (
                    "No relevant results found. Try rephrasing your query."
                )
            else:
                # Keep lines under the style guide limit
                result_text = (
                    "🔍 DuckDuckGo Results:\n\n"
                    + str(raw)
                    + "\n\nSource: duckduckgo.com"
                )
            _cache.set(cache_key, result_text)
            return result_text
        except Exception as exc:
            logger.exception("DuckDuckGo search failed: %s", exc)
            return f"DuckDuckGo search failed: {exc}"

    return Tool(name="web_search", description=description, func=_run)


# Provider: Tavily
def create_tavily_tool() -> Optional[Tool]:
    """Create a Tavily-based search tool if configured.

    Returns None if Tavily is not available or not configured.
    """
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        logger.info("Tavily API key not found; skipping Tavily provider.")
        return None
    if TavilySearch is None:
        logger.warning(
            "TavilySearch library not available; cannot create Tavily tool."
        )
        return None

    description = (
        "Tavily web search for CURRENT information. Prioritize Tavily when "
        "available for better quality results."
    )

    def _run(query: str) -> str:
        try:
            # Validate and sanitize the search query
            validated_query = validate_search_query(query)
        except InputValidationError as e:
            logger.warning("Invalid search query: %s", e)
            return f"Invalid search query: {e}"
        except Exception as e:
            logger.error("Query validation error: %s", e)
            return "Search query validation failed."

        cache_key = f"tavily:{validated_query}"
        cached = _cache.get(cache_key)
        if cached:
            logger.info("Tavily cache hit for query: %s", query)
            return cached

        def _do():
            search = TavilySearch(
                max_results=MAX_RESULTS,
                api_key=tavily_api_key,
                topic="general",
                include_domains=None,
                exclude_domains=None,
                timeout=SEARCH_TIMEOUT,
            )
            return search.run(validated_query)

        try:
            raw = retry_call(_do)
            formatted = _format_tavily_result(raw)
            _cache.set(cache_key, formatted)
            return formatted
        except Exception as exc:
            logger.exception("Tavily search failed: %s", exc)
            return f"Tavily search failed: {exc}"

    return Tool(name="web_search", description=description, func=_run)


def _format_tavily_result(raw: Any) -> str:
    """Normalize Tavily's response into a readable string with attribution."""
    if raw is None:
        return "No relevant results found."

    try:
        if isinstance(raw, dict):
            if "results" in raw and isinstance(raw["results"], list):
                parts: List[str] = []
                for i, item in enumerate(
                    raw["results"][:MAX_RESULTS], start=1
                ):
                    title = (
                        item.get("title")
                        or item.get("headline")
                        or "No title"
                    )
                    content = (
                        item.get("content")
                        or item.get("snippet")
                        or ""
                    )
                    url = item.get("url") or item.get("link") or ""
                    block = f"{i}. {title}\n{content}\n{url}".strip()
                    parts.append(block)
                result_text = "\n\n".join(parts)
            elif "answer" in raw:
                result_text = str(raw["answer"])
            else:
                result_text = str(raw)
        else:
            result_text = str(raw)
    except Exception as exc:
        logger.exception("Error formatting Tavily result: %s", exc)
        result_text = str(raw)

    if not result_text.strip():
        return "No relevant results found. Try rephrasing your query."

    return f"🔍 Tavily Results:\n\n{result_text}\n\nSource: tavily.com"


def get_search_tools() -> List[Tool]:
    """Return a prioritized list of search tools.

    Priority:
    1. Tavily (if configured and usable)
    2. DuckDuckGo (fallback)

    Returns an empty list if no providers are usable.
    """
    tools: List[Tool] = []

    # Try Tavily first
    try:
        tavily_tool = create_tavily_tool()
        if tavily_tool:
            test = tavily_tool.func("health check")
            if isinstance(test, str) and test.lower().startswith(
                "tavily search failed"
            ):
                logger.warning(
                    "Tavily provider returned a failure on test: %s", test
                )
            else:
                tools.append(tavily_tool)
                logger.info("Using Tavily provider for web search.")
                return tools
    except Exception:
        logger.exception(
            "Tavily initialization/test failed; falling back to DuckDuckGo"
        )

    # DuckDuckGo fallback
    try:
        duck_tool = create_duckduckgo_tool()
        test = duck_tool.func("health check")
        if isinstance(test, str) and "not available" in test:
            logger.error("DuckDuckGo provider not available: %s", test)
            return []
        tools.append(duck_tool)
        logger.info("Using DuckDuckGo provider for web search.")
    except Exception:
        logger.exception("DuckDuckGo initialization failed.")

    return tools
