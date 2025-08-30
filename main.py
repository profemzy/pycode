import asyncio
import logging
from typing import Optional

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

from llm_factory import get_llm, get_llm_async

load_dotenv()


def create_llm_client(model: Optional[str] = None) -> BaseChatModel:
    """Create and return an LLM client with an optional model override."""
    try:
        llm = get_llm(model)
        return llm
    except ValueError as e:
        logging.error(f"Configuration error initializing LLM client: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error initializing LLM client: {e}")
        raise


def main():
    """Main function to demonstrate LLM usage with proper error handling."""
    try:
        llm = create_llm_client()
        prompt = "What are Large Language Models?"
        print(f"Prompt: {prompt}")
        print("-" * 50)
        result = llm.invoke(prompt)
        print("Response:")
        if result.content:
            print(result.content)
        else:
            print("Response is empty")
            print(f"Result object: {result}")
            print(f"Result type: {type(result)}")
            print(f"Available attributes: {dir(result)}")

    except ValueError as ve:
        print(f"Configuration error: {ve}")
        print("Please check your .env file and ensure OPENAI_API_KEY is set.")
    except Exception as e:
        print(f"An error occurred: {e}")
        logging.error(f"Unexpected error: {e}")


async def main_async():
    """Async version of the main function demonstrating LLM usage."""
    try:
        llm = await get_llm_async()

        prompt = "Explain the benefits of asynchronous programming."
        print(f"Prompt: {prompt}")
        print("-" * 55)

        result = await llm.ainvoke(prompt)
        print("Response:")
        if result.content:
            print(result.content)
        else:
            print("Response is empty")
            print(f"Result object: {result}")

    except ValueError as ve:
        print(f"Configuration error: {ve}")
        print("Please check your .env file and ensure OPENAI_API_KEY is set.")
    except Exception as e:
        print(f"An error occurred: {e}")
        logging.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    import sys
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if "--sync-only" in sys.argv:
        print("=== Synchronous LLM Demo ===")
        main()
    else:
        # Default: async-only (optimal for production)
        print("=== Asynchronous LLM Demo ===")
        asyncio.run(main_async())
