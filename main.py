import asyncio
import logging
import argparse
from typing import Optional

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from llm_factory import get_llm, get_llm_async

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()


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
        code_prompt = PromptTemplate.from_template(
            input_variables=["language", "task"],
            template="Write a very short {language} function that will {task}"
        )

        code_chain = code_prompt | llm

        print(f"Prompt: {code_prompt}")
        print("-" * 50)
        result = code_chain.invoke(
            {
                "language": args.language,
                "task": args.task
             }
        )
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

        code_prompt = PromptTemplate(
            input_variables=["language", "task"],
            template="Write a very short {language} function that will {task}"
        )

        test_prompt = PromptTemplate(
            input_variables=["code", "language"],
            template="Write a test for the following {language} code {code}"
        )
        
        def extract_code(result):
            return {"code": result.content, "language": args.language}
        
        # Create a combined chain that flows code -> test
        combined_chain = (
            code_prompt | llm | RunnableLambda(extract_code) | test_prompt | llm
        )

        print(f"Code Prompt: {code_prompt}")
        print("-" * 55)

        final_result = await combined_chain.ainvoke(
            {
                "language": args.language,
                "task": args.task
            }
        )
        
        print("Test Response:")
        print(final_result.content)

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
