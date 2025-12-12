#!/usr/bin/env python3
"""
Simple CLI chat interface using tinker sampling client.
"""

import asyncio
import logging
import sys
import os

import chz
import tinker
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(filename)s:%(lineno)-4s %(message)s",
    level=logging.WARNING,
    datefmt="%Y-%m-%d %H:%M:%S",
)


@chz.chz
class Config:
    base_model: str = "meta-llama/Llama-3.2-1B"
    model_path: str | None = None
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    base_url: str | None = None


class ChatSession:
    """Manages a chat session with conversation history."""

    def __init__(
        self,
        sampling_client: tinker.SamplingClient,
        renderer: renderers.Renderer,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ):
        self.sampling_client: tinker.SamplingClient = sampling_client
        self.renderer: renderers.Renderer = renderer
        self.max_tokens: int = max_tokens
        self.temperature: float = temperature
        self.top_p: float = top_p
        self.conversation_history: list[renderers.Message] = []

    def add_user_message(self, content: str):
        """Add a user message to the conversation history."""
        self.conversation_history.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        """Add an assistant message to the conversation history."""
        self.conversation_history.append({"role": "assistant", "content": content})

    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history.clear()

    async def generate_response(self) -> str:
        """Generate a response from the model."""
        try:
            # Build the model input from conversation history
            model_input = self.renderer.build_generation_prompt(self.conversation_history)

            # Set up sampling parameters
            sampling_params = types.SamplingParams(
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stop=self.renderer.get_stop_sequences(),
            )

            # Generate response
            response = await self.sampling_client.sample_async(
                prompt=model_input, num_samples=1, sampling_params=sampling_params
            )

            # Parse the response
            parsed_message, _ = self.renderer.parse_response(response.sequences[0].tokens)
            content = renderers.ensure_text(parsed_message["content"])

            self.add_assistant_message(content)
            return content

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {e}"


async def main(config: Config):
    """Main chat loop."""

    print(f"🚀 Initializing chat with model: {config.base_model}")
    print(f"📦 Using Path: {config.model_path}")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        # Create service client
        service_client = tinker.ServiceClient(base_url=config.base_url)

        # Create sampling client
        sampling_client = service_client.create_sampling_client(
            base_model=config.base_model,
            model_path=config.model_path if config.model_path else None,
        )

        # Get tokenizer and renderer
        tokenizer = get_tokenizer(config.base_model)
        renderer = renderers.get_renderer(
            get_recommended_renderer_name(config.base_model), tokenizer
        )

        # Create chat session
        chat_session = ChatSession(
            sampling_client=sampling_client,
            renderer=renderer,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )

        print("\n💬 Chat started! Type 'quit', 'exit', or Ctrl+C to end the session.")
        print("🗑️  Type 'n' to clear conversation history and start a new conversation.")
        print("🤖 You can start chatting now...\n")

        # Main chat loop
        while True:
            try:
                # Get user input
                user_input = input("User: ").strip()

                # Check for exit commands
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break

                # Check for clear history command
                if user_input.lower() == "n":
                    chat_session.clear_history()
                    print("🗑️  Conversation history cleared! Starting a new conversation.")
                    continue

                if not user_input:
                    continue

                # Add user message to conversation
                chat_session.add_user_message(user_input)

                # Generate and display response
                print("Assistant: ", end="", flush=True)
                response = await chat_session.generate_response()
                print(response)
                print()  # Empty line for readability

            except KeyboardInterrupt:
                print("\n👋 Chat interrupted. Goodbye!")
                break
            except EOFError:
                print("\n👋 Chat ended. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                logger.exception("Unexpected error in chat loop")

    except Exception as e:
        print(f"❌ Failed to initialize chat: {e}")
        logger.exception("Failed to initialize chat")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(chz.nested_entrypoint(main))
