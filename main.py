import asyncio
import json
import sys
from typing import Optional
from contextlib import AsyncExitStack
from datetime import datetime as dt

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger


logger.remove()
logger.add(sys.stderr, level='INFO')
load_dotenv()  # load environment variables from .env

# MODEL = "anthropic/claude-3.7-sonnet"
MODEL = "google/gemini-2.5-flash"
# MODEL = "google/gemini-2.5-flash-lite"

def convert_tool_format(tool):
    converted_tool = {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": tool.inputSchema["properties"],
                "required": tool.inputSchema["required"]
            }
        }
    }
    return converted_tool

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai = OpenAI(
            base_url="https://openrouter.ai/api/v1"
        )

    async def connect_to_server(self):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        server_params = StdioServerParameters(
            command='npx',
            args=["-y",
                    "@modelcontextprotocol/server-filesystem",
                    f"/home/xl2296/mcp/"],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools from the MCP server
        response = await self.session.list_tools()
        logger.info("\nConnected to server with tools:",
              [tool.name for tool in response.tools])

        self.messages = []

    async def process_query(self, query: str) -> str:
        self.messages.append({
            "role": "user",
            "content": query
        })
        response = await self.session.list_tools()
        available_tools = [convert_tool_format(tool) for tool in response.tools]
        response = self.openai.chat.completions.create(
            model=MODEL,
            tools=available_tools,
            messages=self.messages
        )
        self.messages.append(response.choices[0].message.model_dump())
        final_text = []
        content = response.choices[0].message
        if content.tool_calls is not None:
            tool_name = content.tool_calls[0].function.name
            tool_args = content.tool_calls[0].function.arguments
            tool_args = json.loads(tool_args) if tool_args else {}
            # Execute tool call
            try:
                result = await self.session.call_tool(tool_name, tool_args)
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
            except Exception as e:
                print(f"Error calling tool {tool_name}: {e}")
                result = None
            self.messages.append({
                "role": "tool",
                "tool_call_id": content.tool_calls[0].id,
                "name": tool_name,
                "content": result.content
            })
            response = self.openai.chat.completions.create(
                model=MODEL,
                max_tokens=1000,
                messages=self.messages,
            )
            logger.info(response.provider)
            final_text.append(response.choices[0].message.content)
        else:
            final_text.append(content.content)
        # breakpoint()
        return "\n".join(final_text)

    async def chat_loop(self):
        """ Run an interactive chat loop """

        logger.debug("MCP Client Started!")
        logger.info("Type your queries or 'q' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'q':
                    break
                
                response = await self.process_query(query)
                logger.info('\n' + response)
            except Exception as e:
                logger.error(e)

    async def cleanup(self):
        await self.exit_stack.aclose()

async def main():
    client = MCPClient()
    try:
        await client.connect_to_server()
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
