from langchain_groq import ChatGroq
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage 
import os

load_dotenv()

llm = ChatGroq(model='openai/gpt-oss-20b')
mcp_token= os.getenv("MCP_API_KEY")
SERVERS = {
    "expense-tracker-shubham": {
        "transport": "streamable_http",
        "url": "https://squealing-emerald-cardinal.fastmcp.app/mcp",
        # Uncomment and add your token here if you are still getting the 401 error!
        "headers": {
             "Authorization": f"Bearer {mcp_token}"
         }
    }git commit --amend --no-edit
}

async def main():
    client = MultiServerMCPClient(SERVERS)
    named_tool = {}
    
    tools = await client.get_tools()
    for tool in tools:
        
        named_tool[tool.name] = tool
        
    llm_with_tools = llm.bind_tools(tools)
    
    # Initialize the memory list OUTSIDE the loop so it remembers past questions
    messages = []
    print("Enter your prompt (or type 'exit' to quit):")
    prompt = input("User Question: ")
    
    while prompt.lower() != "exit":
        # Append the new user input to the running history
        messages.append(HumanMessage(content=prompt))
        
        # First LLM call
        response1 = await llm_with_tools.ainvoke(messages)
        messages.append(response1) 
        
        if response1.tool_calls:
            tool_call = response1.tool_calls[0]
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            tool_id = tool_call['id']
            
            print(f"\n[🔧 Tool Called: {tool_name}]")
            
            # Invoke the tool
            tool_result = await named_tool[tool_name].ainvoke(tool_args)
            
            # Pass the result back to the LLM
            messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_id))
            
            # Call the LLM one final time to summarize the data
            final_response = await llm.ainvoke(messages)
            
            # Append the final answer to history so the LLM remembers what it told you
            messages.append(final_response)
            
            print("\n[AI Answer]")
            print(final_response.content)
            
        else:
            # If no tool was needed, print the response (it's already in 'messages')
            print("\n[AI Answer]")
            print(response1.content)
            
        print("-" * 40)
        prompt = input("User Question: ")

if __name__ == "__main__":
    asyncio.run(main())