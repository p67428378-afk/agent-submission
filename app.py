import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import traceback

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

# Initialize FastAPI
app = FastAPI(title="LangChain Bedrock Chatbot")

# Global variables for the agent and error state
agent_executor = None
init_error: Optional[str] = None

# --- Define Tools ---

@tool
def get_current_time() -> str:
    """Returns the current date and time. Useful for finding out what time it is."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiplies two numbers. Useful for simple mathematical calculations."""
    return a * b

tools = [get_current_time, multiply_numbers]

def initialize_agent():
    global agent_executor, init_error
    try:
        # Standard Nova Lite model ID
        model_id = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")
        region = os.getenv("AWS_REGION", "us-east-1")
        
        llm = ChatBedrock(
            model_id=model_id,
            model_kwargs={"temperature": 0},
            region_name=region
        )
        
        # This uses the new LangChain 1.x / LangGraph agent factory
        agent_executor = create_agent(
            llm, 
            tools
        )
        init_error = None
        print(f"Successfully initialized agent with model: {model_id}")
    except Exception as e:
        init_error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"Agent Initialization Error:\n{init_error}")
        agent_executor = None

# Initialize on startup
initialize_agent()

# --- API Models ---

class InvokeRequest(BaseModel):
    input: str

class InvokeResponse(BaseModel):
    output: str

# --- Endpoints ---

@app.get("/ping")
async def ping():
    return {"message": "pong"}

@app.get("/tools")
async def get_tools():
    tool_info = []
    for t in tools:
        tool_info.append({
            "name": t.name,
            "description": t.description,
            "input_format": t.args,
            "output_format": "string" if t.name == "get_current_time" else "number"
        })
    return {"tools": tool_info}

@app.post("/invoke", response_model=InvokeResponse)
async def invoke(request: InvokeRequest):
    if not agent_executor:
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Agent not initialized",
                "debug_info": init_error or "Unknown error"
            }
        )
    
    try:
        # Include system message in the list
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Use the provided tools to answer questions if necessary."},
            {"role": "user", "content": request.input}
        ]
        result = agent_executor.invoke({"messages": messages})
        output_text = result["messages"][-1].content
        return InvokeResponse(output=output_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
