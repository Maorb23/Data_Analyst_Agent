from typing import Literal, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage

from memory import MemoryStore

import os, json, re
from langchain_core.pydantic_v1 import BaseModel, Field

class AgentState(BaseModel):
    thread_id:       str
    user_input:      str
    history:         str
    route:           Optional[str] = Field(default=None)
    assistant_reply: Optional[str] = None
    last_user:       Optional[str] = None  # Add this field

KEY_PATH = "Nebius_api_key.txt"
with open(KEY_PATH, "r") as f:
    NEBIUS_API_KEY = f.read().strip()

llm = ChatOpenAI(
    model="Qwen/Qwen3-30B-A3B",
    temperature=0.0,
    openai_api_key=os.getenv("NEBIUS_API_KEY") or NEBIUS_API_KEY,
    openai_api_base="https://api.studio.nebius.ai/v1",
    # Add stop sequences to prevent <think> tags
)

## --- Classifier Node -------------------------------------------------- ##
def classify_node(state: AgentState) -> Dict[str, Any]:
    """Return {'route': 'structured' | 'unstructured' | 'refuse'}."""
    q = state.user_input
    prompt = (
        "Classify the query strictly into one of three labels:\n"
        "1. structured – asking for statistical / analytical insight on structured data.\n"
        "2. unstructured – asking about qualitative / text‑heavy questions.\n"
        "3. refuse – unrelated/out‑of‑scope (e.g. weather).\n"
        "\n"
        "Examples:\n"
        "User: What is the average order value by month?\nLabel: structured\n"
        "User: Can you summarize the class's instructions for this intent?\nLabel: unstructured\n"
        "User: What's the weather in Paris today?\nLabel: refuse\n"
        "User: Show me a bar chart of sales by region.\nLabel: structured\n"
        "User: Write a poem about our company.\nLabel: unstructured\n"
        "User: Tell me a joke.\nLabel: refuse\n"
        "\n"
        "Return ONLY the label."
    )
    label = llm.invoke(prompt + "\nQuery: " + q).content.lower().strip()
    route = (
        "structured" if "structured" in label else
        "unstructured" if "unstructured" in label else
        "refuse"
    )
    return {"route": route}

# Improved and more explicit prompt that prevents <think> tags
SUMMARY_PROMPT = """You are a memory extraction assistant. Your job is to determine if the user's message contains preferences that should be stored in memory.

CRITICAL: Do NOT use <think> tags or any reasoning text. Output ONLY the JSON response.

Analyze this user message:
"{last_user}"

Look for these specific types of information:
- locale/location preferences (e.g., "I'm in Germany", "Use US format")
- output format preferences (e.g., "Give me PDF", "I prefer tables", "Show as chart")
- project context (e.g., "I'm working on X project", "This is for Y analysis")

Instructions:
1. If the message contains ANY of the above information, respond with valid JSON:
   {{"should_write": true, "memory_to_write": {{"key": "value"}}}}

2. If the message contains NONE of the above information, respond with:
   {{"should_write": false}}

3. IMPORTANT: Output ONLY JSON, no explanations, no <think> tags, no other text.

Examples:
User: "I'm in France and need this in PDF format"
Response: {{"should_write": true, "memory_to_write": {{"locale": "France", "preferred_output_format": "PDF"}}}}

User: "What's the weather today?"
Response: {{"should_write": false}}

User: "Can you analyze my sales data?"
Response: {{"should_write": false}}

JSON response only:"""

def summary_node(state: AgentState) -> Dict[str, Any]:
    # Get the last user message
    last_user = getattr(state, 'last_user', None) or state.user_input
    thread_id = state.thread_id
    
    print(f"[summary_node] Analyzing message: {last_user}")
    
    # Get LLM response
    try:
        response = llm.invoke(SUMMARY_PROMPT.format(last_user=last_user))
        raw_content = response.content.strip()
        print(f"[summary_node] RAW LLM response: {raw_content}")
    except Exception as e:
        print(f"[summary_node] LLM invoke error: {e}")
        return {}

    # Clean and extract JSON - FIXED VERSION
    try:
        # First, remove <think> blocks completely
        cleaned = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
        
        # Remove any markdown formatting
        cleaned = cleaned.replace('```json', '').replace('```', '').strip()
        
        # Normalize quotes
        cleaned = (
            cleaned
            .replace('"', '"').replace('"', '"')
            .replace("'", "'").replace("'", "'")
        )
        
        # --- Improved JSON extraction: try full parse, then fallback to regex ---
        def extract_json_with_should_write(text):
            # First, try to parse the whole string
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict) and "should_write" in parsed:
                    return parsed
            except Exception:
                pass
            # Fallback: find all JSON-like substrings
            json_candidates = re.findall(r'\{.*?\}', text, re.DOTALL)
            for candidate in json_candidates:
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict) and "should_write" in parsed:
                        return parsed
                except Exception:
                    continue
            return None

        parsed = extract_json_with_should_write(cleaned)
        if not parsed:
            print("[summary_node] ERROR: No valid JSON with should_write found!")
            return {}
        print(f"[summary_node] Parsed JSON: {parsed}")
    except json.JSONDecodeError as e:
        print(f"[summary_node] JSON parse error: {e}")
        print(f"[summary_node] Attempted to parse: {json_str if 'json_str' in locals() else cleaned}")
        return {}
    except Exception as e:
        print(f"[summary_node] Unexpected error: {e}")
        return {}

    # Validate structure
    if not isinstance(parsed, dict):
        print(f"[summary_node] Response is not a dict: {type(parsed)}")
        return {}

    should_write = parsed.get("should_write", False)
    print(f"[summary_node] should_write flag: {should_write}")
    
    if not should_write:
        print("[summary_node] Not writing to memory - should_write is False")
        return {}

    # Extract and store memory
    memory_data = parsed.get("memory_to_write", {})
    if not isinstance(memory_data, dict):
        print(f"[summary_node] memory_to_write is not a dict: {type(memory_data)}")
        return {}

    if not memory_data:
        print("[summary_node] memory_to_write is empty")
        return {}

    # Store each key-value pair
    stored_count = 0
    for key, value in memory_data.items():
        try:
            MemoryStore.set(thread_id, key, str(value))
            print(f"[summary_node] Stored: {key} = {value}")
            stored_count += 1
        except Exception as e:
            print(f"[summary_node] Error storing {key}={value}: {e}")

    print(f"[summary_node] Successfully stored {stored_count} memory items")
    return {}

def safe_summary_node(state: AgentState):
    """Wrapper to catch any errors in summary_node"""
    try:
        return summary_node(state)
    except Exception as e:
        print(f"[safe_summary_node] Caught error: {e!r}")
        return {}

## --- Refusal Node ---------------------------------------------------- ##
def refusal_node(state: AgentState) -> Dict[str, Any]:
    return {
        "assistant_reply": (
            "I'm afraid I can't help with that—my scope is limited to data‑analysis "
            "questions. Please rephrase within that domain."
        )
    }

# Debug version of summary_node for standalone testing

def debug_summary_node(state: AgentState) -> Dict[str, Any]:
    """Enhanced debug version of summary_node"""
    print("\n" + "="*60)
    print("[DEBUG_SUMMARY] === STARTING SUMMARY NODE ===")
    print("="*60)
    last_user = getattr(state, 'last_user', None) or getattr(state, 'user_input', None)
    thread_id = getattr(state, 'thread_id', 'unknown_thread')
    print(f"[DEBUG_SUMMARY] Input data:")
    print(f"  - last_user: {repr(last_user)}")
    print(f"  - thread_id: {repr(thread_id)}")
    print(f"  - state type: {type(state)}")
    if not last_user:
        print("[DEBUG_SUMMARY] ERROR: No user message found!")
        return {}
    try:
        prompt = SUMMARY_PROMPT.format(last_user=last_user)
        print(f"[DEBUG_SUMMARY] Calling LLM with prompt:")
        print(f"  {prompt}")
        response = llm.invoke(prompt)
        raw_content = response.content.strip()
        print(f"[DEBUG_SUMMARY] Raw LLM response:")
        print(f"  {repr(raw_content)}")
    except Exception as e:
        print(f"[DEBUG_SUMMARY] LLM call failed: {e}")
        import traceback
        traceback.print_exc()
        return {}
    try:
        cleaned = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
        print(f"[DEBUG_SUMMARY] After cleaning <think> tags:")
        print(f"  {repr(cleaned)}")
        def extract_json_with_should_write(text):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict) and "should_write" in parsed:
                    return parsed
            except Exception:
                pass
            json_candidates = re.findall(r'\{.*?\}', text, re.DOTALL)
            for candidate in json_candidates:
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict) and "should_write" in parsed:
                        return parsed
                except Exception:
                    continue
            return None
        parsed = extract_json_with_should_write(cleaned)
        if not parsed:
            print("[DEBUG_SUMMARY] ERROR: No valid JSON with should_write found!")
            return {}
        print(f"[DEBUG_SUMMARY] Parsed JSON:")
        print(f"  {parsed}")
    except json.JSONDecodeError as e:
        print(f"[DEBUG_SUMMARY] JSON parsing error: {e}")
        print(f"[DEBUG_SUMMARY] Failed to parse: {repr(cleaned)}")
        return {}
    except Exception as e:
        print(f"[DEBUG_SUMMARY] Unexpected error in JSON processing: {e}")
        return {}
    should_write = parsed.get("should_write", False)
    print(f"[DEBUG_SUMMARY] should_write flag: {should_write}")
    if not should_write:
        print("[DEBUG_SUMMARY] Not writing to memory (should_write=False)")
        return {}
    memory_data = parsed.get("memory_to_write", {})
    print(f"[DEBUG_SUMMARY] memory_to_write: {memory_data}")
    if not isinstance(memory_data, dict):
        print(f"[DEBUG_SUMMARY] ERROR: memory_to_write is not a dict: {type(memory_data)}")
        return {}
    if not memory_data:
        print("[DEBUG_SUMMARY] ERROR: memory_to_write is empty!")
        return {}
    print("[DEBUG_SUMMARY] === STORING TO MEMORY ===")
    stored_count = 0
    for key, value in memory_data.items():
        try:
            print(f"[DEBUG_SUMMARY] Storing: {key} = {value}")
            MemoryStore.set(thread_id, key, str(value))
            stored_count += 1
            print(f"[DEBUG_SUMMARY] ✓ Successfully stored {key}")
        except Exception as e:
            print(f"[DEBUG_SUMMARY] ✗ Failed to store {key}: {e}")
    print(f"[DEBUG_SUMMARY] === SUMMARY COMPLETE ===")
    print(f"[DEBUG_SUMMARY] Stored {stored_count} items to memory")
    print("[DEBUG_SUMMARY] === VERIFICATION ===")
    try:
        all_memory = MemoryStore.all(thread_id)
        print(f"[DEBUG_SUMMARY] All memory for thread {thread_id}: {all_memory}")
    except Exception as e:
        print(f"[DEBUG_SUMMARY] Error reading memory: {e}")
    print("="*60)
    return {}

def test_the_issue():
    """Test with the exact case that's failing"""
    print("TESTING THE EXACT FAILING CASE...")
    state = AgentState(
        thread_id="test_thread_001",
        user_input="Is there a significant relationship between column category and column intent? show as chart",
        history="",
        last_user="Is there a significant relationship between column category and column intent? show as chart"
    )
    result = debug_summary_node(state)
    print("\n" + "="*40)
    print("DATABASE CHECK AFTER TEST:")
    print("="*40)
    with sqlite3.connect("agent_memory.db") as conn:
        rows = conn.execute("SELECT * FROM memory WHERE thread_id = 'test_thread_001'").fetchall()
        print(f"Database rows: {rows}")
        all_rows = conn.execute("SELECT * FROM memory").fetchall()
        print(f"All database rows: {all_rows}")

def test_llm_directly():
    """Test the LLM call directly"""
    print("\n" + "="*40)
    print("DIRECT LLM TEST:")
    print("="*40)
    test_message = "Is there a significant relationship between column category and column intent? show as chart"
    prompt = SUMMARY_PROMPT.format(last_user=test_message)
    print(f"Prompt: {prompt}")
    try:
        response = llm.invoke(prompt)
        print(f"Response: {repr(response.content)}")
        content = response.content.strip()
        cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        json_matches = re.findall(r'\{[^{}]*\}', cleaned)
        if json_matches:
            for i, match in enumerate(json_matches):
                try:
                    parsed = json.loads(match)
                    print(f"JSON match {i}: {parsed}")
                except:
                    print(f"JSON match {i}: INVALID - {match}")
    except Exception as e:
        print(f"LLM error: {e}")

# Test function to debug the memory logic
def test_memory_logic():
    """Test function to verify memory extraction works"""
    test_cases = [
        "I'm in Germany and need this as a PDF",
        "Can you analyze my sales data?",
        "I prefer charts for visualization and I'm working on Project Alpha",
        "What's the weather today?",
        "I'm located in Japan, please use local formatting"
    ]
    
    class MockState:
        def __init__(self, msg, tid="test_thread"):
            self.user_input = msg
            self.last_user = msg
            self.thread_id = tid
            self.assistant_reply = None
    
    for i, test_msg in enumerate(test_cases):
        print(f"\n--- Test {i+1}: {test_msg} ---")
        state = MockState(test_msg)
        result = safe_summary_node(state)
        print(f"Result: {result}")

# Uncomment to run tests
# test_memory_logic()