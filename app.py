import os
import json
import time
from typing import List, Dict, Any, Optional
import streamlit as st
from datasets import load_dataset
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI
from tqdm import tqdm
from collections import Counter
from collections import Counter, defaultdict
from scipy.stats import chi2_contingency, ttest_ind, f_oneway

# --- Streamlit Config (must be first Streamlit call) ---
st.set_page_config(page_title="Data Analyst Agent", layout="wide")

# --- Load Nebius API Key ---
KEY_PATH = "Nebius_api_key.txt"
with open(KEY_PATH, "r") as f:
    api_key = f.read().strip()

# --- Initialize OpenAI-Compatible Client ---
client = OpenAI(
    base_url="https://api.studio.nebius.ai/v1/",
    api_key=api_key,
)

# --- Build or load FAISS index and records with persistence ---
@st.cache_resource
def build_faiss_index(model_name: str = "all-MiniLM-L6-v2", batch_size: int = 128):
    print("[DEBUG] Starting build_faiss_index")
    index_path = "faiss.index"
    records_path = "records.json"
    if os.path.exists(index_path) and os.path.exists(records_path):
        print("[DEBUG] Found persisted FAISS index and records, loading...")
        index = faiss.read_index(index_path)
        with open(records_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        # Ensure response_length field exists on loaded records
        for rec in records:
            if "response_length" not in rec:
                rec["response_length"] = len(rec.get("response", "").split())
        ds = load_dataset(
            "bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train"
        )
        model = SentenceTransformer(model_name)
        print("[DEBUG] Loaded FAISS index and enriched records successfully")
        return ds, records, index, model

# Initialize once
if "faiss_ready" not in st.session_state:
    print("[DEBUG] Initializing FAISS resources")
    ds, records, faiss_index, embed_model = build_faiss_index()
    st.session_state.update({
        "ds": ds,
        "records": records,
        "faiss_index": faiss_index,
        "embed_model": embed_model,
        "faiss_ready": True
    })
    print(f"[DEBUG] Loaded {len(records)} records into FAISS index")

# --- Tool functions with debug logs ---
def fetch_relevant_examples(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Retrieve relevant examples from the customer support dataset based on semantic similarity"""
    print(f"[DEBUG] fetch_relevant_examples called with query='{query}', top_k={top_k}")
    q_vec = st.session_state.embed_model.encode([query], convert_to_numpy=True)
    q_vec = np.array(q_vec, dtype="float32")
    D, I = st.session_state.faiss_index.search(q_vec, top_k)
    results = [st.session_state.records[idx] for idx in I[0]]
    print(f"[DEBUG] fetch_relevant_examples returning {len(results)} results")
    return {"results": results}

def compute_dataset_metrics(group_by: str = "intent") -> Dict[str, Any]:
    """Get distribution/counts of different field values in the dataset"""
    print(f"[DEBUG] compute_dataset_metrics called with group_by='{group_by}'")
    vals = [rec[group_by] for rec in st.session_state.records]
    cnt = Counter(vals)
    dist = [{group_by: k, "count": v} for k, v in cnt.most_common()]
    print(f"[DEBUG] compute_dataset_metrics distribution: {dist[:3]}... (+more)")
    return {"distribution": dist}

def count_word_occurrences(field: str = "response") -> Dict[str, Any]:
    """Count most frequent words in a specific field"""
    print(f"[DEBUG] count_word_occurrences called for field='{field}'")
    word_freq = Counter()
    for rec in st.session_state.records:
        for word in rec[field].lower().split():
            word_freq[word] += 1
    top_words = word_freq.most_common(20)
    print(f"[DEBUG] count_word_occurrences top words: {top_words[:5]}")
    return {"top_words": top_words}

def analyze_category_patterns(intent_filter: Optional[str] = None, category_filter: Optional[str] = None) -> Dict[str, Any]:
    """Analyze patterns in categories and intents, with optional filtering"""
    print(f"[DEBUG] analyze_category_patterns called with intent_filter='{intent_filter}', category_filter='{category_filter}'")
    
    filtered_records = st.session_state.records
    
    # Apply filters if provided
    if intent_filter:
        filtered_records = [r for r in filtered_records if intent_filter.lower() in r["intent"].lower()]
    if category_filter:
        filtered_records = [r for r in filtered_records if category_filter.lower() in r["category"].lower()]
    
    # Count patterns
    intent_counts = Counter(r["intent"] for r in filtered_records)
    category_counts = Counter(r["category"] for r in filtered_records)
    
    # Create cross-tabulation
    intent_category_pairs = Counter((r["intent"], r["category"]) for r in filtered_records)
    
    result = {
        "total_filtered_records": len(filtered_records),
        "intent_distribution": dict(intent_counts.most_common()),
        "category_distribution": dict(category_counts.most_common()),
        "intent_category_combinations": dict(intent_category_pairs.most_common(10))
    }
    
    print(f"[DEBUG] analyze_category_patterns returning analysis for {len(filtered_records)} records")
    return result

def analyze_chi_square(col1: str, col2: str) -> Dict[str, Any]:
    # Build contingency table
    table = defaultdict(lambda: defaultdict(int))
    for rec in st.session_state.records:
        table[rec[col1]][rec[col2]] += 1
    rows = list(table.keys())
    cols = sorted({c for row in table.values() for c in row})
    matrix = [[table[r][c] for c in cols] for r in rows]
    chi2, p, dof, expected = chi2_contingency(matrix)
    effect_size = chi2 / (sum(sum(row) for row in matrix) - 1)  # Cramer's V-like effect size
    return {
        "chi2_statistic": chi2,
        "p_value": p,
        "degrees_of_freedom": dof,
        "expected_frequencies": [dict(zip(cols, row)) for row in expected],
        "rows": rows,
        "columns": cols,
        "contingency_matrix": matrix,
        "effect_size": effect_size
    }

def analyze_col_length_by_column(column: str, field: str = "response_length") -> Dict[str, Any]:
    grouped = defaultdict(list)
    for rec in st.session_state["records"]:
        if rec.get(column) and isinstance(rec.get(field), int):
            grouped[rec[column]].append(rec[field])
    results = {}
    for k, lengths in grouped.items():
        results[k] = {
            "mean": float(np.mean(lengths)),
            "std": float(np.std(lengths)),
            "count": len(lengths)
        }
    return {"analysis": results, "grouped_by": column, "field": field}



def analyze_t_test(col1: str, col2: str) -> Dict[str, Any]:
    values1 = [rec[col1] for rec in records if isinstance(rec[col1], (int, float))]
    values2 = [rec[col2] for rec in records if isinstance(rec[col2], (int, float))]
    if not values1 or not values2:
        return {"error": "No numeric data in one or both columns"}
    t_stat, p_value = ttest_ind(values1, values2)
    effect_size = np.abs(np.mean(values1) - np.mean(values2)) / np.sqrt((np.std(values1) ** 2 + np.std(values2) ** 2) / 2)
    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "mean_col1": np.mean(values1),
        "mean_col2": np.mean(values2),
        "std_col1": np.std(values1),
        "std_col2": np.std(values2),
        "effect_size": effect_size,
    }
def analyze_anova_col_length_by_column(column: str, field: str = "response_length") -> Dict[str, Any]:
    grouped = defaultdict(list)
    for rec in st.session_state["records"]:
        if rec.get(column) and isinstance(rec.get(field), int):
            grouped[rec[column]].append(rec[field])

    if len(grouped) < 2:
        return {"error": "Not enough groups to perform ANOVA."}
    
    samples = list(grouped.values())
    f_stat, p_value = f_oneway(*samples)
    effect_size = f_stat / (len(samples) - 1)  # Simple effect size based on F-statistic

    return {
        "f_statistic": f_stat,
        "p_value": p_value,
        "groups": {k: {"mean": np.mean(v), "std": np.std(v), "count": len(v)} for k, v in grouped.items()},
        "grouped_by": column,
        "effect_size": effect_size,
    }

# --- Function registry for dynamic execution ---
AVAILABLE_FUNCTIONS = {
    "fetch_relevant_examples": fetch_relevant_examples,
    "compute_dataset_metrics": compute_dataset_metrics,
    "count_word_occurrences": count_word_occurrences,
    "analyze_category_patterns": analyze_category_patterns,
    "analyze_chi_square": analyze_chi_square,
    "analyze_col_length_by_column": analyze_col_length_by_column,
    "analyze_t_test": analyze_t_test,
    "analyze_anova_col_length_by_column": analyze_anova_col_length_by_column
}

def execute_function_call(func_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a function call with error handling"""
    try:
        if func_name in AVAILABLE_FUNCTIONS:
            # Filter out None values from args
            clean_args = {k: v for k, v in args.items() if v is not None}
            result = AVAILABLE_FUNCTIONS[func_name](**clean_args)
            return {"success": True, "result": result}
        else:
            return {"success": False, "error": f"Unknown function: {func_name}"}
    except Exception as e:
        return {"success": False, "error": f"Function execution error: {str(e)}"}

# --- Improved Agent with better function calling ---
def agentic_generate_answer(user_query: str) -> Dict[str, Any]:
    print(f"[DEBUG] agentic_generate_answer received query: '{user_query}'")
    
    # Define tools in the most compatible format
    tools = [
        {
            "type": "function",
            "function": {
                "name": "fetch_relevant_examples",
                "description": "Retrieve relevant examples from the customer support dataset based on semantic similarity. Use this when users ask about specific topics, examples, or want to see actual data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string", 
                            "description": "The search query to find relevant examples"
                        },
                        "top_k": {
                            "type": "integer", 
                            "description": "Number of examples to retrieve",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "compute_dataset_metrics",
                "description": "Get distribution/counts of different field values in the dataset. Use this when users ask about categories, intents, or general statistics.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "group_by": {
                            "type": "string", 
                            "description": "Field to group by (intent, category, etc.)",
                            "enum": ["intent", "category"],
                            "default": "intent"
                        }
                    },
                    "required": ["group_by"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "count_word_occurrences",
                "description": "Count most frequent words in a specific field. Use this when users ask about common words or text analysis.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "field": {
                            "type": "string", 
                            "description": "Field to analyze (response, instruction, etc.)",
                            "enum": ["response", "text", "instruction"],
                            "default": "response"
                        }
                    },
                    "required": ["field"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "analyze_category_patterns",
                "description": "Analyze patterns between categories and intents, with optional filtering. Use this for cross-tabulation or pattern analysis.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "intent_filter": {
                            "type": "string", 
                            "description": "Filter records by intent (partial match)"
                        },
                        "category_filter": {
                            "type": "string", 
                            "description": "Filter records by category (partial match)"
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "analyze_chi_square",
                "description": "Perform chi-square analysis on two categorical fields to find patterns. Use this for statistical analysis of relationships.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "col1": {
                            "type": "string", 
                            "description": "First categorical field for analysis"
                        },
                        "col2": {
                            "type": "string", 
                            "description": "Second categorical field for analysis"
                        }
                    },
                    "required": ["col1", "col2"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "analyze_col_length_by_column",
                "description": "Analyze the length of responses grouped by a specific column. Use this to understand response lengths across categories or intents.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "column": {
                            "type": "string", 
                            "description": "Column to group by (e.g., category, intent)",
                            "default": "category"
                        },
                        "field": {
                            "type": "string", 
                            "description": "Field to analyze length of (e.g., response_length)",
                            "default": "response_length"
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "analyze_t_test",
                "description": "Perform t-test analysis between two numeric fields. Use this for comparing means of two groups.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "col1": {
                            "type": "string", 
                            "enum": ["response_length", "instruction_length"],
                            "description": "First numeric field for analysis"
                        },
                        "col2": {
                            "type": "string", 
                            "description": "Second numeric field for analysis"
                        }
                    },
                    "required": ["col1", "col2"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "analyze_anova_col_length_by_column",
                "description": "Perform ANOVA analysis on response lengths grouped by a specific column. Use this to compare response lengths across multiple categories or intents.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "column": {
                            "type": "string", 
                            "description": "Column to group by (e.g., category, intent)",
                            "enum": ["category", "intent"],
                            "default": "category"
                        },
                        "field": {
                            "type": "string", 
                            "description": "Field to analyze length of (e.g., response_length)",
                            "enum": ["response_length", "instruction_length"],
                            "default": "response_length"
                        }
                    }
                }
            }
        }
    ]

    # Enhanced system prompt that forces function usage
    system_prompt = """You are a data analyst agent working with a customer support dataset. 

    DATASET STRUCTURE:
    - instruction: customer queries/requests
    - category: broad category of the request  
    - intent: specific intent/purpose
    - response: support response

    CRITICAL INSTRUCTIONS:
    1. You MUST use the provided functions to answer questions - DO NOT provide theoretical responses
    2. For ANY question about the data, immediately call the appropriate function
    3. NEVER say "I would" or "I could" - actually DO it by calling functions
    4. Always start your response by calling a function

    FUNCTION USAGE GUIDE:
    - Categories/Intents distribution → compute_dataset_metrics
    - Specific examples/topics → fetch_relevant_examples  
    - Word frequency analysis → count_word_occurrences
    - Cross-tabulation/patterns → analyze_category_patterns
    - Categorical relationship Statistical analysis (chi-square) → analyze_chi_square
    - Response length analysis → analyze_col_length_by_column
    - T-test for comparing means → analyze_t_test. In a large datast focus on Effect Size
    - ANOVA for comparing multiple groups → analyze_anova_col_length_by_column. In a large dataset focus on Effect Size

    Call the function FIRST, then analyze the actual results returned."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    
    tool_results = {}
    max_iterations = 3
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"[DEBUG] Iteration {iteration}")
        
        try:
            # Make API call with tools
            response = client.chat.completions.create( # Using OpenAI-compatible API
                model="Qwen/Qwen3-30B-A3B",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                function_call="auto",  # Automatically call functions if needed
                temperature=0.1,  # Lower temperature for more consistent function calling
                max_tokens=1500
            )
            
            message = response.choices[0].message # Get the assistant's message
            print(f"[DEBUG] LLM response: {message}")
            
            # Check if the model wants to use tools
            if hasattr(message, 'tool_calls') and message.tool_calls:
                print(f"[DEBUG] Processing {len(message.tool_calls)} tool calls")
                
                # Add the assistant message
                messages.append({
                    "role": "assistant", 
                    "content": message.content,
                    "tool_calls": message.tool_calls
                })
                
                # Execute each tool call
                for tool_call in message.tool_calls: # Assuming tool_call is a structured object
                    func_name = tool_call.function.name # Get the function name
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as e:
                        print(f"[DEBUG] JSON decode error: {e}")
                        args = {}
                    
                    print(f"[DEBUG] Executing {func_name} with args: {args}")
                    
                    # Execute the function
                    execution_result = execute_function_call(func_name, args) # Execute the function call
                    
                    if execution_result["success"]:
                        result = execution_result["result"]
                        tool_results[func_name] = result
                        result_content = json.dumps(result)
                    else:
                        result_content = json.dumps({"error": execution_result["error"]})
                    
                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_content
                    })
                
                # Continue the loop to get the final response
                continue
                
            else:
                # No tool calls - this should be the final answer
                final_answer = message.content
                print(f"[DEBUG] Final answer: {final_answer}")
                
                # If no functions were called and it's the first iteration, force a function call
                if iteration == 1 and not tool_results:
                    print("[DEBUG] No function called on first iteration, trying to force one")
                    
                    # Analyze the query to determine which function to call
                    forced_message = f"""The user asked: "{user_query}"

You must call one of the available functions to get actual data. Based on the query, call the most appropriate function now."""
                    
                    messages.append({"role": "user", "content": forced_message})
                    continue
                    
                return {"answer": final_answer, **tool_results}
                
        except Exception as e:
            print(f"[DEBUG] Error in API call: {e}")
            return {"answer": f"API Error: {str(e)}", **tool_results}
    
    # If we reach max iterations
    return {"answer": "Maximum iterations reached. Please try rephrasing your question.", **tool_results}

# --- Streamlit UI ---
if "history" not in st.session_state:
    st.session_state.history = []

# Debug expander to confirm data access
with st.expander("🔧 Debug: Data Connection & Function Testing"):
    st.write(f"✅ FAISS Ready: {st.session_state.get('faiss_ready', False)}")
    st.write(f"📊 Total records loaded: {len(st.session_state.records) if 'records' in st.session_state else 0}")
    
    if 'records' in st.session_state and st.session_state.records:
        st.write("📝 Example record:")
        st.json(st.session_state.records[0])
    
    st.write("🧪 Test Functions:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Test: Fetch 'refund' examples"):
            result = fetch_relevant_examples("refund", 3)
            st.json(result)
    
    with col2:
        if st.button("Test: Intent metrics"):
            result = compute_dataset_metrics("intent")
            st.json(result)
    
    with col3:
        if st.button("Test: Word analysis"):
            result = count_word_occurrences("response")
            st.json(result)

st.title("🧠 Agentic Data Analyst")
st.write("Ask questions about the customer support dataset. The agent will automatically use the right tools to get you actual data!")

# Sample questions
with st.expander("💡 Example Questions"):
    sample_questions = [
        "What are the most common categories in the dataset?",
        "Show me examples of refund requests",
        "What are the most frequent words in customer responses?",
        "Analyze patterns between billing-related intents and categories",
        "How many different types of intents are there?"
    ]
    for q in sample_questions:
        if st.button(f"Ask: {q}", key=f"sample_{hash(q)}"):
            st.session_state.current_query = q

# Main query input
query = st.text_input("Ask your question about the support dataset:", 
                     value=st.session_state.get('current_query', ''),
                     key="main_query")

if st.button("🔍 Ask", type="primary") and query:
    with st.spinner("Analyzing data..."):
        start_time = time.time()
        output = agentic_generate_answer(query)
        output["duration"] = round(time.time() - start_time, 2)
        st.session_state.history.append({"query": query, **output})
    
    # Clear the query
    if 'current_query' in st.session_state:
        del st.session_state.current_query

# Display conversation history
if st.session_state.history:
    st.write("---")
    st.subheader("📊 Analysis Results")
    
    for i, turn in enumerate(reversed(st.session_state.history)):
        with st.container():
            st.write(f"**Q{len(st.session_state.history)-i}:** {turn['query']}")
            st.write(f"**Answer:** {turn['answer']}")
            
            # Show function results in expandable sections
            func_results_shown = False
            for tool_name in ["fetch_relevant_examples", "compute_dataset_metrics", "count_word_occurrences", "analyze_category_patterns"]:
                if tool_name in turn:
                    func_results_shown = True
                    with st.expander(f"📋 {tool_name.replace('_', ' ').title()} Results"):
                        st.json(turn[tool_name])
            
            if func_results_shown:
                st.caption(f"⏱ Duration: {turn['duration']}s")
            st.write("---")