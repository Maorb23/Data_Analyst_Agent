import os
import json
import time
import warnings
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional, Callable, Union
from datetime import datetime
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.runnables import RunnableConfig
from memory import MemoryStore
from nodes import classify_node, refusal_node, safe_summary_node
from langchain_core.pydantic_v1 import BaseModel, Field
import sqlite3
from graphviz import Digraph

# Tell Streamlit not to watch torch's internals
os.environ["STREAMLIT_WATCH_SERVICE"] = "false"

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")

import streamlit as st
import torch
from datasets import load_dataset
import numpy as np
from collections import Counter, defaultdict
from scipy.stats import chi2_contingency, ttest_ind, f_oneway

# LangChain - Updated imports
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain_community.chat_models import ChatOpenAI

from langchain.agents import initialize_agent
from langchain.tools import Tool
from langchain_community.vectorstores import FAISS

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from tqdm import tqdm

# --- Debugging Infrastructure ---
class DebugLogger:
    def __init__(self):
        self.logs = []
        self.tool_calls = []
        self.iterations = []
        self.current_iteration = 0
        
    def log(self, level: str, message: str, data: Any = None):
        """Add a debug log entry"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "data": data,
            "iteration": self.current_iteration
        }
        self.logs.append(entry)
        
    def log_tool_call(self, tool_name: str, args: Dict, result: Any, duration: float, error: str = None):
        """Log a tool call with its parameters and results"""
        call = {
            "timestamp": datetime.now().isoformat(),
            "tool_name": tool_name,
            "arguments": args,
            "result": result,
            "duration": duration,
            "error": error,
            "iteration": self.current_iteration
        }
        self.tool_calls.append(call)
        
    def start_iteration(self, thought: str = ""):
        """Mark the start of a new agent iteration"""
        self.current_iteration += 1
        iteration = {
            "iteration": self.current_iteration,
            "start_time": time.time(),
            "thought": thought,
            "tool_calls": [],
            "end_time": None,
            "duration": None
        }
        self.iterations.append(iteration)
        
    def end_iteration(self):
        """Mark the end of the current iteration"""
        if self.iterations:
            current = self.iterations[-1]
            current["end_time"] = time.time()
            current["duration"] = current["end_time"] - current["start_time"]
            # Collect tool calls from this iteration
            current["tool_calls"] = [
                call for call in self.tool_calls 
                if call["iteration"] == self.current_iteration
            ]
    
    def clear(self):
        """Clear all debug logs"""
        self.logs.clear()
        self.tool_calls.clear()
        self.iterations.clear()
        self.current_iteration = 0

# Initialize debug logger
if "debug_logger" not in st.session_state:
    st.session_state.debug_logger = DebugLogger()

debug_logger = st.session_state.debug_logger

# --- Tool Wrapper for Debugging ---
def debug_tool_wrapper(func: Callable, tool_name: str) -> Callable:
    """Wrapper to add debugging to tool functions"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        debug_logger.log("INFO", f"Calling tool: {tool_name}", {"args": args, "kwargs": kwargs})
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            debug_logger.log_tool_call(tool_name, kwargs or {}, result, duration)
            debug_logger.log("SUCCESS", f"Tool {tool_name} completed", {"duration": duration})
            return result
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            debug_logger.log_tool_call(tool_name, kwargs or {}, None, duration, error_msg)
            debug_logger.log("ERROR", f"Tool {tool_name} failed", {"error": error_msg, "duration": duration})
            raise e
    return wrapper

# --- Streamlit Config ---
st.set_page_config(page_title="Data Analyst Agent with Debug", layout="wide")

# --- Load Nebius API Key ---
KEY_PATH = "Nebius_api_key.txt"
with open(KEY_PATH, "r") as f:
    NEBIUS_API_KEY = f.read().strip()

# --- Resources Loader (cached) ---
INDEX_DIR = "faiss_store"   # a folder name, not ending in ".index"

@st.cache_resource
def get_resources(
    index_dir: str = "faiss_store",
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 128,
) -> Tuple[FAISS, List[Dict[str, Any]]]:
    # 1) Load the dataset
    debug_logger.log("INFO", "Loading dataset")
    ds = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split="train"
    )
    records: List[Dict[str, Any]] = ds.to_pandas().to_dict(orient="records")
    debug_logger.log("INFO", f"Dataset loaded: {len(records)} records")

    # 2) Instantiate embeddings (use GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    debug_logger.log("INFO", f"Using device: {device}")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device}
    )

    # 3) Load or build FAISS store
    if os.path.isdir(index_dir) and os.path.exists(os.path.join(index_dir, "index.faiss")):
        debug_logger.log("INFO", f"Loading existing FAISS index from {index_dir}")
        vectorstore = FAISS.load_local(
            index_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        debug_logger.log("INFO", "Building new FAISS index")
        texts = [r.get("response", "") for r in records]
        # Compute embeddings in batches with progress bar
        all_embeddings: List[List[float]] = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch = texts[i : i + batch_size]
            emb_batch = embeddings.embed_documents(batch)
            all_embeddings.extend(emb_batch)
            debug_logger.log("DEBUG", f"Processed batch {i//batch_size + 1}")
        
        # Pair each text with its embedding
        text_embeddings: List[Tuple[str, List[float]]] = list(zip(texts, all_embeddings))
        # Build FAISS index from precomputed embeddings
        vectorstore = FAISS.from_embeddings(
            text_embeddings,
            embeddings,
            metadatas=records,
        )
        vectorstore.save_local(index_dir)
        debug_logger.log("INFO", f"FAISS index saved to {index_dir}")

    return vectorstore, records

# --- Initialize resources once ---
if "vectorstore" not in st.session_state:
    vectorstore, records = get_resources()
    st.session_state["vectorstore"] = vectorstore
    st.session_state["records"] = records

# --- Analytics Wrappers ---
def fetch_relevant_examples(query: str, top_k: int = 50) -> List[Dict[str, Any]]:
    """Retrieve semantically similar customer-support examples."""
    debug_logger.log("INFO", f"Fetching {top_k} examples for query: '{query}'")
    docs = st.session_state["vectorstore"].similarity_search(query, k=top_k)
    results = [d.metadata for d in docs]
    debug_logger.log("SUCCESS", f"Found {len(results)} relevant examples")
    return results

def count_word_occurrences(field: str = "response") -> Dict[str, Any]:
    """Count the most frequent words in a specific text field."""
    debug_logger.log("INFO", f"Counting word occurrences in field: {field}")
    freq = Counter()
    for rec in st.session_state["records"]:
        text = rec.get(field, "").lower().split()
        freq.update(text)
    top = freq.most_common(20)
    debug_logger.log("SUCCESS", f"Found {len(freq)} unique words, returning top 20")
    return {"top_words": top}

def analyze_category_patterns(
    intent_filter: Optional[str] = None, category_filter: Optional[str] = None
) -> Dict[str, Any]:
    """Cross-tabulate and count patterns between intent and category."""
    debug_logger.log("INFO", f"Analyzing patterns with filters - intent: {intent_filter}, category: {category_filter}")
    recs = st.session_state["records"]
    original_count = len(recs)
    
    if intent_filter:
        recs = [r for r in recs if intent_filter.lower() in r.get("intent", "").lower()]
        debug_logger.log("DEBUG", f"After intent filter: {len(recs)} records")
    if category_filter:
        recs = [r for r in recs if category_filter.lower() in r.get("category", "").lower()]
        debug_logger.log("DEBUG", f"After category filter: {len(recs)} records")
    
    intent_counts = Counter(r.get("intent") for r in recs)
    category_counts = Counter(r.get("category") for r in recs)
    combo = Counter((r.get("intent"), r.get("category")) for r in recs)
    
    result = {
        "total_filtered_records": len(recs),
        "original_records": original_count,
        "intent_distribution": dict(intent_counts.most_common()),
        "category_distribution": dict(category_counts.most_common()),
        "intent_category_combinations": dict(combo.most_common(10)),
    }
    
    debug_logger.log("SUCCESS", f"Pattern analysis complete: {len(recs)} records analyzed")
    return result

import json
from collections import defaultdict
from scipy.stats import chi2_contingency
import streamlit as st

def analyze_chi_square(input_param, extra=None):
    """Perform a chi-square test between two categorical fields with flexible input parsing."""
    
    # ── 1) Normalize input into two column names ────────────────────────────
    if isinstance(input_param, dict):
        col1 = input_param.get("col1", "").strip()
        col2 = input_param.get("col2", "").strip()
    elif isinstance(input_param, str):
        raw = input_param.strip()
        # If it looks like JSON, parse it
        if (raw.startswith("{") and raw.endswith("}")) or raw.endswith("}\n"):
            try:
                parsed = json.loads(raw)
                col1 = parsed.get("col1", "").strip()
                col2 = parsed.get("col2", "").strip()
            except json.JSONDecodeError:
                # Not valid JSON → treat as single column name (need both columns)
                return {"error": "Chi-square test requires two columns. Please specify both col1 and col2."}
        else:
            # Plain string → assume it's col1, use extra as col2
            col1 = raw
            col2 = extra if isinstance(extra, str) else ""
    else:
        return {"error": "Invalid input type. Expected dict, JSON string, or column name."}
    
    # ── 2) Validate we have both columns ────────────────────────────────────
    if not col1 or not col2:
        return {"error": "Both col1 and col2 are required for chi-square test."}
    
    # ── 3) Check columns exist in data ──────────────────────────────────────
    records = st.session_state.get("records", [])
    if not records:
        return {"error": "No records loaded"}
    
    sample_keys = list(records[0].keys())
    if col1 not in sample_keys:
        return {"error": f"Column '{col1}' not found. Available: {sample_keys}"}
    if col2 not in sample_keys:
        return {"error": f"Column '{col2}' not found. Available: {sample_keys}"}
    
    debug_logger.log("INFO", f"Performing chi-square test between {col1} and {col2}")
    
    # ── 4) Build contingency table ──────────────────────────────────────────
    table = defaultdict(lambda: defaultdict(int))
    
    for r in records:
        val1 = r.get(col1)
        val2 = r.get(col2)
        if val1 is not None and val2 is not None:
            table[val1][val2] += 1
    
    if not table:
        return {"error": "No valid data pairs found for chi-square test"}
    
    rows = list(table.keys())
    cols = sorted({c for row in table.values() for c in row})
    matrix = [[table[r][c] for c in cols] for r in rows]
    
    debug_logger.log("DEBUG", f"Contingency table: {len(rows)} rows x {len(cols)} columns")
    
    # ── 5) Perform chi-square test ──────────────────────────────────────────
    try:
        chi2, p, dof, exp = chi2_contingency(matrix)
        effect = chi2 / (sum(sum(r) for r in matrix) - 1)
        
        result = {
            "chi2_statistic": chi2,
            "p_value": p,
            "degrees_of_freedom": dof,
            "expected_frequencies": [dict(zip(cols, row)) for row in exp],
            "rows": rows,
            "columns": cols,
            "contingency_matrix": matrix,
            "effect_size": effect,
            "col1": col1,
            "col2": col2
        }
        
        debug_logger.log("SUCCESS", f"Chi-square test complete: χ² = {chi2:.4f}, p = {p:.4f}")
        return result
        
    except Exception as e:
        debug_logger.log("ERROR", f"Chi-square calculation failed: {str(e)}")
        return {"error": f"Chi-square calculation failed: {str(e)}"}



def analyze_t_test(input_param, extra=None):
    """Perform an independent t-test between two numeric fields with flexible input parsing."""
    
    # ── 1) Normalize input into two column names ────────────────────────────
    if isinstance(input_param, dict):
        col1 = input_param.get("col1", "").strip()
        col2 = input_param.get("col2", "").strip()
    elif isinstance(input_param, str):
        raw = input_param.strip()
        # If it looks like JSON, parse it
        if (raw.startswith("{") and raw.endswith("}")) or raw.endswith("}\n"):
            try:
                parsed = json.loads(raw)
                col1 = parsed.get("col1", "").strip()
                col2 = parsed.get("col2", "").strip()
            except json.JSONDecodeError:
                return {"error": "T-test requires two columns. Please specify both col1 and col2."}
        else:
            # Plain string → assume it's col1, use extra as col2
            col1 = raw
            col2 = extra if isinstance(extra, str) else ""
    else:
        return {"error": "Invalid input type. Expected dict, JSON string, or column name."}
    
    # ── 2) Validate we have both columns ────────────────────────────────────
    if not col1 or not col2:
        return {"error": "Both col1 and col2 are required for t-test."}
    
    # ── 3) Check columns exist and get data ─────────────────────────────────
    records = st.session_state.get("records", [])
    if not records:
        return {"error": "No records loaded"}
    
    sample_keys = list(records[0].keys())
    if col1 not in sample_keys:
        return {"error": f"Column '{col1}' not found. Available: {sample_keys}"}
    if col2 not in sample_keys:
        return {"error": f"Column '{col2}' not found. Available: {sample_keys}"}
    
    debug_logger.log("INFO", f"Performing t-test between {col1} and {col2}")
    
    # ── 4) Extract numeric values ───────────────────────────────────────────
    v1 = [r[col1] for r in records if isinstance(r.get(col1), (int, float))]
    v2 = [r[col2] for r in records if isinstance(r.get(col2), (int, float))]
    
    debug_logger.log("DEBUG", f"Sample sizes: {col1}={len(v1)}, {col2}={len(v2)}")
    
    if not v1 or not v2:
        return {"error": f"One or both fields lack numeric data. {col1}: {len(v1)} values, {col2}: {len(v2)} values"}
    
    # ── 5) Perform t-test ────────────────────────────────────────────────────
    try:
        t_stat, p = ttest_ind(v1, v2)
        effect = abs(np.mean(v1) - np.mean(v2)) / np.sqrt((np.std(v1)**2 + np.std(v2)**2) / 2)
        
        result = {
            "t_statistic": t_stat,
            "p_value": p,
            "mean_" + col1: np.mean(v1),
            "mean_" + col2: np.mean(v2),
            "std_" + col1: np.std(v1),
            "std_" + col2: np.std(v2),
            "effect_size": effect,
            "col1": col1,
            "col2": col2,
            "sample_size_" + col1: len(v1),
            "sample_size_" + col2: len(v2)
        }
        
        debug_logger.log("SUCCESS", f"T-test complete: t = {t_stat:.4f}, p = {p:.4f}")
        return result
        
    except Exception as e:
        debug_logger.log("ERROR", f"T-test calculation failed: {str(e)}")
        return {"error": f"T-test calculation failed: {str(e)}"}

import json
from collections import Counter, defaultdict
from typing import Any, Dict, Optional
import streamlit as st

import json
from collections import Counter, defaultdict
from typing import Any, Dict, Optional
import streamlit as st

def plot_column_distribution(input_param: Any, extra: Any = None) -> Dict[str, Any]:
    """Generate and store a bar chart for column distribution."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import json

    # ── 1) Normalize input_param into three variables ──
    if isinstance(input_param, dict):
        col = input_param.get("column", "").strip()
        subset_by = input_param.get("subset_by", "").strip()
        subset_value = input_param.get("subset_value", "").strip()
    elif isinstance(input_param, str):
        raw = input_param.strip()
        
        if raw.startswith("{") and (raw.endswith("}") or raw.endswith("}\n")):
            try:
                json_str = raw.rstrip('\n')
                parsed = json.loads(json_str)
                col = parsed.get("column", "").strip()
                subset_by = parsed.get("subset_by", "").strip()
                subset_value = parsed.get("subset_value", "").strip()
            except json.JSONDecodeError as e:
                col = raw
                subset_by = extra if isinstance(extra, str) else ""
                subset_value = ""
        else:
            col = raw
            subset_by = extra if isinstance(extra, str) else ""
            subset_value = ""
    else:
        return {"error": f"Invalid input type: {type(input_param)}. Expected dict, JSON string, or column name."}

    # ── 2) Validate inputs ──
    if not col:
        return {"error": "'column' is required. Example: {\"column\":\"category\"}"}
    
    records = st.session_state.get("records", [])
    if not records:
        return {"error": "No records loaded"}

    sample_keys = list(records[0].keys())
    if col not in sample_keys:
        return {"error": f"Column '{col}' not found. Available columns: {sample_keys}"}
    if subset_by and subset_by not in sample_keys:
        return {"error": f"'subset_by' column '{subset_by}' not found. Available columns: {sample_keys}"}

    # ── 3) Create DataFrame and apply filters ──
    df = pd.DataFrame(records)
    original_count = len(df)
    
    if subset_by and subset_value:
        df = df[df[subset_by] == subset_value]
        filtered_count = len(df)
        if filtered_count == 0:
            return {"error": f"No records found where {subset_by} = {subset_value}"}
    else:
        filtered_count = original_count

    # ── 4) Create the plot ──
    try:
        plt.style.use('default')  # Reset any previous style
        fig, ax = plt.subplots(figsize=(10, 6))
        
        counts = df[col].value_counts().sort_index()
        
        # Create bar plot
        bars = ax.bar(range(len(counts)), counts.values)
        
        # Customize the plot
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=45, ha='right')
        
        title = f"Distribution of {col}"
        subset_info = ""
        if subset_by and subset_value:
            title += f" (where {subset_by} = {subset_value})"
            subset_info = f"{subset_by} = {subset_value}"
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel("Count", fontsize=12)
        ax.set_xlabel(col, fontsize=12)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # ── 5) Store the plot in session state for display ──
        if "current_plots" not in st.session_state:
            st.session_state.current_plots = []
        
        plot_info = {
            "title": title,
            "figure": fig,
            "subset_info": subset_info,
            "column": col,
            "total_records": filtered_count,
            "unique_values": len(counts)
        }
        
        st.session_state.current_plots.append(plot_info)
        
        # ── 6) Return summary for the agent ──
        summary = {
            "success": True,
            "message": f"Created bar chart for column '{col}'",
            "column": col,
            "total_records": filtered_count,
            "unique_values": len(counts),
            "top_5_values": dict(counts.head().items()),
            "subset_applied": bool(subset_by and subset_value),
            "subset_info": subset_info
        }
        
        debug_logger.log("SUCCESS", f"Plot created for column '{col}'", summary)
        return summary
        
    except Exception as e:
        error_msg = f"Failed to create plot: {str(e)}"
        debug_logger.log("ERROR", error_msg)
        return {"error": error_msg}








def analyze_col_length_by_column(input_param: Any, extra: Any = None) -> Dict[str, Any]:
    """
    Compute summary statistics (mean, std, count, min, max) of a numeric field
    grouped by a categorical column.

    The agent might call this tool by passing:
      • a plain string:       "category"
      • a dict:               {"column":"category","field":"response_length"}
      • a JSON‐encoded string:'{"column":"category","field":"response_length"}\n'

    If `field=="response_length"`, we recalc it via word‐count of `response`.
    """

    # ── 1) “Unwrap” the agent’s input into two plain strings col, fld ─────────
    if isinstance(input_param, dict):
        col = input_param.get("column", "").strip()
        fld = input_param.get("field", "response_length").strip()
    elif isinstance(input_param, str):
        raw = input_param.strip()
        # If it’s valid JSON or ends with '}\n', parse it:
        if (raw.startswith("{") and raw.endswith("}")) or raw.endswith("}\n"):
            try:
                parsed = json.loads(raw)
                col = parsed.get("column", "").strip()
                fld = parsed.get("field", "response_length").strip()
            except json.JSONDecodeError:
                # Not valid JSON → treat raw as the column name
                col = raw
                fld = extra if isinstance(extra, str) else "response_length"
        else:
            # A bare string → that’s the column name
            col = raw
            fld = extra if isinstance(extra, str) else "response_length"
    else:
        return {"error": "invalid input—must be dict, JSON string, or plain string"}

    # ── 2) Grab all records ─────────────────────────────────────────────────
    records = st.session_state.get("records", [])
    if not records:
        return {"error": "no records loaded"}

    # ── 3) Verify that `col` actually exists ─────────────────────────────────
    sample_keys = list(records[0].keys())
    if col not in sample_keys:
        return {
            "error": f"column '{col}' not found. available keys: {sample_keys}"
        }

    # ── 4) Ensure numeric field exists (recompute if missing) ────────────────
    for r in records:
        if fld == "response_length":
            # If it doesn’t exist or isn’t numeric, recalc from `response`
            if not isinstance(r.get(fld), (int, float)):
                r[fld] = len(r.get("response", "").split())
        else:
            # If they requested some other numeric field, try to cast strings→float
            v = r.get(fld)
            if isinstance(v, str) and v.replace("-", "").replace(".", "").isdigit():
                try:
                    r[fld] = float(v)
                except:
                    pass

    # ── 5) Group values by category, skip invalid rows ────────────────────────
    grouped = defaultdict(list)
    total_rows = 0
    skipped   = 0

    for r in records:
        key = r.get(col)
        val = r.get(fld)
        total_rows += 0  # still count total records, even if invalid

        if key is None or not isinstance(val, (int, float)):
            skipped += 1
            continue

        grouped[key].append(float(val))

    # ── 6) Build summary stats per group ─────────────────────────────────────
    analysis = {}
    for k, arr in grouped.items():
        if len(arr) == 0:
            continue
        a = np.array(arr)
        analysis[k] = {
            "mean": float(a.mean()),
            "std":  float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "count": len(a),
            "min":  float(a.min()),
            "max":  float(a.max())
        }

    # If every group is empty—or only one group exists—ANOVA will fail later.
    if len(analysis) < 2:
        return {
            "error": f"Need ≥2 groups with ≥1 sample each. Found {len(analysis)} valid group(s).",
            "group_sizes": {k: len(grouped[k]) for k in grouped},
            "total_records": len(records),
            "skipped_records": skipped
        }

    return {
        "groups": analysis,
        "grouped_by": col,
        "field": fld,
        "total_records": len(records),
        "skipped_records": skipped
    }


import json
from collections import defaultdict, Counter
from typing import Dict, Any

def compute_dataset_metrics(group_by: Any = "intent") -> Dict[str, Any]:
    """
    Input normalization: accept either
      - a bare string (e.g. "category")
      - a dict like {"group_by": "category"}
      - a JSON‐encoded string like '{"group_by": "category"}\n'
    Then group records by that column and sum token counts.
    """
    # 1) Normalize the incoming argument to a simple string
    if isinstance(group_by, dict):
        col = group_by.get("group_by", "").strip()
    elif isinstance(group_by, str):
        raw = group_by.strip()
        # if it looks like '{"group_by": "…"}', parse it
        if raw.startswith("{") and raw.endswith("}") or raw.endswith("}\n"):
            try:
                parsed = json.loads(raw)
                col = parsed.get("group_by", "").strip()
            except json.JSONDecodeError:
                col = raw  # fallback to using the raw string
        else:
            col = raw
    else:
        return {"error": "invalid input type: must be str or dict"}

    # 2) Validate column name
    records = st.session_state.get("records", [])
    if not records:
        return {"error": "no records loaded"}

    if col not in records[0]:
        return {
            "error": f"column '{col}' not found. available keys: {list(records[0].keys())}"
        }

    # 3) Accumulate token counts
    grouped_totals = defaultdict(lambda: {"total_tokens": 0, "num_records": 0})
    total_processed = 0

    for r in records:
        key = r.get(col)
        if key is None:
            continue
        response_text = r.get("response", "") or ""
        token_count = len(response_text.split())

        grouped_totals[key]["total_tokens"] += token_count
        grouped_totals[key]["num_records"] += 1
        total_processed += 1

    if total_processed == 0:
        return {"error": f"no non-null values found in '{col}' column"}

    return {
        "group_by": col,
        "total_records": total_processed,
        "groups": dict(grouped_totals)
    }





# ─── 4. Analyze ANOVA on response_length by category ───────────────────────
import json
import numpy as np
from collections import defaultdict
from scipy.stats import f_oneway
from typing import Any, Dict
import streamlit as st

def analyze_anova_col_length_by_column(input_param: Any, extra: Any = None) -> Dict[str, Any]:
    """
    Perform one-way ANOVA on a numeric field grouped by a categorical column.

    The agent may pass:
      • a dict:        {"column": "category", "field": "response_length"}
      • a JSON string: '{"column":"category","field":"response_length"}\n'
      • a plain str:   "category"  (then we default field→"response_length")

    Returns either:
      • ANOVA results if ≥2 valid groups, or
      • an informative error if something is wrong.
    """

    # ── 1) Normalize input into two strings: col and fld ────────────────────
    if isinstance(input_param, dict):
        col = input_param.get("column", "").strip()
        fld = input_param.get("field", "response_length").strip()
    elif isinstance(input_param, str):
        raw = input_param.strip()
        # If it looks like JSON, parse it
        if (raw.startswith("{") and raw.endswith("}")) or raw.endswith("}\n"):
            try:
                parsed = json.loads(raw)
                col = parsed.get("column", "").strip()
                fld = parsed.get("field", "response_length").strip()
            except json.JSONDecodeError:
                # Not valid JSON → treat the entire raw as the column name
                col = raw
                fld = extra if isinstance(extra, str) else "response_length"
        else:
            # Bare string → no JSON: raw is the column name
            col = raw
            fld = extra if isinstance(extra, str) else "response_length"
    else:
        return {"error": "invalid input—expected dict, JSON string, or plain string"}

    # ── 2) Fetch all records ─────────────────────────────────────────────────
    records = st.session_state.get("records", [])
    if not records:
        return {"error": "no records loaded"}

    # ── 3) Check that the column really exists ───────────────────────────────
    sample_keys = list(records[0].keys())
    if col not in sample_keys:
        return {
            "error": f"column '{col}' not found. available keys: {sample_keys}"
        }

    # ── 4) Ensure numeric field exists (compute from response if needed) ─────
    for r in records:
        # If user wants "response_length", compute it from `response`
        if fld == "response_length":
            if fld not in r or not isinstance(r.get(fld), (int, float)):
                r[fld] = len(r.get("response", "").split())
        else:
            # If they specified a different numeric field, try converting str→float
            v = r.get(fld)
            if isinstance(v, str) and v.replace("-", "").replace(".", "").isdigit():
                try:
                    r[fld] = float(v)
                except:
                    pass

    # ── 5) Group values by category ─────────────────────────────────────────
    grouped = defaultdict(list)
    total_rows = 0
    skipped   = 0

    for r in records:
        key = r.get(col)
        val = r.get(fld)
        total_rows += 1

        if key is None or not isinstance(val, (int, float)):
            skipped += 1
            continue

        grouped[key].append(float(val))

    # ── 6) Filter only groups that have ≥2 samples ───────────────────────────
    valid = {k: v for k, v in grouped.items() if len(v) >= 2}
    if len(valid) < 2:
        return {
            "error": f"Need ≥2 groups with ≥2 observations. Found {len(valid)} valid group(s).",
            "group_sizes": {k: len(v) for k, v in grouped.items()},
            "total_rows": total_rows,
            "skipped_rows": skipped
        }

    # ── 7) Run the ANOVA test ────────────────────────────────────────────────
    try:
        samples = list(valid.values())
        f_stat, p_val = f_oneway(*samples)
        all_vals  = np.concatenate(samples)
        mu_total  = all_vals.mean()
        ss_total  = ((all_vals - mu_total) ** 2).sum()

        ss_between = 0.0
        for arr in samples:
            a = np.array(arr)
            n_k  = len(a)
            mu_k = a.mean()
            ss_between += n_k * (mu_k - mu_total) ** 2

        effect_size = ss_between / ss_total
        debug_logger.log("DEBUG", f"ANOVA samples: {len(samples)} groups")
    except Exception as e:
        return {
            "error": f"ANOVA calculation failed: {str(e)}",
            "group_sizes": {k: len(v) for k, v in valid.items()}
        }

    # ── 8) Build per-group statistics: mean, std, count, min, max ───────────
    stats = {}
    for k, arr in valid.items():
        a = np.array(arr)
        stats[k] = {
            "mean": float(a.mean()),
            "std":  float(a.std(ddof=1)),
            "count": len(a),
            "min":  float(a.min()),
            "max":  float(a.max())
        }

    return {
        "f_statistic": float(f_stat),
        "p_value": float(p_val),
        "effect_size": float(effect_size),
        "groups": stats,
        "grouped_by": col,
        "field": fld,
        "total_rows": total_rows,
        "skipped_rows": skipped
    }


# Simple tool definitions
TOOLS = [
    Tool.from_function(
        name="fetch_relevant_examples",
        func=fetch_relevant_examples,
        description="Get semantically similar examples. Args: query (str), top_k (int, default=5)",
    ),
    Tool.from_function(
        name="count_word_occurrences",
        func=count_word_occurrences,
        description="Count word frequencies in a field. Args: field (str, default='response')",
    ),
    Tool.from_function(
        name="analyze_category_patterns",
        func=analyze_category_patterns,
        description="Cross-tabulate intent/category patterns. Args: intent_filter (str, optional), category_filter (str, optional)",
    ),
    Tool.from_function(
        name="analyze_chi_square",
        func=analyze_chi_square,
        description= "Perform a chi-square test between two categorical fields, Args (both required): col1 (first categorical column), col2 (second categorical column), Requires both col1 and col 2. ⚠️ You must pass exactly a JSON object with two keys:",
    ),
    Tool.from_function(
        name="analyze_t_test",
        func=analyze_t_test,
        description="Perform t-test between two numeric fields. Args: col1 (str), col2 (str)",
    ),
    Tool.from_function(
        name="compute_dataset_metrics",
        func=compute_dataset_metrics,
        description="Get distribution counts by field. Args: group_by (str, default='intent')"
    ),
    Tool.from_function(
        name="analyze_col_length_by_column",
        func=analyze_col_length_by_column,
        description="Get statistics of numeric field by category. Args: column (str), field (str, default='response_length')"
    ),
    Tool.from_function(
        name="analyze_anova_col_length_by_column",
        func=analyze_anova_col_length_by_column,
        description="Perform ANOVA on numeric field by category. Args: column (str), field (str, default='response_length'). If the dataset is large focus on effect size",
    ),
    Tool.from_function(
    name="plot_column_distribution", 
    func=plot_column_distribution,
    description="Generate bar chart. Pass JSON or plain column name."
    ),
]

# --- LLM & Agent Setup ---
llm = ChatOpenAI(
    model="Qwen/Qwen3-30B-A3B",
    temperature=0.1,
    openai_api_key=NEBIUS_API_KEY,
    openai_api_base="https://api.studio.nebius.ai/v1",
)

# Add a simple LLM-based summarization tool
from langchain_core.runnables import Runnable

def summarize_text(text: str = None, column: str = None) -> dict:
    """Summarize the given text or the content of a column from the dataset using the LLM. Robust to case/whitespace in column names."""
    import streamlit as st
    # If a column is specified, summarize that column from the records
    if column:
        records = st.session_state.get("records", [])
        if not records:
            return {"error": "No records loaded"}
        sample_keys = list(records[0].keys())
        # Normalize keys for robust matching
        normalized_keys = {k.strip().lower(): k for k in sample_keys}
        requested = column.strip().lower()
        if requested not in normalized_keys:
            return {"error": f"Column '{column}' not found. Available columns: {sample_keys}. Normalized: {list(normalized_keys.keys())}"}
        real_column = normalized_keys[requested]
        # Concatenate all values in the column (as text)
        values = [str(r.get(real_column, "")) for r in records if r.get(real_column)]
        if not values:
            return {"error": f"No data found in column '{real_column}'"}
        text_to_summarize = "\n".join(values)
        prompt = f"Summarize the following data from the '{real_column}' column as a helpful analyst:\n\n{text_to_summarize}\n\nSummary:"
    elif text:
        prompt = f"Summarize the following text as a helpful analyst:\n\n{text}\n\nSummary:"
    else:
        return {"error": "Please provide either 'text' or 'column' to summarize."}
    summary = llm.invoke(prompt).content.strip()
    return {"summary": summary}

# Update the tool definition to accept both text and column
UNSTRUCTURED_TOOLS = [
    Tool.from_function(
        name="fetch_relevant_examples",
        func=fetch_relevant_examples,
        description="Get semantically similar examples. Args: query (str), top_k (int, default=5)",
    ),
    Tool.from_function(
        name="count_word_occurrences",
        func=count_word_occurrences,
        description="Count word frequencies in a field. Args: field (str, default='response')",
    ),
    Tool.from_function(
        name="analyze_category_patterns",
        func=analyze_category_patterns,
        description="Cross-tabulate and count patterns between intent and category. Args: intent_filter (str, optional), category_filter (str, optional)",
    ),
    Tool.from_function(
        name="summarize_text",
        func=summarize_text,
        description="Summarize a given text or a column from the dataset. Args: text (str, optional), column (str, optional)",
    ),
]

unstructured_agent = initialize_agent(
    tools=UNSTRUCTURED_TOOLS,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True,
)

# --- Custom Agent Callback for Debugging ---
class DebugAgentCallback:
    def __init__(self, logger):
        self.logger = logger
        
    def on_agent_action(self, action, **kwargs):
        self.logger.start_iteration(f"Action: {action.tool}")
        self.logger.log("INFO", f"Agent action: {action.tool}", {"input": action.tool_input})
        
    def on_agent_finish(self, finish, **kwargs):
        self.logger.end_iteration()
        self.logger.log("INFO", "Agent finished", {"output": finish.return_values})

# --- LLM & Agent Setup ---
llm = ChatOpenAI(
    model="Qwen/Qwen3-30B-A3B",
    temperature=0.1,
    openai_api_key=NEBIUS_API_KEY,
    openai_api_base="https://api.studio.nebius.ai/v1",
)

#memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent = initialize_agent(
    tools=TOOLS,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,  # Enable verbose for more debugging info
    handle_parsing_errors=True,  # Allow agent to retry on output parsing errors
)

# ── Compile analyst graph with checkpointing ───────────────────────────────
class AgentState(BaseModel):
    thread_id: str
    user_input: str
    history: str
    last_user: str
    # populated downstream:
    route: Optional[str] = Field(default=None)
    assistant_reply: Optional[str] = None
    
# ── Structured‑agent node (LangGraph expects a dict in / dict out) ─────────
def structured_agent_node(state: AgentState) -> Dict[str, Any]:
    from memory import MemoryStore
    # Load persistent memory for this thread
    memory = MemoryStore.all(state.thread_id)
    memory_context = ""
    if memory:
        memory_context = "# User Preferences and Context\n" + "\n".join(f"- {k}: {v}" for k, v in memory.items()) + "\n\n"
    # Build prompt with memory, history, and current input
    prompt = f"{memory_context}{state.history}\nUser: {state.user_input}\nAssistant:"
    result = agent.invoke({"input": prompt})
    return {"assistant_reply": result.get("output", "")}

def unstructured_agent_node(state: AgentState) -> Dict[str, Any]:
    from memory import MemoryStore
    memory = MemoryStore.all(state.thread_id)
    memory_context = ""
    if memory:
        memory_context = "# User Preferences and Context\n" + "\n".join(f"- {k}: {v}" for k, v in memory.items()) + "\n\n"
    prompt = (
        f"{memory_context}"
        f"{state.history}\n"
        f"User: {state.user_input}\n"
        f"Assistant (answer as a helpful analyst: summarize, explain, or provide qualitative insight as needed):"
    )
    result = unstructured_agent.invoke({"input": prompt})
    return {"assistant_reply": result.get("output", "")}

# sqlite_conn = sqlite3.connect("graph_checkpoint.db")

sqlite_conn = sqlite3.connect(
    "graph_checkpoint.db",
    check_same_thread=False,
)

# 2) Pass that Connection into SqliteSaver
cp = SqliteSaver(sqlite_conn)
graph = StateGraph(AgentState)

graph.add_node("classifier",   classify_node)
graph.add_node("structured",   structured_agent_node)
graph.add_node("unstructured", unstructured_agent_node)
graph.add_node("refuse",       refusal_node)
graph.add_node("summary",      safe_summary_node)

# ── conditional routing out of the classifier ─────────────────────────────
graph.add_conditional_edges(
    "classifier",                       # source
    lambda s: s.route,                  # selector: pick state.route
    {
        "structured":   "structured",
        "unstructured": "unstructured",
        "refuse":       "refuse",
    },
)

# ── Set the entry point (CORRECTED) ─────────────────────────────────────────
graph.set_entry_point("classifier")  # Use set_entry_point instead of add_edge(None, ...)

# ── Add edges to summary and END ───────────────────────────────────────────
graph.add_edge("structured",   "summary")
graph.add_edge("unstructured", "summary")
graph.add_edge("refuse",       END)
graph.add_edge("summary",      END)
print("Graph nodes:", list(graph.nodes.keys()))
analyst_graph = graph.compile(checkpointer=cp)

def langgraph_to_dot(graph):
    lines = ["digraph G {"]
    for node in graph.nodes:
        lines.append(f'    "{node}";')
    for src, dst in graph.edges:
        lines.append(f'    "{src}" -> "{dst}";')
    lines.append("}")
    return "\n".join(lines)

# After building the graph and before compile:
dot_str = langgraph_to_dot(graph)
import streamlit as st
st.graphviz_chart(dot_str)

# --- Streamlit UI ---
st.title("🧠 Agentic Data Analyst with Advanced Debugging")




# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Chat", "🔍 Tool Calls", "🔄 Iterations", "📝 Debug Logs", "🧪 Tests"])
def display_plots_in_chat():
    """Display any plots created during the current analysis"""
    if "current_plots" in st.session_state and st.session_state.current_plots:
        st.subheader("📊 Generated Plots")
        
        for i, plot_info in enumerate(st.session_state.current_plots):
            st.write(f"**{plot_info['title']}**")
            if plot_info['subset_info']:
                st.caption(f"Filtered by: {plot_info['subset_info']}")
            
            # Display the matplotlib figure
            st.pyplot(plot_info['figure'])
            
        # Clear plots after displaying
        st.session_state.current_plots = []
with tab1:
    st.header("Chat Interface")
    
    with st.expander("🔧 System Status"):
        st.write(f"✅ Resources Loaded: {'vectorstore' in st.session_state}")
        records = st.session_state.get("records", [])
        st.write(f"📊 Total records: {len(records)}")
        if records:
            st.write("📝 Example record:")
            st.json(records[0])
    session_id = st.text_input("Session ID", value="default-user-001")
    if "thread_id" not in st.session_state or st.session_state.get("session_id") != session_id:
        st.session_state["session_id"] = session_id
        st.session_state["thread_id"]  = session_id          # one thread per user
        st.session_state["chat_history"] = []
    query = st.text_input("Ask your question about the support dataset:")
    
    if st.button("🔍 Ask") and query:
        with st.spinner("Analyzing data..."):
            try:
                debug_logger.clear()
                
                # Clear any previous plots
                if "current_plots" in st.session_state:
                    st.session_state.current_plots = []
                if "current_plotly_figs" in st.session_state:
                    st.session_state.current_plotly_figs = []
                
                debug_logger.log("INFO", f"Starting analysis for query: {query}")

                start = time.time()
                thread_id = st.session_state["thread_id"]
                MemoryStore.set(thread_id, "test_key", "hello_world")

                history   = "\n".join(
                    f"User: {u}\nAssistant: {a}" for u, a in st.session_state["chat_history"]
                )

                out = analyst_graph.invoke(
                    {
                        "thread_id": thread_id,
                        "user_input": query,
                        "history": history,
                        "last_user": query,
                    },
                    config=RunnableConfig(thread_id=thread_id),
                )
                answer = out.get("assistant_reply", "")
                duration = round(time.time() - start, 2)
                debug_logger.log("SUCCESS", f"Analysis completed in {duration}s")

                # Display the LLM's textual answer
                st.markdown(f"**Answer:** {answer}")
                st.caption(f"⏱ Duration: {duration}s")

                # Display any plots that were created
                display_plots_in_chat()
                
                # Or if using Plotly:
                # display_plotly_charts()

                # Update history
                if "history" not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.append({
                    "query": query,
                    "answer": answer,
                    "duration": duration,
                    "tool_calls": len(debug_logger.tool_calls),
                    "iterations": len(debug_logger.iterations)
                })
                st.session_state["chat_history"].append((query, answer))


            except Exception as e:
                debug_logger.log("ERROR", f"Analysis failed: {str(e)}")
                st.error(f"Error: {str(e)}")
                st.info("Check the Debug Logs tab for detailed error information.")

    
    st.sidebar.subheader("📌 Memory")
    thread_id = st.session_state.get("thread_id", "default-user-001")
    st.sidebar.json(MemoryStore.all(thread_id))

    # Display conversation history (rest of your code remains the same)
    if st.session_state.get("history"):
        st.subheader("📊 Analysis History")
        for i, turn in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Query {len(st.session_state.history) - i}: {turn['query'][:50]}..."):
                st.markdown(f"**Q:** {turn['query']}")
                st.markdown(f"**A:** {turn['answer']}")
                

with tab2:
    st.header("🔍 Tool Call Analysis")
    
    if debug_logger.tool_calls:
        # Tool call summary
        tool_summary = Counter([call["tool_name"] for call in debug_logger.tool_calls])
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Tool Usage Summary")
            for tool, count in tool_summary.most_common():
                st.metric(tool, count)
        
        with col2:
            st.subheader("Performance Metrics")
            durations = [call["duration"] for call in debug_logger.tool_calls if call["error"] is None]
            if durations:
                st.metric("Avg Duration", f"{np.mean(durations):.3f}s")
                st.metric("Max Duration", f"{max(durations):.3f}s")
                st.metric("Min Duration", f"{min(durations):.3f}s")
        
        # Detailed tool calls
        st.subheader("Detailed Tool Calls")
        for i, call in enumerate(debug_logger.tool_calls):
            with st.expander(f"Call {i+1}: {call['tool_name']} ({'❌' if call['error'] else '✅'})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Arguments:**")
                    st.json(call["arguments"])
                with col2:
                    st.write("**Metadata:**")
                    st.write(f"Duration: {call['duration']:.3f}s")
                    st.write(f"Iteration: {call['iteration']}")
                    st.write(f"Timestamp: {call['timestamp']}")
                
                if call["error"]:
                    st.error(f"Error: {call['error']}")
                else:
                    st.write("**Result:**")
                    if isinstance(call["result"], (dict, list)):
                        st.json(call["result"])
                    else:
                        st.write(str(call["result"]))
    else:
        st.info("No tool calls recorded yet. Run a query to see tool call details.")

with tab3:
    st.header("🔄 Agent Iterations")
    
    if debug_logger.iterations:
        for i, iteration in enumerate(debug_logger.iterations):
            with st.expander(f"Iteration {iteration['iteration']}: {iteration.get('duration', 0):.3f}s"):
                st.write(f"**Thought:** {iteration.get('thought', 'N/A')}")
                st.write(f"**Duration:** {iteration.get('duration', 0):.3f}s")
                st.write(f"**Tool Calls in Iteration:** {len(iteration.get('tool_calls', []))}")
                
                if iteration.get('tool_calls'):
                    st.write("**Tools Used:**")
                    for call in iteration['tool_calls']:
                        status = "❌" if call['error'] else "✅"
                        st.write(f"- {status} {call['tool_name']} ({call['duration']:.3f}s)")
    else:
        st.info("No iterations recorded yet. Run a query to see iteration details.")

with tab4:
    st.header("📝 Debug Logs")
    
    # Log filtering
    col1, col2 = st.columns(2)
    with col1:
        level_filter = st.multiselect(
            "Filter by Level",
            ["INFO", "DEBUG", "SUCCESS", "ERROR", "WARNING"],
            default=["INFO", "SUCCESS", "ERROR"]
        )
    with col2:
        max_logs = st.slider("Max Logs to Show", 10, 100, 50)
    
    if debug_logger.logs:
        filtered_logs = [
            log for log in debug_logger.logs 
            if log["level"] in level_filter
        ][-max_logs:]
        
        st.write(f"Showing {len(filtered_logs)} of {len(debug_logger.logs)} logs")
        
        for log in reversed(filtered_logs):
            level_color = {
                "INFO": "🔵",
                "DEBUG": "⚪",
                "SUCCESS": "🟢", 
                "ERROR": "🔴",
                "WARNING": "🟡"
            }.get(log["level"], "⚪")
            
            with st.expander(f"{level_color} [{log['level']}] {log['message']} (Iter {log['iteration']})"):
                st.write(f"**Timestamp:** {log['timestamp']}")
                st.write(f"**Iteration:** {log['iteration']}")
                if log.get("data"):
                    st.write("**Data:**")
                    st.json(log["data"])
    else:
        st.info("No debug logs yet. Run a query to see detailed logs.")

with tab5:
    st.header("🧪 Tool Tests")
    
    st.write("Test individual tools to verify they're working correctly:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Test: Fetch 'refund' examples"):
            with st.spinner("Testing..."):
                result = fetch_relevant_examples("refund", 3)
                st.json(result)
    
    with col2:
        if st.button("Test: Intent metrics"):
            with st.spinner("Testing..."):
                result = compute_dataset_metrics("intent")
                st.json(result)
    
    with col3:
        if st.button("Test: Word analysis"):
            with st.spinner("Testing..."):
                result = count_word_occurrences("response")
                st.json(result)
    
    # Advanced tests
    st.subheader("Advanced Statistical Tests")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Test: Chi-square (intent vs category)"):
            with st.spinner("Testing..."):
                try:
                    result = analyze_chi_square("intent", "category")
                    st.json(result)
                except Exception as e:
                    st.error(f"Test failed: {str(e)}")
    
    with col2:
        if st.button("Test: Response length by intent"):
            with st.spinner("Testing..."):
                try:
                    result = analyze_col_length_by_column("intent", "response_length")
                    st.json(result)
                except Exception as e:
                    st.error(f"Test failed: {str(e)}")

# Footer with debug summary
if debug_logger.logs:
    st.write("---")