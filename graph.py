# Re-import required libraries after code execution state reset
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Define categorized nodes with their groups
nodes_by_type = {
    "Data": ["HF Dataset", "ETL & Preprocessing", "Embedding "],
    "Storage": ["FAISS Store"],
    "LLM": ["Qwen LLM", "Tokenizer"],
    "Planner": ["Tool Planner"],
    "Tools": ["Statistical Endpoints", "Example Fetcher", "Summarizers"],
    "UI": ["Streamlit UI", "User Query"],
    "Controller": ["Agent Controller"],
    "Output": ["Answer to User"]
}

# Assign colors and shapes
node_styles = {
    "Data": {"color": "skyblue", "shape": "o"},
    "Storage": {"color": "lightgray", "shape": "8"},
    "LLM": {"color": "mediumpurple", "shape": "D"},
    "Planner": {"color": "darkorange", "shape": "h"},
    "Tools": {"color": "gold", "shape": "s"},
    "UI": {"color": "mediumseagreen", "shape": "^"},
    "Controller": {"color": "tomato", "shape": "p"},
    "Output": {"color": "lightgreen", "shape": "v"}
}

# Flatten nodes into list and tag types
nodes = []
node_colors = {}
node_shapes = {}
node_positions = {}

for group, group_nodes in nodes_by_type.items():
    for node in group_nodes:
        nodes.append(node)
        node_colors[node] = node_styles[group]["color"]
        node_shapes[node] = node_styles[group]["shape"]

# Create the directed graph
G = nx.DiGraph()
G.add_nodes_from(nodes)

# Define logical edges
edges = [
    # Data branch
    ("HF Dataset", "ETL & Preprocessing"),
    ("ETL & Preprocessing", "Embedding "),
    ("Embedding ", "FAISS Store"),

    # UI + Agent Controller + LLM
    ("Streamlit UI", "User Query"),
    ("User Query", "Tokenizer"),
    ("Tokenizer", "Agent Controller"),
    ("Agent Controller", "Qwen LLM"),
    #("Agent Controller", "Tokenizer"),
    #("Tokenizer", "Qwen LLM"),
    ("Qwen LLM", "Answer to User"),

    # Tool use path
    ("Agent Controller", "Tool Planner"),
    ("Tool Planner", "Statistical Endpoints"),
    ("Tool Planner", "Example Fetcher"),
    ("Tool Planner", "Summarizers"),

    # FAISS interaction
    ("Agent Controller", "FAISS Store"),
    ("FAISS Store", "Qwen LLM"),
]

G.add_edges_from(edges)

# Custom positions for improved layout (horizontal style)
pos = {
    # Top-left: Data flow
    "HF Dataset": (-4, 2),
    "ETL & Preprocessing": (-3, 2),
    "Embedding ": (-2, 2),

    # FAISS in the middle layer
    "FAISS Store": (-0.5, 1),

    # Top-middle: User entry
    "Streamlit UI": (1, 3),
    "User Query": (1, 2.5),
    "Tokenizer": (1, 2),
    "Agent Controller": (1, 1.5),

    # LLM between FAISS and Answer
    "Qwen LLM": (1, 1),

    # Bottom-center: Final output
    "Answer to User": (1, 0),

    # Top-right: Tool logic
    "Tool Planner": (3, 2),
    "Statistical Endpoints": (4, 1.5),
    "Example Fetcher": (3, 1.5),
    "Summarizers": (5, 1.5)
}

fig, ax = plt.subplots(figsize=(14, 8))

# Draw nodes by shape group
for shape in set(node_shapes.values()):
    shape_nodes = [n for n in G.nodes if node_shapes[n] == shape]
    nx.draw_networkx_nodes(G, pos, nodelist=shape_nodes,
                           node_shape=shape,
                           node_color=[node_colors[n] for n in shape_nodes],
                           ax=ax)

# Draw edges
nx.draw_networkx_edges(G, pos, ax=ax, arrowstyle='->', arrowsize=12, connectionstyle='arc3,rad=0.15')

# Add labels
for node, (x, y) in pos.items():
    if node == "FAISS Store":
        ax.text(x, y - 0.15, node, fontsize=9, ha='center', va='bottom', color='black')
    elif node == "Qwen LLM":
        ax.text(x + 0.5, y - 0.2, node, fontsize=9, ha='center', va='bottom', color='black')
    elif node == "Tool Planner":
        ax.text(x, y + 0.15, node, fontsize=9, ha='center', va='bottom', color='black')
    elif node == "Agent Controller":
        ax.text(x + 0.45, y - 0.2, node, fontsize=9, ha='center', va='bottom', color='black')
    elif node in ["Statistical Endpoints",  "Summarizers"]:
        ax.text(x, y - 0.15, node, fontsize=9, ha='center', va='bottom', color='black')
    elif node == "Example Fetcher":
        ax.text(x - 0.2, y - 0.15, node, fontsize=9, ha='center', va='bottom', color='black')
    else:
        ax.text(x - 0.4, y - 0.15, node, fontsize=9, ha='center', va='bottom')

# --- LEGEND ---
legend_patches = [
    mpatches.Patch(color="skyblue", label="Data (●)"),
    mpatches.Patch(color="lightgray", label="Storage (◉)"),
    mpatches.Patch(color="mediumpurple", label="LLM (◆)"),
    mpatches.Patch(color="darkorange", label="Planner (⬢)"),
    mpatches.Patch(color="gold", label="Tools (■)"),
    mpatches.Patch(color="mediumseagreen", label="UI (▲)"),
    mpatches.Patch(color="tomato", label="Agent Controller (⬟)"),
    mpatches.Patch(color="lightgreen", label="Output (▼)")
]
ax.legend(
    handles=legend_patches,
    loc='upper left',
    bbox_to_anchor=(1.01, 1),
    borderaxespad=0,
    frameon=False,
    title="Node Roles and Shapes",
    title_fontsize=11,
    fontsize=10
)

# Final styling
ax.set_title("Data Analyst Agent System – Modular Branch View", fontsize=14)
ax.axis('off')
plt.tight_layout()
plt.savefig("data_analyst_agent_system_graph_with_legend.png", dpi=300, bbox_inches='tight')
plt.show()