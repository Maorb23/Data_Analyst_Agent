import sqlite3
conn = sqlite3.connect("agent_memory.db")
print(conn.execute("SELECT * FROM memory").fetchall())
