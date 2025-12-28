import os
import json
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional

from lazzaro.core.memory_system import MemorySystem

app = FastAPI(title="Lazzaro Memory Dashboard")

# Global reference to the memory system
# In a real app, this would be injected or shared
_ms: Optional[MemorySystem] = None

# Set up templates
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(CURRENT_DIR, "templates"))

def set_memory_system(ms: MemorySystem):
    global _ms
    _ms = ms

@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/stats")
async def get_stats():
    if not _ms:
        return {"error": "Memory system not initialized"}
    _ms.check_for_updates()
    stats = _ms.get_stats()
    stats["user_id"] = _ms.user_id # Add current user_id to stats
    return stats

@app.get("/api/users")
async def get_users():
    if not _ms:
        return []
    return _ms.get_all_users()

@app.post("/api/users/switch")
async def switch_user(request: Request):
    if not _ms:
        return {"error": "Memory system not initialized"}
    data = await request.json()
    new_user_id = data.get("user_id")
    if not new_user_id:
        return {"error": "User ID required"}
    
    _ms.switch_user(new_user_id)
    return {"status": "success", "user_id": _ms.user_id}

@app.get("/api/insights")
async def get_insights():
    if not _ms:
        return {"error": "Memory system not initialized"}
    insights = _ms.get_insights()
    return {"insights": insights}

@app.get("/api/export")
async def export_observations(format: str = "markdown"):
    if not _ms:
        return {"error": "Memory system not initialized"}
    content = _ms.export_observations(format=format)
    return {"content": content}

@app.get("/api/graph")
async def get_graph():
    if not _ms:
        return {"nodes": [], "links": []}
    
    _ms.check_for_updates()
    nodes = []
    links = []
    
    # Extract nodes and edges from all shards
    for shard_key, shard in _ms.shards.items():
        for node_id, node in shard.nodes.items():
            nodes.append({
                "id": node_id,
                "content": node.content,
                "type": node.type,
                "salience": node.salience,
                "shard": shard_key,
                "access_count": node.access_count,
                "is_super_node": node.is_super_node
            })
        
        for (src, tgt), edge in shard.edges.items():
            links.append({
                "source": src,
                "target": tgt,
                "weight": edge.weight,
                "type": edge.edge_type
            })
            
    # Add super nodes
    for node_id, node in _ms.super_nodes.items():
         nodes.append({
                "id": node_id,
                "content": node.content,
                "type": "super_node",
                "salience": node.salience,
                "shard": "global",
                "is_super_node": True
            })
    
    return {"nodes": nodes, "links": links}

@app.get("/api/profile")
async def get_profile():
    if not _ms:
        return {}
    _ms.check_for_updates()
    return _ms.profile.to_dict()

@app.post("/api/consolidate")
async def consolidate():
    if not _ms:
        return {"error": "Memory system not initialized"}
    
    # Forcing consolidation if there's anything in the queue or buffer
    status = _ms.run_consolidation()
    return {"status": status}

def entry_point():
    """CLI entry point for lazzaro-dashboard"""
    print("🚀 Starting Lazzaro Premium Dashboard on http://localhost:5299")
    
    # Initialize a default memory system if one doesn't exist
    # In practice, the user might want to load their own
    ms = MemorySystem(load_from_disk=True)
    set_memory_system(ms)
    
    uvicorn.run(app, host="0.0.0.0", port=5299)

if __name__ == "__main__":
    entry_point()
