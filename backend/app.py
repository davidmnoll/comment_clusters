from doctest import debug
from fastapi import FastAPI, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, AsyncGenerator
import os
from comment_tree import CommentTree
import asyncio

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
async def get_root_comment() -> Dict:
    """Get the root comment with its children"""
    tree = CommentTree()
    if not tree:
       raise HTTPException(status_code=500, detail="Failed to load root comment")
    if not tree.table['']:
        raise HTTPException(status_code=500, detail="Root comment not found")

    return tree.table[''].to_dict()

@app.get("/{path}")
async def get_comment_info(
    path: str,
) -> Dict:
    """Get metadata about a specific comment"""
    tree = CommentTree()
    if not tree:
       raise HTTPException(status_code=500, detail="Failed to load root comment")
    if not tree.table['']:
        raise HTTPException(status_code=500, detail="Root comment not found")

    try:
        comment = await tree.get_comment_by_path(path)
        return comment.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/info")
async def health_check():
    """Health check endpoint with usage information"""
    tree = CommentTree()
    if not tree:
       raise HTTPException(status_code=500, detail="Failed to load root comment")
    if not tree.table['']:
        raise HTTPException(status_code=500, detail="Root comment not found")


    cluster_status = {
        "kmeans": tree.table[''] and hasattr(tree.table[''], 'kmeans_clusters') and bool(tree.table[''].kmeans_clusters),
        "louvain": tree.table[''] and hasattr(tree.table[''], 'louvain_clusters') and bool(tree.table[''].louvain_clusters),
    }
    
    return {
        "message": "Comment Clustering API",
        "clustering_status": {
            "version": tree.cluster_version,
            "algorithms": cluster_status
        },
        "usage": {
            "get_root": "/",
            "get_comment": "/{path}",
            "path_format": "Use hyphen-separated indices, e.g. '0-1-2'"
        }
    }

# Mount static files for frontend
static_files_dir = os.path.join(os.path.dirname(__file__), "../frontend/dist")
if os.path.exists(static_files_dir):
    app.mount("/", StaticFiles(directory=static_files_dir, html=True), name="frontend")

if __name__ == "__main__":
    # import sys
    # if len(sys.argv) > 1 and sys.argv[1] == "generate":
    async def generate_comment_tree():
        await CommentTree().generate()
    
    asyncio.run(generate_comment_tree())
    # else:
    #     uvicorn.run(app, host="0.0.0.0", port=8000)
