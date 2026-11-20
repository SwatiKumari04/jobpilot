#!/usr/bin/env python3
"""Launcher. Locally: http://127.0.0.1:8000. On Render/Docker, $PORT is set and we bind 0.0.0.0."""

import os

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"
    reload = os.environ.get("RELOAD", "").lower() in {"1", "true", "yes"}
    print(f"JobPilot starting on http://{host}:{port}  (Ctrl+C to stop)")
    uvicorn.run("jobpilot.api.main:app", host=host, port=port, reload=reload,
                proxy_headers=True, forwarded_allow_ips="*")
