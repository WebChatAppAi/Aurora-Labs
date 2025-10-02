#!/usr/bin/env python3
"""
Professional startup script for the AI Music Generation API.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="AI Music Generation API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--config", default="config/models.yaml", help="Configuration file")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")

    args = parser.parse_args()

    # Validate config file exists
    if not Path(args.config).exists():
        print(f"❌ Configuration file not found: {args.config}")
        print("Please create the configuration file or specify a different path.")
        return 1

    print("🎵 AI Music Generation API Server")
    print("=" * 50)
    print(f"🌐 Server URL: http://{args.host}:{args.port}")
    print(f"📚 API Docs: http://{args.host}:{args.port}/docs")
    print(f"⚙️ Configuration: {args.config}")
    print(f"🔄 Auto-reload: {'Enabled' if args.reload else 'Disabled'}")
    print("=" * 50)

    # Build uvicorn command
    cmd = [
        sys.executable, "-m", "uvicorn",
        "main:app",
        "--host", args.host,
        "--port", str(args.port)
    ]

    if args.reload:
        cmd.append("--reload")
    else:
        cmd.extend(["--workers", str(args.workers)])

    # Set environment variable for config path
    import os
    os.environ["CONFIG_PATH"] = args.config

    try:
        print("🚀 Starting server...")
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Server failed to start: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())