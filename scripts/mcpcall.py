#!/usr/bin/env python3
"""Tiny stdio MCP client for the `blender-mcp` server.

Usage:
  uv run --with mcp scripts/mcpcall.py --list
  uv run --with mcp scripts/mcpcall.py get_scene_info
  uv run --with mcp scripts/mcpcall.py execute_blender_code '{"code": "import bpy; print(len(bpy.data.objects))"}'
  uv run --with mcp scripts/mcpcall.py get_viewport_screenshot '{"max_size": 1200}' --save renders/viewport.png

This spawns `uvx blender-mcp` over stdio (same as Claude Code does) so it tests
the real MCP path, not just the addon socket.
"""
import asyncio, base64, json, sys, os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    args = sys.argv[1:]
    save = None
    if "--save" in args:
        i = args.index("--save"); save = args[i+1]; del args[i:i+2]
    params = StdioServerParameters(command="uvx", args=["blender-mcp"], env=dict(os.environ))
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            if not args or args[0] in ("--list", "--schema"):
                tools = await s.list_tools()
                for t in tools.tools:
                    print(f"- {t.name}: {(t.description or '').strip().splitlines()[0][:110]}")
                    if args and args[0] == "--schema":
                        req = (getattr(t,"input_schema",None) or getattr(t,"inputSchema",{})).get("required", []); props = (getattr(t,"input_schema",None) or getattr(t,"inputSchema",{})).get("properties", {})
                        print("    args:", ", ".join(f"{k}{'*' if k in req else ''}" for k in props))
                return
            name = args[0]
            tool_args = json.loads(args[1]) if len(args) > 1 else {}
            tool_args.setdefault("user_prompt", "cli call")  # required by blender-mcp >= protocol 5
            res = await s.call_tool(name, tool_args)
            for c in res.content:
                if c.type == "text":
                    print(c.text)
                elif c.type == "image":
                    data = base64.b64decode(c.data)
                    if save:
                        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
                        open(save, "wb").write(data); print(f"[image saved -> {save} ({len(data)} bytes)]")
                    else:
                        print(f"[image {c.mimeType} {len(data)} bytes; use --save PATH]")
            if getattr(res,'is_error',False) or getattr(res,'isError',False):
                sys.exit(1)

asyncio.run(main())
