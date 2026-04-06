#!/usr/bin/env python
"""Colab MCP spike – prove the real transport works end-to-end.

This script spawns the ``colab-mcp`` server via ``uvx``, performs the MCP
handshake (``initialize`` + ``list_tools``), and reports findings.  No
browser-connected Colab notebook is required — the goal is to validate
that the MCP stdio transport layer is functional.

Usage:
    python scripts/colab_mcp_spike.py

Exit codes:
    0 – spike succeeded (server spawned, handshake completed, tools listed)
    1 – prerequisites missing or handshake failed
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import time


def _check_prerequisites() -> list[str]:
    """Return a list of missing prerequisite descriptions."""
    missing: list[str] = []
    if shutil.which("uvx") is None:
        missing.append("uvx not on PATH (pip install uv)")
    try:
        from mcp import ClientSession  # noqa: F401
    except ImportError:
        missing.append("mcp SDK not installed (pip install 'mcp>=1.0')")
    return missing


async def _run_spike() -> dict:
    """Spawn the colab-mcp server and perform the MCP handshake."""
    from contextlib import AsyncExitStack

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    uvx_path = shutil.which("uvx")
    server_params = StdioServerParameters(
        command=uvx_path,
        args=["git+https://github.com/googlecolab/colab-mcp"],
    )

    report: dict = {
        "server_spawned": False,
        "handshake_ok": False,
        "tools": [],
        "tool_count": 0,
        "server_info": None,
        "elapsed_secs": 0.0,
        "error": None,
    }

    t0 = time.monotonic()
    exit_stack = AsyncExitStack()
    try:
        transport = await exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        report["server_spawned"] = True
        read_stream, write_stream = transport

        session = await exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        init_result = await session.initialize()
        report["handshake_ok"] = True

        # Capture server info from the init result
        if hasattr(init_result, "server_info"):
            si = init_result.server_info
            report["server_info"] = {
                "name": getattr(si, "name", None),
                "version": getattr(si, "version", None),
            }
        elif hasattr(init_result, "serverInfo"):
            si = init_result.serverInfo
            report["server_info"] = {
                "name": getattr(si, "name", None),
                "version": getattr(si, "version", None),
            }

        tools_result = await session.list_tools()
        tools = []
        for t in tools_result.tools:
            tool_info = {"name": t.name}
            if hasattr(t, "description") and t.description:
                tool_info["description"] = t.description[:120]
            if hasattr(t, "inputSchema") and t.inputSchema:
                tool_info["input_schema_keys"] = list(
                    t.inputSchema.get("properties", {}).keys()
                )
            tools.append(tool_info)
        report["tools"] = tools
        report["tool_count"] = len(tools)

    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        report["elapsed_secs"] = round(time.monotonic() - t0, 2)
        try:
            await exit_stack.aclose()
        except Exception:
            pass

    return report


def main() -> int:
    print("=" * 60)
    print("  Colab MCP Spike – Real Transport Validation")
    print("=" * 60)
    print()

    # 1. Prerequisites
    missing = _check_prerequisites()
    if missing:
        print("FAIL: Missing prerequisites:")
        for m in missing:
            print(f"  - {m}")
        return 1

    print("[OK] Prerequisites: uvx + mcp SDK present")
    print()

    # 2. Real MCP handshake
    print("Spawning colab-mcp server via uvx (first run may download)…")
    report = asyncio.run(_run_spike())

    print()
    print("-" * 60)
    print("  SPIKE REPORT")
    print("-" * 60)
    print(json.dumps(report, indent=2))
    print("-" * 60)
    print()

    if report["error"]:
        print(f"FAIL: {report['error']}")
        return 1

    if not report["server_spawned"]:
        print("FAIL: Server did not spawn")
        return 1

    if not report["handshake_ok"]:
        print("FAIL: MCP handshake did not complete")
        return 1

    print(f"[OK] Server spawned in {report['elapsed_secs']}s")
    if report["server_info"]:
        si = report["server_info"]
        print(f"[OK] Server: {si.get('name', '?')} v{si.get('version', '?')}")
    print("[OK] MCP handshake completed")
    print(f"[OK] {report['tool_count']} tool(s) available:")
    for t in report["tools"]:
        desc = f" – {t.get('description', '')}" if t.get("description") else ""
        schema = f" (params: {t.get('input_schema_keys', [])})" if t.get("input_schema_keys") else ""
        print(f"     • {t['name']}{desc}{schema}")

    print()
    print("SPIKE PASSED – Colab MCP transport layer is functional.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
