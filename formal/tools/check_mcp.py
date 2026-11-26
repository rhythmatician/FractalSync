"""Exercise the optional Lean MCP server against an actual project theorem.

Run with the Python environment containing the pinned optional server:
    python formal/tools/check_mcp.py
This is an authoring-environment check, separate from the Lean kernel audit.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys

from lean_lsp_mcp.models import DiagnosticsResult, VerifyResult
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client


async def main() -> None:
    formal_root = Path(__file__).resolve().parents[1]
    server = Path(sys.executable).with_name("lean-lsp-mcp")
    if sys.platform == "win32":
        server = server.with_suffix(".exe")
    parameters = StdioServerParameters(
        command=str(server),
        cwd=formal_root,
        env={
            **os.environ,
            "LEAN_PROJECT_PATH": str(formal_root),
            "LEAN_LOG_LEVEL": "NONE",
        },
    )
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(reader, writer, read_timeout_seconds=120) as session:
            await session.initialize()
            catalog = await session.list_tools()
            required = {"lean_diagnostic_messages", "lean_verify"}
            if not required.issubset({tool.name for tool in catalog.tools}):
                raise RuntimeError("Required Lean diagnostics/verification tools are missing")

            diagnostics_result = await session.call_tool(
                "lean_diagnostic_messages",
                {"file_path": "FractalSync/DistanceToSet.lean"},
            )
            if not isinstance(diagnostics_result, types.CallToolResult):
                raise RuntimeError("Diagnostics did not return a completed tool result")
            if diagnostics_result.is_error:
                raise RuntimeError(diagnostics_result.model_dump_json())
            diagnostics = DiagnosticsResult.model_validate(diagnostics_result.structured_content)
            if diagnostics.partial or not diagnostics.success or diagnostics.items:
                raise RuntimeError(diagnostics.model_dump_json())

            verification_result = await session.call_tool(
                "lean_verify",
                {
                    "file_path": str(formal_root / "FractalSync/DistanceToSet.lean"),
                    "theorem_name": "FractalSync.DistanceGraph.exists_nearest_segment",
                },
            )
            if not isinstance(verification_result, types.CallToolResult):
                raise RuntimeError("Verification did not return a completed tool result")
            if verification_result.is_error:
                raise RuntimeError(verification_result.model_dump_json())
            verification = VerifyResult.model_validate(verification_result.structured_content)
            if verification.trust.value != "standard" or verification.non_standard_axioms:
                raise RuntimeError(verification.model_dump_json())
            print(f"Lean MCP: {len(catalog.tools)} tools discovered; diagnostics clean")
            print("Nearest-segment theorem verified with standard axioms only")


if __name__ == "__main__":
    asyncio.run(main())
