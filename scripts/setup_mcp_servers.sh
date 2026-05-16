#!/usr/bin/env bash
# Setup helper for community MCP servers used by Health-Guide-Agent.
#
# jlfwong/food-data-central-mcp-server (USDA, Nutritionist) is NOT published
# to npm — we have to clone and `npm install` once. wger-mcp and medical-mcp
# are both on npm, so `npx -y <pkg>` picks them up on first launch without
# any setup here.
#
# Usage:
#   bash scripts/setup_mcp_servers.sh
#
# Override the clone target via MCP_USDA_DIR.
set -euo pipefail

if ! command -v node >/dev/null 2>&1; then
  echo "[setup] error: Node.js not found on PATH. Install Node.js ≥ 18 first." >&2
  exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
  echo "[setup] error: npm not found on PATH." >&2
  exit 1
fi

TARGET="${MCP_USDA_DIR:-$HOME/.cache/mcp-servers/usda-fdc}"
mkdir -p "$(dirname "$TARGET")"

if [ ! -d "$TARGET/.git" ]; then
  echo "[setup] cloning jlfwong/food-data-central-mcp-server -> $TARGET"
  git clone --depth=1 https://github.com/jlfwong/food-data-central-mcp-server.git "$TARGET"
else
  echo "[setup] reusing existing clone at $TARGET"
  git -C "$TARGET" pull --ff-only || true
fi

# Upstream pins @modelcontextprotocol/sdk to ^0.1.0 which has been unpublished
# from npm. The src/ uses 1.x APIs (McpServer, StdioServerTransport) so just
# bumping the pin makes it build with no source changes.
echo "[setup] patching @modelcontextprotocol/sdk pin to ^1.10.0"
(cd "$TARGET" && npm pkg set 'dependencies.@modelcontextprotocol/sdk=^1.10.0')

echo "[setup] running npm install (this may take a minute)"
(cd "$TARGET" && npm install --no-audit --no-fund)

SCRIPT_PATH="$TARGET/src/index.ts"
if [ ! -f "$SCRIPT_PATH" ]; then
  echo "[setup] error: expected $SCRIPT_PATH but it does not exist." >&2
  exit 1
fi

# medical-mcp (Critic) — installed locally because its npm `bin` link is a
# raw ES module without a shebang, so `npx -y medical-mcp` is broken. We
# point Python at the build artifact and invoke `node` on it directly.
MEDICAL_DIR="${MCP_MEDICAL_DIR:-$HOME/.cache/mcp-servers/medical-mcp}"
mkdir -p "$MEDICAL_DIR"
if [ ! -f "$MEDICAL_DIR/package.json" ]; then
  echo "[setup] initializing $MEDICAL_DIR"
  (cd "$MEDICAL_DIR" && npm init -y >/dev/null)
fi
echo "[setup] installing medical-mcp into $MEDICAL_DIR (may take a minute)"
(cd "$MEDICAL_DIR" && npm install medical-mcp --no-audit --no-fund)

MEDICAL_SCRIPT="$MEDICAL_DIR/node_modules/medical-mcp/build/index.js"
if [ ! -f "$MEDICAL_SCRIPT" ]; then
  echo "[setup] error: expected $MEDICAL_SCRIPT but it does not exist." >&2
  exit 1
fi
# medical-mcp 1.0.8 uses console.log for status/banner/progress lines, which
# goes to stdout and corrupts the MCP JSON-RPC stream. Rewrite all such
# log sites to console.error (stderr) so the JSON-RPC channel stays clean.
echo "[setup] redirecting medical-mcp stdout logs to stderr (~7 console.log sites)"
sed -i 's|console\.log(|console.error(|g' \
  "$MEDICAL_DIR/node_modules/medical-mcp/build/index.js" \
  "$MEDICAL_DIR/node_modules/medical-mcp/build/utils.js"

# wger-mcp (Trainer) — also installed locally because the package's Zod
# schemas are out of sync with the live wger.de API (the response's
# `variations` field is now sometimes undefined, but wger-mcp requires it).
# We patch the compiled schema to make it optional. The schema file lives
# inside the package and gets reset on each npm install, so the patch
# happens unconditionally every time this script runs.
WGER_DIR="${MCP_WGER_DIR:-$HOME/.cache/mcp-servers/wger-mcp}"
mkdir -p "$WGER_DIR"
if [ ! -f "$WGER_DIR/package.json" ]; then
  echo "[setup] initializing $WGER_DIR"
  (cd "$WGER_DIR" && npm init -y >/dev/null)
fi
echo "[setup] installing @juxsta/wger-mcp into $WGER_DIR"
(cd "$WGER_DIR" && npm install @juxsta/wger-mcp --no-audit --no-fund)

WGER_SCHEMA="$WGER_DIR/node_modules/@juxsta/wger-mcp/dist/schemas/api.js"
WGER_SCRIPT="$WGER_DIR/node_modules/@juxsta/wger-mcp/dist/index.js"
if [ ! -f "$WGER_SCHEMA" ] || [ ! -f "$WGER_SCRIPT" ]; then
  echo "[setup] error: expected wger files missing under $WGER_DIR/node_modules/@juxsta/wger-mcp/dist/" >&2
  exit 1
fi
echo "[setup] patching wger schema (variations: nullable → nullable+optional)"
sed -i 's|variations: zod_1\.z\.number()\.int()\.positive()\.nullable(),|variations: zod_1.z.number().int().positive().nullable().optional(),|' "$WGER_SCHEMA"

echo
echo "[setup] done. Add the following to your .env (or shell):"
echo
echo "    # Nutritionist (USDA)"
echo "    MCP_USDA_SCRIPT_PATH=$SCRIPT_PATH"
echo "    USDA_API_KEY=<free key from https://fdc.nal.usda.gov/api-key-signup>"
echo "    MCP_NUTRITIONIST_ENABLED=true"
echo
echo "    # Critic (medical-mcp)"
echo "    MCP_MEDICAL_SCRIPT_PATH=$MEDICAL_SCRIPT"
echo "    MCP_CRITIC_ENABLED=true"
echo
echo "    # Trainer (wger) — needs a free account at https://wger.de/en/user/registration"
echo "    # then generate an API key on your wger profile page."
echo "    MCP_WGER_SCRIPT_PATH=$WGER_SCRIPT"
echo "    WGER_API_KEY=<your wger key>"
echo "    MCP_TRAINER_ENABLED=true"
echo
