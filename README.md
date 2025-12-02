# Claude Code Compression Proxy

Proxy for Claude Code (VS Code) that compresses context using LLMLingua-2 (~50% reduction).

## Requirements

- Python 3.13+
- macOS (Apple Silicon) / Linux

## Installation

```bash
git clone <repo>
cd claude-code-proxy

uv sync
```

## Usage

```bash
uv run proxy.py
```

## VS Code Configuration

Option 1 — launch with env variable:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 code .
```

Option 2 — add to `~/.zshrc` or `~/.bashrc`:

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8080"
```

Restart terminal and VS Code.

## Configuration

In `proxy.py`:

```python
MIN_CHARS = 1000        # minimum chars to trigger compression
COMPRESSION_RATE = 0.5  # compression rate (0.5 = 50%)
device_map = "mps"      # mps (Apple Silicon) / cpu / cuda
```

## Output

```
[COMPRESSED] 12,450 → 6,225 tokens (50.0% reduced)
```

## License

MIT
