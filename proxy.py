from fastapi import FastAPI, Request, Response
from llmlingua import PromptCompressor
import httpx
import torch

app = FastAPI()

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    use_llmlingua2=True,
    device_map=device
)

MIN_CHARS = 1000
TEXT_COMPRESSION_RATE = 0.5
CODE_COMPRESSION_RATE = 0.75

def looks_like_code(text: str) -> bool:
    indicators = [
        "def ", "class ", "import ", "from ", "return ",
        "function ", "const ", "let ", "var ", "async ", "await ",
        "if (", "for (", "while (", "switch (",
        "```", "self.", "this.", "->", "=>",
        "public ", "private ", "protected ",
        "struct ", "impl ", "fn ", "pub ",
    ]
    code_chars = text.count("{") + text.count("}") + text.count(";") + text.count("()")
    line_count = text.count("\n")
    indented_lines = sum(1 for line in text.split("\n") if line.startswith("    ") or line.startswith("\t"))
    
    has_indicators = any(ind in text for ind in indicators)
    has_code_structure = code_chars > 10
    has_indentation = line_count > 3 and indented_lines / max(line_count, 1) > 0.3
    
    return has_indicators or has_code_structure or has_indentation

def compress_text(text: str) -> tuple[str, int, int]:
    if len(text) < MIN_CHARS:
        return text, 0, 0
    
    is_code = looks_like_code(text)
    rate = CODE_COMPRESSION_RATE if is_code else TEXT_COMPRESSION_RATE
    
    result = compressor.compress_prompt(text, rate=rate)
    
    content_type = "code" if is_code else "text"
    print(f"  [{content_type}] {result['origin_tokens']:,} → {result['compressed_tokens']:,} (rate={rate})")
    
    return result["compressed_prompt"], result["origin_tokens"], result["compressed_tokens"]

def process_content(content, role: str) -> tuple[any, int, int]:
    total_before = 0
    total_after = 0
    
    if isinstance(content, str):
        compressed, before, after = compress_text(content)
        return compressed, before, after
    
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            
            block_type = block.get("type")
            
            if block_type == "text" and "text" in block:
                compressed, before, after = compress_text(block["text"])
                if before > 0:
                    block["text"] = compressed
                    total_before += before
                    total_after += after
            
            elif block_type == "tool_result" and "content" in block:
                tool_content = block["content"]
                if isinstance(tool_content, str):
                    compressed, before, after = compress_text(tool_content)
                    if before > 0:
                        block["content"] = compressed
                        total_before += before
                        total_after += after
                elif isinstance(tool_content, list):
                    for tc in tool_content:
                        if isinstance(tc, dict) and tc.get("type") == "text":
                            compressed, before, after = compress_text(tc.get("text", ""))
                            if before > 0:
                                tc["text"] = compressed
                                total_before += before
                                total_after += after
    
    return content, total_before, total_after

@app.post("/v1/messages")
async def proxy(request: Request):
    body = await request.json()
    headers = dict(request.headers)
    
    for h in ["host", "content-length", "transfer-encoding"]:
        headers.pop(h, None)
    
    total_before = 0
    total_after = 0
    
    for msg in body.get("messages", []):
        content, before, after = process_content(msg.get("content"), msg.get("role", ""))
        msg["content"] = content
        total_before += before
        total_after += after
    
    if total_before > 0:
        ratio = (1 - total_after / total_before) * 100
        print(f"[TOTAL] {total_before:,} → {total_after:,} tokens ({ratio:.1f}% reduced)")
        print("-" * 50)
    
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            json=body,
            headers=headers
        )
    
    return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def passthrough(request: Request, path: str):
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ["host", "content-length"]}
    
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.request(
            request.method,
            f"https://api.anthropic.com/{path}",
            headers=headers,
            content=await request.body()
        )
    return Response(content=resp.content, status_code=resp.status_code)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
