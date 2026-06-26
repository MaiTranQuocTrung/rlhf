import ast, re, builtins, threading

def reward_model(generated_text: str) -> float:

    source_code, has_format = None, False
    for pattern in (r"```python[ \t]*\n(.*?)```", r"```[ \t]*\n(.*?)```"):
        blocks = re.findall(pattern, generated_text, re.DOTALL)
        if blocks:
            source_code = max(blocks, key=len).strip() or None
            has_format = True
            break
    if source_code is None:
        source_code = generated_text.strip() or None
    if source_code is None:
        return -1.0

    reward = 0.10 if has_format else 0.0


    try:
        ast.parse(source_code)
    except SyntaxError as e:
        lines = source_code.splitlines()
        if lines and e.lineno:
            reward += 0.10 * ((e.lineno - 1) / max(len(lines), 1))
        return max(-1.0, reward - 0.40)

    reward += 0.60

    substantive = [l for l in source_code.splitlines() if l.strip() and not l.strip().startswith("#")]
    if len(substantive) >= 3:
        reward += 0.10

    result = {"status": "pending"}
    def _run():
        try:
            exec(compile(source_code, "<generated>", "exec"), {"__builtins__": builtins})
            result["status"] = "ok"
        except Exception:
            result["status"] = "error"
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=3.0)
    reward += -0.10 if t.is_alive() else (0.20 if result["status"] == "ok" else 0.0)

    return max(-1.0, min(1.0, reward))


def format_prompt(text, tokenizer, system_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )