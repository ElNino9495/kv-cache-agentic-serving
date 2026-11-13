"""Real agent recordings for the gate experiment, cut into (history, fresh, target) samples.

Source: mini-SWE-agent runs on SWE-bench Verified (same recordings as Phase 1):
github.com/john-b-yang/20260901_mini-v2.4.2_gemini-3-5-flash

A sample mirrors one tool-call pause:
  history  everything up to and including the agent message that issued the tool call.
           This is the KV that sits offloaded on CPU during the pause.
  fresh    the tool output that just came back + the assistant header. Prefilled fresh
           in full precision on top of the reloaded history.
  target   the real next agent message, teacher-forced to measure damage.

    python -m experiments.gate.data            # download + convert all recordings
"""

import argparse
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "swe"
TRAJ_REPO = "john-b-yang/20260901_mini-v2.4.2_gemini-3-5-flash"
RAW_URL = f"https://raw.githubusercontent.com/{TRAJ_REPO}/main"

# Longest recording from each of 10 projects (Phase 1, Experiment 1): 18k-56k tokens.
INSTANCES = [
    "sphinx-doc__sphinx-8548",
    "django__django-11265",
    "sympy__sympy-12419",
    "astropy__astropy-13398",
    "pytest-dev__pytest-5787",
    "pydata__xarray-4687",
    "mwaskom__seaborn-3069",
    "psf__requests-2931",
    "scikit-learn__scikit-learn-11310",
    "pallets__flask-5014",
]


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as r:
        dest.write_bytes(r.read())


def _agent_text(msg: dict) -> str:
    """The agent's sentence plus the shell command(s) it ran."""
    parts = [msg.get("content") or ""]
    for call in msg.get("tool_calls") or []:
        args = call.get("function", {}).get("arguments", "")
        try:
            cmd = json.loads(args).get("command", args)
        except (json.JSONDecodeError, AttributeError):
            cmd = args
        parts.append(f"```bash\n{cmd}\n```")
    return "\n\n".join(p for p in parts if p)


def prepare(instance_id: str, data_dir: Path = DATA_DIR) -> Path:
    raw = data_dir / "raw" / f"{instance_id}.traj.json"
    _download(f"{RAW_URL}/trajs/{instance_id}.traj.json", raw)
    traj = json.loads(raw.read_text(encoding="utf-8"))
    messages = []
    for m in traj["messages"]:
        role = m.get("role")
        if role in ("system", "user", "tool") and isinstance(m.get("content"), str):
            messages.append({"role": role, "content": m["content"]})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": _agent_text(m)})
    out = data_dir / "replay" / f"{instance_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"instance_id": instance_id, "messages": messages}), encoding="utf-8")
    return out


def load(instance_id: str, data_dir: Path = DATA_DIR) -> dict:
    path = data_dir / "replay" / f"{instance_id}.json"
    if not path.exists():
        prepare(instance_id, data_dir)
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ChatFormat:
    """How one model family's chat template renders the replayed conversation.

    hist_header  opens an assistant turn that is already part of the history
    gen_header   opens the assistant turn being generated (the teacher-forced target)
    hist_end     closes an assistant turn inside the history
    target_end   closes the target (the model's own end-of-turn token)
    native_context  longest sequence the model is trained for without RoPE scaling (0 = no cap)
    """
    name: str
    templates: dict
    hist_header: str
    gen_header: str
    hist_end: str
    target_end: str
    native_context: int = 0

    def msg(self, role: str, content: str) -> str:
        return self.templates.get(role, self.templates["default"]).format(role=role, content=content)


# Llama-3: tool results use the "ipython" role. The default template's "Cutting Knowledge Date"
# system header is left out, as in Phase 1 and the first gate run.
LLAMA3 = ChatFormat(
    name="llama3",
    templates={"default": "<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>",
               "tool": "<|start_header_id|>ipython<|end_header_id|>\n\n{content}<|eot_id|>"},
    hist_header="<|start_header_id|>assistant<|end_header_id|>\n\n",
    gen_header="<|start_header_id|>assistant<|end_header_id|>\n\n",
    hist_end="<|eot_id|>",
    target_end="<|eot_id|>",
)

# Qwen3 ChatML with thinking off, matching tokenizer.apply_chat_template(..., enable_thinking=False,
# add_generation_prompt=True): tool results become a user turn wrapped in <tool_response>, earlier
# assistant turns (no reasoning) render plainly, and only the turn being generated gets the empty
# <think></think> block. Native context is 32,768 tokens (no YaRN).
QWEN3 = ChatFormat(
    name="qwen3",
    templates={"default": "<|im_start|>{role}\n{content}<|im_end|>\n",
               "tool": "<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"},
    hist_header="<|im_start|>assistant\n",
    gen_header="<|im_start|>assistant\n<think>\n\n</think>\n\n",
    hist_end="<|im_end|>\n",
    target_end="<|im_end|>",
    native_context=32768,
)

FORMATS = {"llama": LLAMA3, "qwen3": QWEN3}


def chat_format(model_type: str) -> ChatFormat:
    """Chat format for a HF config.model_type."""
    if model_type not in FORMATS:
        raise ValueError(f"no chat format for model_type {model_type!r}; known: {sorted(FORMATS)}")
    return FORMATS[model_type]


# Llama-3 names kept for bench/replay.py.
def msg(role: str, content: str) -> str:
    return LLAMA3.msg(role, content)


ASSISTANT_HEADER = LLAMA3.hist_header
EOT = LLAMA3.hist_end


def conversation(traj: dict, bos: str, nonce: str = "",
                 fmt: ChatFormat = LLAMA3) -> tuple[list[str], list[tuple[str, str, str]]]:
    """Returns (prefix pieces, [(agent message, tool output, next agent message), ...]).

    `nonce` is prepended to the system prompt so replayed copies of one recording do not
    share prefix-cache blocks.
    """
    msgs = traj["messages"]
    head = [m for m in msgs if m["role"] in ("system", "user")]
    prefix = [bos] + [fmt.msg(m["role"], (nonce if i == 0 else "") + m["content"]) for i, m in enumerate(head)]
    prefix.append(fmt.hist_header)
    rest = [m for m in msgs if m["role"] in ("assistant", "tool")]
    pairs, pending = [], None
    for m in rest:
        if m["role"] == "assistant":
            pending = m["content"]
        elif m["role"] == "tool" and pending is not None:
            pairs.append((pending, m["content"]))
            pending = None
    agents = [a for a, _ in pairs] + ([pending] if pending is not None else [])
    turns = [(a, t, agents[i + 1]) for i, (a, t) in enumerate(pairs) if i + 1 < len(agents)]
    return prefix, turns


def make_samples(tokenizer, traj: dict, hist_lens: list[int], max_target_tokens: int,
                 fmt: ChatFormat = LLAMA3, turns: list[int] | None = None) -> list[dict]:
    """Cut a recording into (history, fresh, target) samples.

    By default, one sample per requested history length: the first tool-call pause whose history
    reaches that many tokens. With `turns`, exactly those pauses (1-based turn numbers, as in
    samples.csv), so different models are evaluated on the same pauses.
    """
    enc = lambda s: tokenizer(s, add_special_tokens=False).input_ids
    prefix, pauses = conversation(traj, tokenizer.bos_token or "", fmt=fmt)
    hist = [t for p in prefix for t in enc(p)]
    samples, wants = [], sorted(hist_lens)
    pick = set(turns) if turns is not None else None
    for k, (agent, tool, nxt) in enumerate(pauses):
        hist = hist + enc(agent + fmt.hist_end)
        if pick is not None:
            take = k + 1 in pick
        else:
            take = bool(wants) and len(hist) >= wants[0]
            while wants and len(hist) >= wants[0]:
                wants.pop(0)
        if take:
            samples.append({
                "instance_id": traj["instance_id"],
                "turn": k + 1,
                "hist_ids": list(hist),
                "fresh_ids": enc(fmt.msg("tool", tool) + fmt.gen_header),
                "target_ids": enc(nxt + fmt.target_end)[:max_target_tokens],
            })
        hist = hist + enc(fmt.msg("tool", tool) + fmt.hist_header)
        if (pick is not None and len(samples) == len(pick)) or (pick is None and not wants):
            break
    if pick is not None and len(samples) != len(pick):
        found = {s["turn"] for s in samples}
        raise ValueError(f"{traj['instance_id']}: turns {sorted(pick - found)} not in the recording")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="*", default=INSTANCES)
    args = parser.parse_args()
    for i in args.ids:
        print(f"{i}: {prepare(i)}")


if __name__ == "__main__":
    main()
