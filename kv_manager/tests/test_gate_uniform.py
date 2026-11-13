import pytest
import torch

from experiments.gate import data
from experiments.gate.fakequant import roundtrip, roundtrip_blocks_unpacked
from kvm.quant import block_bytes, roundtrip_blocks


def _kv(seed=0, tokens=3 * 64 + 5):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 8, tokens, 128, generator=g)
    x[..., :, 7] *= 20  # a key outlier channel
    x[..., 11, :] *= 5  # a large token
    return x.to(torch.bfloat16)


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("B", [16, 64])
@pytest.mark.parametrize("kind", ["key", "value"])
def test_unpacked_matches_kvm_quant_bit_for_bit(bits, B, kind):
    x = _kv()
    assert torch.equal(roundtrip_blocks_unpacked(x, bits, B, kind), roundtrip_blocks(x, bits, B, kind))


@pytest.mark.parametrize("kind", ["key", "value"])
def test_int3_error_sits_between_int2_and_int4_and_keeps_tail(kind):
    x, B = _kv(1), 16
    err = {b: (roundtrip(x, b, B, kind) - x).float().abs().mean().item() for b in (2, 3, 4)}
    assert err[4] < err[3] < err[2]
    y = roundtrip(x, 3, B, kind)
    full = (x.shape[-2] // B) * B
    assert torch.equal(y[..., full:, :], x[..., full:, :])
    # at most 8 distinct levels per quantization group (keys: per channel within a block)
    if kind == "key":
        assert all(y[0, 0, b * B:(b + 1) * B, c].unique().numel() <= 8 for b in range(3) for c in (0, 7))


def test_block_bytes_int3():
    # 8 KV heads x 128, B=16: codes 2 x 6144 + key scale/zero 4096 + value scale/zero 512
    assert block_bytes(3, 16, 8, 128) == 16896
    assert block_bytes(2, 16, 8, 128) < block_bytes(3, 16, 8, 128) < block_bytes(4, 16, 8, 128)


class CharTok:
    """Tokenizer stand-in: one id per character, special strings kept whole."""
    bos_token = "<s>"

    def __call__(self, s, add_special_tokens=False):
        class R:
            pass
        r = R()
        r.input_ids = [ord(c) for c in s]
        return r


TRAJ = {"instance_id": "x", "messages": [
    {"role": "system", "content": "sys"}, {"role": "user", "content": "task"},
    {"role": "assistant", "content": "a1"}, {"role": "tool", "content": "t1"},
    {"role": "assistant", "content": "a2"}, {"role": "tool", "content": "t2"},
    {"role": "assistant", "content": "a3"}, {"role": "tool", "content": "t3"},
    {"role": "assistant", "content": "a4"}]}


def test_llama_format_unchanged_for_replay():
    assert data.msg("tool", "out") == "<|start_header_id|>ipython<|end_header_id|>\n\nout<|eot_id|>"
    assert data.msg("user", "hi") == "<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>"
    assert data.ASSISTANT_HEADER == "<|start_header_id|>assistant<|end_header_id|>\n\n"
    assert data.EOT == "<|eot_id|>"


@pytest.mark.parametrize("fmt", [data.LLAMA3, data.QWEN3])
def test_turn_selection_matches_length_selection(fmt):
    tok = CharTok()
    by_len = data.make_samples(tok, TRAJ, [1], 256, fmt=fmt)
    assert [s["turn"] for s in by_len] == [1]
    by_turn = data.make_samples(tok, TRAJ, [], 256, fmt=fmt, turns=[1, 3])
    assert [s["turn"] for s in by_turn] == [1, 3]
    assert by_turn[0] == by_len[0]
    with pytest.raises(ValueError):
        data.make_samples(tok, TRAJ, [], 256, fmt=fmt, turns=[9])


def test_qwen3_sample_text_matches_chat_template():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen/Qwen3-8B tokenizer not in the local HF cache")
    msgs = TRAJ["messages"]
    # history + fresh for the pause after a2/t2, i.e. turn 2: everything through t2 + generation prompt
    want = tok.apply_chat_template(msgs[:6], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prefix, turns = data.conversation(TRAJ, "", fmt=data.QWEN3)
    got = "".join(prefix)
    for i, (agent, tool, _) in enumerate(turns[:2]):
        got += agent + data.QWEN3.hist_end
        got += data.QWEN3.msg("tool", tool) + (data.QWEN3.gen_header if i == 1 else data.QWEN3.hist_header)
    assert got == want
    # the target closes with the model's end-of-turn token
    assert tok.eos_token == data.QWEN3.target_end
