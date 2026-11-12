# Consensus scoring: sphinx-doc__sphinx-8548 (cut at turn 9, 4180 tokens)

Scores are consensus_N4: attention from 4 sampled 10-token draft guesses, max over each guess, averaged over the 4. Sinks (first 4) and the recent window (last 64) are excluded because they are always kept.

## What the 4 draft guesses wrote next

1. `I will view lines 810 to 820 of`
2. `I will inspect lines 812 to 820 of`
3. `I will search for lines 712 to 780`
4. `I will analyze the line that makes a copy of`

## What the real agent wrote next (first 300 chars)

```
I will search for class definitions preceding line 680 to see which class this is in.

```bash
grep -n "^class " sphinx/ext/autodoc/__init__.py | head -n 30
```<|eot_id|>
```

## Top 15 tokens (most important)

| Rank | Score | Where | Token in context |
|---|---|---|---|
| 1 | 0.0155 | agent | I will print lines 780 to **[810]** of 'sphinx/ext/autod |
| 2 | 0.0122 | agent | assistant<\|end_header_id\|>⏎⏎I will print lines **[780]** to 810 of 'sphinx/ext |
| 3 | 0.0114 | agent | ><\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>⏎⏎I will**[ print]** lines 780 to 810 of ' |
| 4 | 0.0087 | agent | output><\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>⏎⏎I**[ will]** print lines 780 to 810 of |
| 5 | 0.0086 | agent | ><\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>⏎⏎I will**[ view]** lines 760 to 780 in ' |
| 6 | 0.0081 | agent | assistant<\|end_header_id\|>⏎⏎I will view lines **[760]** to 780 in 'sphinx/ext |
| 7 | 0.0079 | agent | <\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>⏎⏎I will view**[ lines]** 760 to 780 in 's |
| 8 | 0.0073 | agent |  will print lines 780 to 810**[ of]** 'sphinx/ext/autodoc |
| 9 | 0.0069 | agent | I will view lines 760 to **[780]** in 'sphinx/ext/autod |
| 10 | 0.0067 | agent | ><\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>⏎⏎I will**[ inspect]** lines 570 to 595 of ' |
| 11 | 0.0067 | agent | output><\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>⏎⏎I**[ will]** view lines 760 to 780 in |
| 12 | 0.0067 | agent | ><\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>⏎⏎I will**[ print]** lines 680 to 760 in ' |
| 13 | 0.0063 | agent | <\|start_header_id\|>assistant<\|end_header_id\|>⏎⏎I will inspect lines**[ ]**570 to 595 of 'sphinx |
| 14 | 0.0062 | agent | <\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>⏎⏎I will inspect**[ lines]** 570 to 595 of 's |
| 15 | 0.0060 | agent | output><\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>⏎⏎I**[ will]** list the contents of the 'sphinx |

## Bottom 15 tokens (least important)

| Rank | Score | Where | Token in context |
|---|---|---|---|
| 1 | 0.000002 | tool | ⏎./tests/roots/test-ext-aut**[od]**oc/bug2437/autod |
| 2 | 0.000002 | tool | oo.py⏎./tests/roots/test**[-ext]**-doctest-with-autodoc⏎./ |
| 3 | 0.000002 | tool | ./tests/roots/test-root/aut**[od]**oc.txt⏎./tests/roots/test |
| 4 | 0.000002 | tool | ./tests/roots/test-root/aut**[od]**oc_target.py⏎./tests/roots |
| 5 | 0.000003 | tool | _dummy_module.py⏎./tests/roots**[/test]**-ext-autodoc/bug2437 |
| 6 | 0.000003 | tool | odoc.txt⏎./tests/roots**[/test]**-root/autodoc_target.py⏎ |
| 7 | 0.000003 | tool | _module.py⏎./tests/roots/test**[-ext]**-autodoc/bug2437/ |
| 8 | 0.000003 | tool | _foo.py⏎./tests/roots**[/test]**-ext-doctest-with-autodoc⏎ |
| 9 | 0.000003 | tool | ./tests/roots/test-ext-autod**[oc]**/bug2437/autodoc |
| 10 | 0.000003 | tool | oc_target.py⏎./tests/roots**[/test]**-warnings/autodoc_fod |
| 11 | 0.000003 | tool | unction.py⏎./tests/test_ext_aut**[od]**oc_configs.py⏎./tests/test_ext |
| 12 | 0.000003 | tool | _attr_docs():⏎sphinx/ext/autos**[um]**mary/generate.py:274:            |
| 13 | 0.000003 | tool | ⏎./tests/roots/test-ext-aut**[od]**oc/autodoc_dummy_module.py |
| 14 | 0.000003 | tool | ⏎./tests/roots/test-root/**[aut]**odoc.txt⏎./tests/roots |
| 15 | 0.000003 | tool | _events.py⏎./tests/test_ext_aut**[od]**oc_mock.py⏎./tests/test_ext |

## Where the top 10% of tokens come from

| Part of conversation | Share of context | Share of top 10% |
|---|---|---|
| system | 0% | 2% |
| task | 26% | 23% |
| agent | 10% | 41% |
| tool | 62% | 27% |
| header | 1% | 7% |
