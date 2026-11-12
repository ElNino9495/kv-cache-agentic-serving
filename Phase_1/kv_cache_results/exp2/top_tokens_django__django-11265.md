# Consensus scoring: django__django-11265 (cut at turn 7, 4037 tokens)

Scores are consensus_N4: attention from 4 sampled 10-token draft guesses, max over each guess, averaged over the 4. Sinks (first 4) and the recent window (last 64) are excluded because they are always kept.

## What the 4 draft guesses wrote next

1. `I will print lines 165 to 175 of`
2. `I will run the tests in `tests/filtered`
3. `I will run the tests for the `with_m`
4. `I will modify the existing tests in `tests/`

## What the real agent wrote next (first 300 chars)

```
I will search for occurrences of `filtered_relations` or `FilteredRelation` in `django/db/models/sql/query.py`.

```bash
grep -rn "filtered_relations" django/db/models/sql/
```<|eot_id|>
```

## Top 15 tokens (most important)

| Rank | Score | Where | Token in context |
|---|---|---|---|
| 1 | 0.0111 | agent | output><\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>⏎⏎I**[ will]** print lines 135 to 155 of |
| 2 | 0.0107 | agent | ><\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>⏎⏎I will**[ print]** lines 135 to 155 of ' |
| 3 | 0.0097 | agent | I will print lines 135 to **[155]** of 'tests/filtered_relation/tests.py |
| 4 | 0.0080 | agent | ><\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>⏎⏎I will**[ print]** lines 80 to 115 from ' |
| 5 | 0.0079 | agent | output><\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>⏎⏎I**[ will]** print lines 80 to 115 from |
| 6 | 0.0077 | agent |  print lines 135 to 155 of**[ ']**tests/filtered_relation/tests.py'.⏎⏎''' |
| 7 | 0.0076 | agent |  of 'tests/filtered_relation/tests.py**['.⏎⏎]**'''bash⏎sed -n '135 |
| 8 | 0.0072 | agent | </output><\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>⏎⏎**[I]** will print lines 135 to 155 |
| 9 | 0.0071 | agent | assistant<\|end_header_id\|>⏎⏎I will print lines **[135]** to 155 of 'tests/filtered |
| 10 | 0.0061 | task | , in Django tests (django/tests/**[filtered]**_relation/tests.py) if we change this |
| 11 | 0.0060 | agent | <\|end_header_id\|>⏎⏎I will run the filtered_relation**[ tests]** to see if our test suite is currently |
| 12 | 0.0060 | agent |  will print lines 135 to 155**[ of]** 'tests/filtered_relation/tests.py'.⏎⏎ |
| 13 | 0.0057 | agent |  from 'tests/filtered_relation/tests.py**['.⏎⏎]**'''bash⏎sed -n '80 |
| 14 | 0.0056 | agent |  lines 135 to 155 of '**[tests]**/filtered_relation/tests.py'.⏎⏎'''bash |
| 15 | 0.0054 | header | (⏎</output><\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>**[⏎⏎]**I will print lines 135 to  |

## Bottom 15 tokens (least important)

| Rank | Score | Where | Token in context |
|---|---|---|---|
| 1 | 0.000001 | task | /django/db/models/sql/query.py",**[ line]** 1028, in solve_lookup_type |
| 2 | 0.000001 | task | /django/db/models/sql/query.py",**[ line]** 1164, in build_filter⏎ |
| 3 | 0.000001 | task | 1028, in solve_lookup_type⏎**[	]**_, field, _, lookup_parts = self |
| 4 | 0.000001 | task |  line 1028, in solve_lookup**[_type]**⏎	_, field, _, lookup_parts |
| 5 | 0.000001 | task | home/lucas/dev/overmind/**[ven]**vs/release/lib/python3.6/site |
| 6 | 0.000001 | task | _to_path(lookup_splitted,**[ self]**.get_meta())⏎ File "/home/lucas |
| 7 | 0.000002 | task | /django/db/models/sql/query.py",**[ line]** 1287, in _add_q |
| 8 | 0.000002 | task | 	"Choices are: %s"**[ %]** (name, ", ".join(available |
| 9 | 0.000002 | task | /release/lib/python3.6/site-packages**[/d]**jango/db/models/sql/query.py", line |
| 10 | 0.000002 | task | /sql/query.py", line 1028**[,]** in solve_lookup_type⏎	_, field |
| 11 | 0.000002 | task | itted, self.get_meta())⏎ File "/**[home]**/lucas/dev/overmind/ven |
| 12 | 0.000002 | task | home/lucas/dev/overmind/**[ven]**vs/release/lib/python3.6/site |
| 13 | 0.000002 | task | /dev/overmind/venvs/release**[/lib]**/python3.6/site-packages/django |
| 14 | 0.000002 | task | /dev/overmind/venvs/release**[/lib]**/python3.6/site-packages/django |
| 15 | 0.000002 | task |  in solve_lookup_type⏎	_, field**[,]** _, lookup_parts = self.names_to_path |

## Where the top 10% of tokens come from

| Part of conversation | Share of context | Share of top 10% |
|---|---|---|
| system | 0% | 3% |
| task | 51% | 22% |
| agent | 8% | 40% |
| tool | 41% | 29% |
| header | 1% | 7% |
