# Uniform reload across target models

dNLL is in nats per target token (0 = identical to bf16 reload). bytes = reload bytes / bf16 reload bytes. int3 is quantized with kvm.quant's arithmetic without packing; its bytes are kvm.quant.block_bytes(3, ...).

## unsloth/Llama-3.1-8B-Instruct

31 pauses evaluated, 0 skipped for context length.

### Block size 16

Control uniform_bf16: max |dNLL| 0.00e+00 (OK)

| scheme | bytes | dNLL | top-1 | dNLL - uniform_int4 [95% CI] |
|---|---|---|---|---|
| uniform_int4 | 0.325 | +0.0009 | 0.991 |  |
| uniform_int3 | 0.262 | -0.0001 | 0.987 | -0.0010 [-0.0051, +0.0031] |
| uniform_int2 | 0.200 | +0.0332 | 0.955 | +0.0323 [+0.0179, +0.0481] |

### Block size 64

Control uniform_bf16: max |dNLL| 0.00e+00 (OK)

| scheme | bytes | dNLL | top-1 | dNLL - uniform_int4 [95% CI] |
|---|---|---|---|---|
| uniform_int4 | 0.282 | +0.0035 | 0.984 |  |
| uniform_int3 | 0.220 | +0.0057 | 0.978 | +0.0022 [-0.0056, +0.0098] |
| uniform_int2 | 0.158 | +0.0432 | 0.945 | +0.0397 [+0.0202, +0.0608] |

## unsloth/Llama-3.2-3B-Instruct

31 pauses evaluated, 0 skipped for context length.

### Block size 16

Control uniform_bf16: max |dNLL| 0.00e+00 (OK)

| scheme | bytes | dNLL | top-1 | dNLL - uniform_int4 [95% CI] |
|---|---|---|---|---|
| uniform_int4 | 0.325 | +0.0006 | 0.991 |  |
| uniform_int3 | 0.262 | +0.0055 | 0.986 | +0.0048 [-0.0006, +0.0108] |
| uniform_int2 | 0.200 | +0.0347 | 0.935 | +0.0341 [+0.0191, +0.0488] |

### Block size 64

Control uniform_bf16: max |dNLL| 0.00e+00 (OK)

| scheme | bytes | dNLL | top-1 | dNLL - uniform_int4 [95% CI] |
|---|---|---|---|---|
| uniform_int4 | 0.282 | -0.0001 | 0.988 |  |
| uniform_int3 | 0.220 | +0.0032 | 0.972 | +0.0033 [-0.0049, +0.0110] |
| uniform_int2 | 0.158 | +0.0378 | 0.925 | +0.0380 [+0.0087, +0.0698] |

## Qwen/Qwen3-8B

27 pauses evaluated, 4 skipped for context length (sphinx-doc__sphinx-8548 t76: 33657, django__django-11265 t64: 33458, sympy__sympy-12419 t72: 37775, pydata__xarray-4687 t85: 34400).

### Block size 16

Control uniform_bf16: max |dNLL| 0.00e+00 (OK)

| scheme | bytes | dNLL | top-1 | dNLL - uniform_int4 [95% CI] |
|---|---|---|---|---|
| uniform_int4 | 0.325 | -0.0061 | 0.996 |  |
| uniform_int3 | 0.263 | -0.0036 | 0.987 | +0.0025 [-0.0100, +0.0154] |
| uniform_int2 | 0.201 | -0.0134 | 0.962 | -0.0073 [-0.0414, +0.0232] |

### Block size 64

Control uniform_bf16: max |dNLL| 0.00e+00 (OK)

| scheme | bytes | dNLL | top-1 | dNLL - uniform_int4 [95% CI] |
|---|---|---|---|---|
| uniform_int4 | 0.282 | -0.0013 | 0.991 |  |
| uniform_int3 | 0.220 | -0.0266 | 0.983 | -0.0254 [-0.0462, -0.0080] |
| uniform_int2 | 0.158 | +0.0223 | 0.939 | +0.0236 [-0.0253, +0.0729] |

## Across models, on the pauses every model ran

27 common pauses. Cells: mean dNLL (bytes).

Block size 16:

| model | uniform_int4 | uniform_int3 | uniform_int2 |
|---|---|---|---|
| llama-3.1-8b-instruct | +0.0001 (0.325) | -0.0008 (0.263) | +0.0229 (0.201) |
| llama-3.2-3b-instruct | -0.0011 (0.325) | +0.0009 (0.263) | +0.0383 (0.201) |
| qwen3-8b | -0.0061 (0.325) | -0.0036 (0.263) | -0.0134 (0.201) |

Block size 64:

| model | uniform_int4 | uniform_int3 | uniform_int2 |
|---|---|---|---|
| llama-3.1-8b-instruct | +0.0037 (0.282) | +0.0053 (0.220) | +0.0359 (0.159) |
| llama-3.2-3b-instruct | -0.0002 (0.282) | +0.0006 (0.220) | +0.0312 (0.159) |
| qwen3-8b | -0.0013 (0.282) | -0.0266 (0.220) | +0.0223 (0.158) |

## Uniform int3 vs draft-guided rows from the first gate run (metrics.csv)

History lengths match the first run on 31/31 pauses.
B=16: 31 paired pauses; max |ref NLL new - old| 0.00e+00, max |uniform_int4 dNLL new - old| 0.00e+00.
B=64: 31 paired pauses; max |ref NLL new - old| 0.00e+00, max |uniform_int4 dNLL new - old| 0.00e+00.

Draft rows with mean bytes within 0.03 of uniform_int3's, paired on the same pauses. Negative difference = uniform int3 better.

### Block size 16: uniform_int3 at 0.262 bytes, dNLL -0.0001

| method | hi | keep | bytes | dNLL (method) | dNLL (int3, same pauses) | int3 - method [95% CI] |
|---|---|---|---|---|---|---|
| consensus | bf16 | 0.05 | 0.240 | +0.0129 | -0.0001 | -0.0131 [-0.0200, -0.0068] |
| consensus | bf16 | 0.1 | 0.280 | +0.0091 | -0.0001 | -0.0092 [-0.0194, -0.0005] |
| consensus | int4 | 0.4 | 0.250 | +0.0057 | -0.0001 | -0.0058 [-0.0131, +0.0010] |
| specprefill | bf16 | 0.05 | 0.240 | +0.0173 | -0.0001 | -0.0175 [-0.0302, -0.0074] |
| specprefill | bf16 | 0.1 | 0.280 | +0.0079 | -0.0001 | -0.0081 [-0.0170, +0.0013] |
| specprefill | int4 | 0.4 | 0.250 | +0.0052 | -0.0001 | -0.0053 [-0.0132, +0.0026] |

### Block size 64: uniform_int3 at 0.220 bytes, dNLL +0.0057

| method | hi | keep | bytes | dNLL (method) | dNLL (int3, same pauses) | int3 - method [95% CI] |
|---|---|---|---|---|---|---|
| consensus | bf16 | 0.05 | 0.200 | +0.0259 | +0.0057 | -0.0202 [-0.0337, -0.0071] |
| consensus | bf16 | 0.1 | 0.242 | +0.0257 | +0.0057 | -0.0200 [-0.0361, -0.0058] |
| consensus | int4 | 0.4 | 0.207 | +0.0187 | +0.0057 | -0.0129 [-0.0245, -0.0019] |
| specprefill | bf16 | 0.05 | 0.200 | +0.0295 | +0.0057 | -0.0238 [-0.0393, -0.0090] |
| specprefill | bf16 | 0.1 | 0.242 | +0.0312 | +0.0057 | -0.0255 [-0.0449, -0.0071] |
| specprefill | int4 | 0.4 | 0.207 | +0.0199 | +0.0057 | -0.0142 [-0.0266, -0.0027] |

