# Text-to-SQL prototype: BIRD Mini-Dev on Kaggle (2× T4)

**Goal:** measure how well open-source LLMs translate English questions into SQL, using the 500-question **BIRD Mini-Dev**
benchmark, on Kaggle's free 2× NVIDIA T4 GPUs. The results are recorded so they can go straight into a thesis.

```
Prototyping/
├── README.md                          ← you are here: concepts, roadmap, best practices
├── kaggle/bird_minidev_text2sql.ipynb ← import this into Kaggle (the whole pipeline, explained step by step)
├── results/                           ← put downloaded run folders here (one folder per experiment)
└── dev/                               ← maintenance tools (you can ignore this folder)
    ├── notebook_source.py             ← editable source of the notebook
    ├── build_notebook.py              ← rebuilds the .ipynb from the source
    └── test_local.py                  ← runs all CPU-only steps on the real data, as a self-test
```

---

## 1. Concepts you need (read once)

| Term | Meaning |
|---|---|
| **Text-to-SQL** (NL2SQL) | Turning a natural-language question ("How many clients from Prague?") into an executable SQL query. |
| **Schema** | The structure of a database: tables, columns, types, and how tables link (primary and foreign keys). The model only sees a *text description* of it. |
| **Schema linking** | Working out which tables and columns a question refers to. It is the most common source of errors. |
| **BIRD** | A large text-to-SQL benchmark with real, messy databases ([bird-bench.github.io](https://bird-bench.github.io/)). |
| **Mini-Dev** | 500 questions from BIRD's dev split over 11 SQLite databases: 148 *simple*, 250 *moderate*, 102 *challenging*. |
| **Evidence** | BIRD's per-question hint, such as a formula or which value means what. The official setting includes it. |
| **Gold SQL** | The human-written reference query. |
| **EX** (execution accuracy) | % of questions where the predicted query returns the same set of rows as the gold query. **The main metric.** |
| **Soft-F1** | Partial credit at the cell level. It ignores column order and is forgiving about extra or missing columns. |
| **R-VES** | Rewards correct queries that are also fast, based on the time ratio to the gold query. It depends on the hardware. |
| **Greedy decoding** | `temperature = 0`: the model always picks its most likely next token, so outputs are repeatable. Standard for reporting. |
| **vLLM** | A fast LLM inference engine that batches all prompts together and manages GPU memory efficiently. |
| **Tensor parallelism (TP)** | Splitting each layer of a model across several GPUs. A 7B model in float16 (about 15 GB) needs both T4s. |
| **float16 vs bfloat16** | Two 16-bit number formats. T4s only support float16, so models run with `dtype="float16"`. |
| **Oracle run** | Feeding the gold SQL in as the "prediction". It must score 100 %, which proves the evaluator works. |

---

## 2. Your roadmap

Work through the milestones in order. Each one ends with a check, so you know it worked before moving on.

- [ ] **M0: Kaggle account ready.** Verify your phone number in Kaggle *Settings*. Kaggle requires this before it allows GPUs and internet access in notebooks.
- [ ] **M1: Evaluator verified (CPU session, costs no GPU quota).**
  1. *Create → New Notebook → File → Import Notebook* → upload `kaggle/bird_minidev_text2sql.ipynb`.
  2. In the right-hand panel: **Accelerator = None**, **Internet = On**.
  3. In Step 1 set `"backend": "oracle"`, then run all cells one by one (Shift+Enter).
  4. ✅ Check: Step 5b prints `Gold query status: {'ok': 500}` and Step 8 prints `EX = 100.00%`.
- [ ] **M2: GPU smoke test (about 10-15 min of GPU time).**
  1. Accelerator = **GPU T4 x2** (this restarts the session).
  2. Step 1: `"backend": "vllm"`, `"model_key": "qwen2.5-coder-1.5b"`, `"n_questions": 30`.
  3. Run all. ✅ Check: Step 2 shows two T4s, Step 7 prints generated SQL, and Step 9 shows a table with EX somewhere above 0.
- [ ] **M3: First real experiment (500 questions).** Set `"n_questions": 500` and a 7B model (start with `"omnisql-7b"`), then
  click **Save Version → Save & Run All (Commit)**. It runs in the background. Afterwards, open the version, go to *Output*, and download the zip into `results/`.
- [ ] **M4: Model comparison.** Repeat M3 for `qwen2.5-coder-7b` (general coding model) and `arctic-r1-7b` (RL-trained reasoning model).
  Step 10 compares them with a paired significance test.
- [ ] **M5: Error analysis.** Open `manual_review.csv` from your best run and label the 30 failures (see §4). This is where real insight comes from.
- [ ] **M6: Ablations.** For example, `use_evidence = False`, or `values_per_column = 0` vs `5`. Change **one** thing per run.

**GPU quota tips:** Kaggle gives roughly 30 GPU-hours per week (check the quota shown in your account). An open GPU session uses
quota even while idle, so **stop the session** (power icon) when you're done. Do all CPU work (M1, analysis) with Accelerator = None.
Interactive sessions also time out when idle, so use *Save & Run All* for anything long.

---

## 3. Models in the notebook (all public; no Hugging Face token needed)

| Key | Model | Type | Published BIRD-**dev** EX (greedy) |
|---|---|---|---|
| `qwen2.5-coder-1.5b` | Qwen2.5-Coder-1.5B-Instruct | small general code model (smoke tests) | n/a |
| `qwen2.5-coder-7b` | Qwen2.5-Coder-7B-Instruct | general code model (baseline) | 50.9 (in the OmniSQL paper) |
| `omnisql-7b` | OmniSQL-7B | fine-tuned on 2.5M synthetic text-to-SQL pairs | 63.9 (OmniSQL paper) |
| `arctic-r1-7b` | Arctic-Text2SQL-R1-7B | Qwen2.5-Coder + reinforcement learning (reasoning) | 68.9 (model card) |

Expect **lower** numbers on Mini-Dev with this notebook. Mini-Dev is a different (and on average harder) subset, and the published
setups also retrieve question-relevant database values, which we don't (yet). What matters for the thesis is that **all models are
measured the same way**.

---

## 4. How results are recorded

Each run writes `runs/<run_id>/`. The run id encodes the date, model, prompt style, evidence on/off, and number of questions.

| File | Contents |
|---|---|
| `run_config.json` | every setting, software versions, GPU, exact model revision (commit hash) |
| `generations.jsonl` | raw model outputs (lets you re-score without spending GPU time) |
| `per_question.csv` | one row per question: gold/pred SQL, EX, Soft-F1, R-VES reward, **error category**, tokens, SQL features |
| `summary.json` | EX / Soft-F1 / R-VES per difficulty, 95 % CI, category counts, timing |
| `predict_official_format.json` | predictions in the official BIRD format, for cross-checking with the official scripts |
| `manual_review.csv` | 30 random failures with blank `error_type` / `gold_is_wrong` / `notes` columns for you to fill in |
| `figures/breakdowns.png` | EX per database, outcome categories, EX by SQL feature |

**Manual error labels** (put one in `error_type`; based on common taxonomies in the literature):
`schema_linking` (wrong table/column) · `join` · `filter_value` (wrong literal or condition) · `aggregation_grouping` ·
`ordering_limit` · `nested_logic` · `projection` (wrong/extra columns) · `evidence_misuse` · `dialect` · `ambiguous_question`.
Always also mark `gold_is_wrong`, because about half of Mini-Dev's labels have problems (see §5).

---

## 5. Best practices from the literature, and how this prototype follows them

1. **Report the official metrics, broken down by difficulty.** EX is primary, with Soft-F1 and R-VES secondary (BIRD Mini-Dev).
   → Step 9 prints the official table layout. Two documented deviations: gold SQL gets 120 s instead of 30 s, and R-VES uses 3 timing repeats
   instead of 100. Note also that the official Soft-F1 is row-order sensitive (Step 5b shows why).
2. **Say exactly how you decoded.** Greedy, single sample, versus sampling with majority voting can differ by several points, and leaderboards
   are not always clear about it (noted in the Arctic-Text2SQL-R1 paper). → `temperature = 0`, `n = 1` by default, saved in `run_config.json`.
3. **Use each model's own prompt format** and report it. A mismatched prompt measures the prompt, not the model. → OmniSQL and Arctic templates copied from their repositories.
4. **Report with *and* without evidence.** BIRD's evidence makes a large difference, and real users won't provide it. → `use_evidence`.
5. **Add uncertainty.** 500 questions give roughly ±4-5 points of 95 % CI. Questions within one database are correlated, so use
   **clustered** standard errors (Miller 2024). → Step 9a.
6. **Compare models question-by-question** with a paired test (McNemar), not by eyeballing two averages. → Step 10.
7. **Go beyond one number.** Break down by database, SQL complexity, and error type, and do a manual error analysis on a random sample
   (NL2SQL survey, Liu et al. 2024; NL2SQL360; NL2SQL-Bugs). → Steps 8, 9b and `manual_review.csv`.
8. **Treat the gold labels with suspicion.** Jin et al. (PVLDB 2026) found annotation errors in **52.8 % of BIRD Mini-Dev**. Wretblad et al.
   (ACL 2024) found noise in about 49 % of BIRD's *financial* domain. FLEX (NAACL 2025) shows EX produces both false positives and false negatives.
   → Step 9c re-scores your predictions against the corrected labels (Arcwise-Plat-SQL). Put this in your *threats to validity*.
9. **Record efficiency, not just accuracy.** Tokens generated, generation time, and throughput on named hardware. → `summary.json`.
10. **Don't tune on your test set.** If you change prompts based on Mini-Dev failures and then report Mini-Dev scores, they are optimistic.
    Fix your setups in advance, or tune on a different split (e.g. part of BIRD *train*).
11. **Make it reproducible.** Fixed seeds, pinned versions, model commit hashes, one notebook version per experiment. → Kaggle version history plus `run_config.json`.

---

## 6. Experiment ideas once the basics work

- **Self-consistency:** sample 8 answers at temperature ≈ 0.8 and pick the most common *execution result*. OmniSQL reports +1 to +10 points from this.
- **Self-correction:** when the SQL errors out, feed the error message back to the model once.
- **Richer schema:** add column descriptions from `database_description/*.csv`, or retrieve values that match the question (BM25), as OmniSQL does.
- **Bigger models:** 14B/32B models in 4-bit (AWQ/GPTQ) can fit on 2× T4.
- **Label-noise study:** evaluate on the fully corrected *Arcwise-Plat* set as a second benchmark version.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `GPU: none visible` | Settings → Accelerator → GPU T4 x2 |
| Download fails / `URLError` | Settings → Internet → On (needs phone verification) |
| vLLM import error after install | *Run → Restart & clear outputs*, run again. Still failing? Set `"vllm_version": ""` (latest), or use `"backend": "hf"` (slow but robust) |
| CUDA out of memory | lower `max_model_len` (e.g. 8192) or `gpu_memory_utilization` (0.85) |
| `Bfloat16 is only supported on GPUs with compute capability ≥ 8.0` | something overrode `dtype`; the notebook uses `float16`, which T4s need |
| Many `no_sql_truncated` | raise `max_new_tokens` in `MODEL_PRESETS`, or the model is rambling |

## 8. References

- BIRD: Li et al., *Can LLM Already Serve as a Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs*, NeurIPS 2023. Mini-Dev: [github.com/bird-bench/mini_dev](https://github.com/bird-bench/mini_dev)
- OmniSQL: Li et al., *OmniSQL: Synthesizing High-quality Text-to-SQL Data at Scale*, 2025. [arXiv:2503.02240](https://arxiv.org/abs/2503.02240), [code](https://github.com/RUCKBReasoning/OmniSQL)
- Arctic-Text2SQL-R1: Yao et al., *Arctic-Text2SQL-R1: Simple Rewards, Strong Reasoning in Text-to-SQL*, 2025. [arXiv:2505.20315](https://arxiv.org/abs/2505.20315), [model](https://huggingface.co/Snowflake/Arctic-Text2SQL-R1-7B)
- Jin, Choi, Zhu, Kang, *Pervasive Annotation Errors Break Text-to-SQL Benchmarks and Leaderboards*, PVLDB 19, 2026. [arXiv:2601.08778](https://arxiv.org/abs/2601.08778), [data](https://github.com/uiuc-kang-lab/text_to_sql_benchmarks)
- Wretblad et al., *Understanding the Effects of Noise in Text-to-SQL: An Examination of the BIRD-Bench Benchmark*, ACL 2024. [arXiv:2402.12243](https://arxiv.org/abs/2402.12243)
- Kim et al., *FLEX: Expert-level False-Less EXecution Metric for Reliable Text-to-SQL Benchmark*, NAACL 2025. [arXiv:2409.19014](https://arxiv.org/abs/2409.19014)
- Liu et al., *A Survey of Text-to-SQL in the Era of LLMs: Where are we, and where are we going?*, 2024 (earlier title: *A Survey of NL2SQL with Large Language Models…*). [arXiv:2408.05109](https://arxiv.org/abs/2408.05109)
- Li et al., *The Dawn of Natural Language to SQL: Are We Fully Ready?* (NL2SQL360), VLDB 2024. [arXiv:2406.01265](https://arxiv.org/abs/2406.01265)
- Miller, *Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations*, 2024. [arXiv:2411.00640](https://arxiv.org/abs/2411.00640)
- vLLM: Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023. [docs](https://docs.vllm.ai/)
