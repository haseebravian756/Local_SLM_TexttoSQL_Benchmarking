# Source of kaggle/bird_minidev_text2sql.ipynb.
# Cells are separated by "# %%" lines; "# %% [markdown]" cells hold Markdown (each line prefixed with "# ").
# Rebuild the notebook with:  py dev/build_notebook.py

# %% [markdown]
# # Text-to-SQL on BIRD Mini-Dev: a Kaggle (2× T4) evaluation prototype
#
# **What this notebook does:** it asks an open-source LLM to turn 500 English questions into SQL, runs that SQL on real
# SQLite databases, and measures how often the answers are correct. It uses the official BIRD Mini-Dev metrics.
#
# ```
#  question + evidence + DB schema ──► prompt ──► LLM (vLLM on 2×T4) ──► text ──► extract SQL ──► run on SQLite ──► predicted rows
#                                                                                                                     │ compare
#  gold SQL (written by humans) ─────────────────────────────────────────────────────────────────► run on SQLite ──► gold rows
#                                                                                                                     ▼
#                                    EX · Soft-F1 · R-VES · error category ──► files in /kaggle/working/runs/<run_id>/ ──► analytics
# ```
#
# **How to use it.** Run the cells from top to bottom. Every step starts with a short explanation, so read it before you run the cell.
#
# | Session | Accelerator | Steps | Goal |
# |---|---|---|---|
# | 1 | **None** (CPU, free) | 1 → 9 with `backend = "oracle"` | Prove the evaluator works: the "oracle" must score **100 % EX** |
# | 2 | **GPU T4 × 2** | 1 → 9 with a small model, `n_questions = 30` | Smoke-test the GPU pipeline in about 10 minutes |
# | 3+ | **GPU T4 × 2** | Full run, 500 questions, via *Save Version → Save & Run All* | Real results, one notebook version per experiment |
#
# Steps 10 and 11 compare runs and save your work.

# %% [markdown]
# ---
# ## STEP 1: Configure the experiment
#
# An **experiment run** is one combination of *model × prompt setup × data subset*. Every setting that can change the
# score is in `CFG` below. The whole `CFG` dict is saved with your results, so months later you can still tell exactly what
# produced a number. (That is the reproducibility habit examiners look for.)
#
# The most important knobs:
# - `model_key`: which model to test (see `MODEL_PRESETS`). Start with the small `qwen2.5-coder-1.5b`.
# - `backend`: `"vllm"` (fast GPU engine), `"hf"` (plain Hugging Face transformers, a slow fallback), or `"oracle"`
#   (no model at all: it "predicts" the gold SQL. Use it to test the evaluator; the score must be 100 %).
# - `use_evidence`: BIRD gives a hint ("evidence") with each question. The official setting uses it. Turning it off is a classic ablation.
# - `n_questions`: 500 for real runs; e.g. 30 for a quick smoke test.
# - `temperature = 0` means **greedy decoding**: the model always picks its most likely next token, so runs are (nearly) repeatable.
#   This is the standard setting for reporting benchmark scores.

# %%
import os, sys, json, re, time, math, random, shutil, zipfile, sqlite3, platform, subprocess, urllib.request
import datetime as dt
import importlib.util
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

CFG = {
    "model_key": "qwen2.5-coder-1.5b",  # a key of MODEL_PRESETS below
    "backend": "vllm",                  # "vllm" | "hf" | "oracle"
    "use_evidence": True,               # include BIRD's "evidence" hint (official setting)
    "n_questions": 500,                 # 500 = full Mini-Dev; use e.g. 30 for a smoke test
    "temperature": 0.0,                 # 0 = greedy decoding (standard for benchmark reporting)
    "seed": 42,
    "values_per_column": 2,             # example values shown per column in the schema prompt
    "max_model_len": 12288,             # prompt + answer token budget per question (KV-cache size)
    "gpu_memory_utilization": 0.90,     # fraction of each GPU's memory vLLM may use
    "enforce_eager": True,              # True = safer/simpler on T4; False = CUDA graphs (faster, more memory)
    "hf_batch_size": 4,                 # only for backend="hf"
    "exec_timeout_s": 30.0,             # per-query time limit for predicted SQL (official BIRD scripts use 30 s)
    "gold_timeout_s": 120.0,            # time limit for gold SQL (deviation from official: see Step 5)
    "ves_iterations": 3,                # timing repetitions for R-VES (official: 100, far too slow here; 0 = skip)
    "vllm_version": "0.18.1",           # pinned (needs torch 2.10.0, which Kaggle shipped in Aug 2026); "" = latest
    "note": "",                         # free text, e.g. "first smoke test"
}

MODEL_PRESETS = {
    # key:                 Hugging Face model id,                prompt style, max new tokens
    "qwen2.5-coder-1.5b": {"model_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "prompt_style": "omnisql", "max_new_tokens": 1024},
    "qwen2.5-coder-7b":   {"model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",   "prompt_style": "omnisql", "max_new_tokens": 1024},
    "omnisql-7b":         {"model_id": "seeklhy/OmniSQL-7B",               "prompt_style": "omnisql", "max_new_tokens": 1024},
    "arctic-r1-7b":       {"model_id": "Snowflake/Arctic-Text2SQL-R1-7B",  "prompt_style": "arctic",  "max_new_tokens": 4096},
}

# (Used only by the local test script in dev/. On Kaggle this line changes nothing.)
CFG.update(json.loads(os.environ.get("T2S_OVERRIDES", "{}")))

MODEL = MODEL_PRESETS[CFG["model_key"]]
ON_KAGGLE = Path("/kaggle").exists()
WORK_DIR = Path(os.environ.get("T2S_WORK_DIR", "/kaggle/working" if ON_KAGGLE else "./work"))
DATA_DIR = Path(os.environ.get("T2S_DATA_DIR", "/tmp/bird_minidev" if ON_KAGGLE else "./data"))
RUNS_DIR = WORK_DIR / "runs"

model_part = "oracle" if CFG["backend"] == "oracle" else CFG["model_key"]
RUN_ID = (f"{dt.datetime.now():%Y%m%d-%H%M}_{model_part}_{MODEL['prompt_style']}"
          f"_{'ev' if CFG['use_evidence'] else 'noev'}_n{CFG['n_questions']}")
RUN_DIR = RUNS_DIR / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

try:
    display  # exists inside Jupyter/Kaggle
except NameError:
    display = print
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it

def save_json(obj, path):
    # default=... converts numpy numbers (np.int64, np.float64) into plain Python numbers
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))

print("Run id :", RUN_ID)
print("Model  :", MODEL["model_id"], "| backend:", CFG["backend"], "| prompt style:", MODEL["prompt_style"])
print("Results will be written to:", RUN_DIR)

# %% [markdown]
# ---
# ## STEP 2: Check the machine and install the inference engine
#
# **What you should see on a GPU session:** two *Tesla T4* GPUs with about 15 GB each and *compute capability 7.5*.
# Two facts about the T4 shape every later setting:
# - It is an older (2018, "Turing") GPU. It has **no bfloat16 support**, so we run models in **float16** (`dtype="float16"`).
# - A 7-billion-parameter model needs about 15 GB just for its weights in float16. That does **not** fit on one T4 with room to
#   spare, so we split every layer across both GPUs. This is **tensor parallelism** (`tensor_parallel_size=2`).
#
# **vLLM** is a fast inference engine. It processes all 500 prompts at once ("continuous batching") and manages GPU memory in pages
# ("PagedAttention"). On this hardware it is roughly 10× faster than generating one question at a time with plain `transformers`.
#
# Installing vLLM takes a few minutes. **If anything fails after the install, use *Run → Restart & clear cell outputs*,
# then run the notebook again.** The install is skipped once vLLM is present.

# %%
def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()

print("Python  :", sys.version.split()[0], "|", platform.platform())
print("CPUs    :", os.cpu_count())
if Path("/proc/meminfo").exists():
    mem_kb = int(re.search(r"MemTotal:\s+(\d+)", Path("/proc/meminfo").read_text()).group(1))
    print(f"RAM     : {mem_kb / 1e6:.1f} GB")
print("Disk    :", f"{shutil.disk_usage(WORK_DIR).free / 1e9:.0f} GB free under {WORK_DIR}")
if shutil.which("nvidia-smi"):
    print(sh("nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv"))
else:
    print("GPU     : none visible (fine for backend='oracle'; for real runs pick 'GPU T4 x2' under Settings → Accelerator)")

# %%
if CFG["backend"] == "vllm":
    if importlib.util.find_spec("vllm") is None:
        spec = f"vllm=={CFG['vllm_version']}" if CFG["vllm_version"] else "vllm"
        print(f"Installing {spec} … (3-6 minutes, output hidden unless it fails)")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", spec], capture_output=True, text=True)
        print("pip exit code:", r.returncode)
        if r.returncode != 0:
            print(r.stdout[-3000:], r.stderr[-3000:])
    import vllm, torch
    print("vLLM", vllm.__version__, "| torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(),
          "| GPUs:", torch.cuda.device_count())
else:
    print("backend =", CFG["backend"], "→ vLLM not needed, skipping install")

# %% [markdown]
# ---
# ## STEP 3: Get the BIRD Mini-Dev data
#
# **BIRD** ([bird-bench.github.io](https://bird-bench.github.io/)) is the main benchmark for text-to-SQL on large, "dirty",
# real-world databases. **Mini-Dev** is a 500-question subset of BIRD's dev split, released by the BIRD team for cheaper
# experiments. It has 11 SQLite databases (about 1.4 GB), each question labelled *simple / moderate / challenging*, plus a human-written gold SQL.
#
# The official zip (about 800 MB) also contains MySQL and PostgreSQL versions. We only extract the SQLite part.
# The download goes to `/tmp`, which is fast scratch disk that is *not* saved with your notebook output.
#
# We also fetch **Arcwise-Plat-SQL**, a corrected set of gold SQL for Mini-Dev from Jin et al. (PVLDB 2026). They found that
# **about 53 % of Mini-Dev examples contain annotation errors**. We use it in Step 9 to measure how much label noise moves your score.
#
# *Tip:* if you add a Kaggle Dataset that contains the `MINIDEV` folder as an input, the cell finds it and skips the download.

# %%
MINIDEV_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip"
PLAT_URL = ("https://raw.githubusercontent.com/uiuc-kang-lab/text_to_sql_benchmarks/main/data/"
            "arcwise_plat_sql_only_with_diff.json")

def find_minidev():
    roots = [DATA_DIR] + ([Path("/kaggle/input")] if ON_KAGGLE else [])
    for root in roots:
        if root.exists():
            for f in root.rglob("mini_dev_sqlite.json"):
                if (f.parent / "dev_databases").is_dir():
                    return f.parent
    return None

def download(url, dest):
    def hook(blocks, block_size, total):
        done = blocks * block_size
        if total > 0 and blocks % 2000 == 0:
            print(f"  {done / 1e6:7.0f} / {total / 1e6:.0f} MB")
    urllib.request.urlretrieve(url, dest, reporthook=hook)

MINIDEV_DIR = find_minidev()
if MINIDEV_DIR is None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "minidev.zip"
    print("Downloading", MINIDEV_URL)
    download(MINIDEV_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if m.startswith("minidev/MINIDEV/") and "__MACOSX" not in m]
        for m in tqdm(members, desc="extracting"):
            z.extract(m, DATA_DIR)
    zip_path.unlink()
    MINIDEV_DIR = find_minidev()

PLAT_FILE = DATA_DIR / "arcwise_plat_sql.json"
if not PLAT_FILE.exists():
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        download(PLAT_URL, PLAT_FILE)
    except Exception as e:
        print("Could not download the corrected labels (optional, Step 9 will skip that part):", e)

print("Mini-Dev folder:", MINIDEV_DIR)
for p in sorted(MINIDEV_DIR.iterdir()):
    print("  ", p.name)

# %% [markdown]
# ---
# ## STEP 4: Look at the data before using it
#
# Each question has: `db_id` (which database), `question`, `evidence` (an expert hint, such as a formula or which value
# means what), `SQL` (the gold answer) and `difficulty`.
#
# ⚠️ **A quirk worth knowing:** `question_id` is **not unique** in this file (ids 137 and 138 appear twice). We therefore key everything
# by the **row position** `idx` (0-499), which is also how the official evaluation scripts align predictions and gold SQL.

# %%
import numpy as np
import pandas as pd
pd.set_option("display.max_colwidth", 120)

QUESTIONS = json.load(open(MINIDEV_DIR / "mini_dev_sqlite.json", encoding="utf-8"))
for i, q in enumerate(QUESTIONS):
    q["idx"] = i
    q["evidence"] = q.get("evidence") or ""

def db_file(db_id):
    return MINIDEV_DIR / "dev_databases" / db_id / f"{db_id}.sqlite"

df_q = pd.DataFrame(QUESTIONS)
print("Questions:", len(df_q), "| unique question_id:", df_q.question_id.nunique())
display(df_q.difficulty.value_counts().rename("questions per difficulty").to_frame())
display(df_q.db_id.value_counts().rename("questions per database").to_frame())

ex_q = QUESTIONS[1]
print("\nExample  →  database:", ex_q["db_id"], "| difficulty:", ex_q["difficulty"])
print("QUESTION:", ex_q["question"])
print("EVIDENCE:", ex_q["evidence"])
print("GOLD SQL:", ex_q["SQL"])

# %% [markdown]
# ---
# ## STEP 5: Build the evaluator, then test it
#
# This is the most important step. **If the evaluator is wrong, every number you report is wrong.**
# The code below mirrors the official Mini-Dev scripts (`evaluation_ex.py`, `evaluation_f1.py`, `evaluation_ves.py` in
# [github.com/bird-bench/mini_dev](https://github.com/bird-bench/mini_dev)) with two safety additions: databases are opened
# **read-only** (a model can't `DROP TABLE` your benchmark), and each query has a hard time limit.
#
# **The three official metrics:**
#
# | Metric | Question it answers | How it is computed |
# |---|---|---|
# | **EX**: Execution Accuracy | Did the query return the right answer? | 1 if `set(predicted rows) == set(gold rows)`, else 0. **Row order and duplicates are ignored, but column order counts** (rows are compared as tuples). |
# | **Soft-F1** | How close was it? (partial credit) | After removing duplicate rows, rows are paired by position, and cell values are compared regardless of column order: precision/recall/F1 over matched cells. Useful when the model returns an extra column or misses one. |
# | **R-VES**: Reward-based Valid Efficiency Score | Is the correct query also fast? | Only for EX = 1: time ratio τ = gold time / predicted time (averaged over repeats, outliers removed) → reward 1.25 (τ≥2), 1 (τ≥1), 0.75 (τ≥0.5), 0.5 (τ≥0.25), 0.25 otherwise. R-VES = mean(√reward) × 100. |
#
# Every metric is reported **per difficulty level** (simple / moderate / challenging / total), as in the BIRD leaderboard.
#
# **Two deliberate deviations from the official scripts.** (1) Gold queries get up to 120 s instead of 30 s. A few Mini-Dev gold queries
# (e.g. idx 340 and 393) can take about 30 s on slow or busy CPUs, and the official script would then mark those questions wrong for *every* model.
# (2) R-VES uses 3 timing repetitions instead of 100, to save time. Both are listed in `run_config.json`, so mention them in your thesis.
#
# ⚠️ R-VES measures *time*, and Kaggle machines are shared and noisy. Only compare R-VES between runs made on the same
# kind of machine, and treat small differences as noise.

# %%
def _connect_ro(path):
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, check_same_thread=False)
    return conn

def run_sql(db_path, sql, timeout_s=None):
    """Execute one query read-only with a time limit.
    Returns {"status": ok|error|timeout, "rows", "ncols", "seconds", "error"}."""
    timeout_s = CFG["exec_timeout_s"] if timeout_s is None else timeout_s
    conn = _connect_ro(db_path)
    deadline = time.perf_counter() + timeout_s
    conn.set_progress_handler(lambda: int(time.perf_counter() > deadline), 10_000)  # returning 1 aborts the query
    t0 = time.perf_counter()
    try:
        cur = conn.execute(sql)
        rows = cur.fetchall()
        ncols = len(cur.description) if cur.description else 0
        return {"status": "ok", "rows": rows, "ncols": ncols, "seconds": time.perf_counter() - t0, "error": ""}
    except sqlite3.OperationalError as e:
        status = "timeout" if "interrupted" in str(e).lower() else "error"
        return {"status": status, "rows": None, "ncols": 0, "seconds": time.perf_counter() - t0, "error": str(e)}
    except Exception as e:
        return {"status": "error", "rows": None, "ncols": 0, "seconds": time.perf_counter() - t0,
                "error": f"{type(e).__name__}: {e}"}
    finally:
        conn.close()

def ex_score(pred_rows, gold_rows):
    """Official EX: set equality of result rows."""
    if pred_rows is None or gold_rows is None:
        return 0
    return int(set(pred_rows) == set(gold_rows))

def soft_f1_score(pred_rows, gold_rows):
    """Official Mini-Dev Soft-F1 (same logic as evaluation_f1.py)."""
    if pred_rows is None or gold_rows is None:
        return 0.0
    if not pred_rows and not gold_rows:
        return 1.0
    pred = list(dict.fromkeys(pred_rows))  # drop duplicate rows, keep order
    gold = list(dict.fromkeys(gold_rows))
    tp = fp = fn = 0.0
    for i, g in enumerate(gold):
        if i >= len(pred):                 # gold row with no predicted row
            fn += 1
            continue
        p, n = pred[i], len(g)
        tp += sum(v in g for v in p) / n
        fp += sum(v not in g for v in p) / n
        fn += sum(v not in p for v in g) / n
    fp += max(0, len(pred) - len(gold))    # predicted rows with no gold row
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

def _timed_run(db_path, sql, timeout_s=None):
    # Like the official script: the timing includes opening the connection and fetching all rows.
    t0 = time.perf_counter()
    r = run_sql(db_path, sql, timeout_s)
    return time.perf_counter() - t0, r["status"]

def ves_reward(db_path, pred_sql, gold_sql, iterations):
    """Official R-VES reward for one *correct* prediction."""
    ratios = []
    for _ in range(iterations):
        tp, sp = _timed_run(db_path, pred_sql)
        tg, sg = _timed_run(db_path, gold_sql, CFG["gold_timeout_s"])
        if sp != "ok" or sg != "ok":
            return 0.0
        ratios.append(tg / max(tp, 1e-9))
    arr = np.array(ratios)
    m, s = arr.mean(), arr.std()
    kept = arr[(arr > m - 3 * s) & (arr < m + 3 * s)]  # official outlier removal (3 standard deviations)
    tau = kept.mean() if len(kept) else m
    if tau >= 2:
        return 1.25
    if tau >= 1:
        return 1.0
    if tau >= 0.5:
        return 0.75
    if tau >= 0.25:
        return 0.5
    return 0.25

def parallel_map(fn, items, desc=""):
    # sqlite releases Python's GIL while a query runs, so threads give real parallelism here.
    with ThreadPoolExecutor(max_workers=max(2, os.cpu_count() or 2)) as pool:
        return list(tqdm(pool.map(fn, items), total=len(items), desc=desc))

# %% [markdown]
# ### 5b: Sanity checks
#
# 1. **Tiny hand-made examples.** Check each metric on cases where you can work out the answer in your head.
# 2. **All 500 gold queries must run** without error or timeout. We also keep their results (`GOLD`) for scoring later, and look at how
#    long they take, since slow queries are where timeouts and R-VES noise come from.

# %%
gold_demo = [(1, "a"), (2, "b")]
cases = {
    "identical":                        [(1, "a"), (2, "b")],
    "same rows, different order":       [(2, "b"), (1, "a")],
    "one wrong cell":                   [(1, "a"), (2, "c")],
    "columns swapped":                  [("a", 1), ("b", 2)],
    "extra row":                        [(1, "a"), (2, "b"), (3, "c")],
    "empty result":                     [],
}
display(pd.DataFrame([{"prediction": k, "EX": ex_score(v, gold_demo), "Soft-F1": round(soft_f1_score(v, gold_demo), 3)}
                      for k, v in cases.items()]).set_index("prediction"))
# Two lessons from this table:
# - "columns swapped": EX = 0 (rows are compared as tuples) but Soft-F1 = 1.0 (same cells). This is what Soft-F1 was designed to reveal.
# - "same rows, different order": EX = 1 but Soft-F1 = 0! The official Soft-F1 pairs rows *by position*, so it is
#   row-order sensitive. We copy this quirk on purpose to stay comparable with the leaderboard, but mention it if you report Soft-F1.

# %%
GOLD_TIMEOUT_S = CFG["gold_timeout_s"]  # gold gets extra time: a slow *gold* query must not count against the model
gold_results = parallel_map(lambda q: run_sql(db_file(q["db_id"]), q["SQL"], GOLD_TIMEOUT_S), QUESTIONS, desc="gold SQL")
GOLD = {q["idx"]: r for q, r in zip(QUESTIONS, gold_results)}

status_counts = Counter(r["status"] for r in gold_results)
print("Gold query status:", dict(status_counts), "← should be {'ok': 500}")
secs = pd.Series([r["seconds"] for r in gold_results])
print(f"Gold query time: median {secs.median()*1000:.1f} ms | 95th pct {secs.quantile(.95):.2f} s | max {secs.max():.2f} s")
print("Gold result size: median", int(np.median([len(r['rows'] or []) for r in gold_results])), "rows")
slow = sorted(zip(secs, QUESTIONS), key=lambda t: -t[0])[:3]
for s, q in slow:
    print(f"  slowest: {s:.2f}s  idx={q['idx']}  db={q['db_id']}")
assert status_counts.get("error", 0) == 0, "Some gold queries raised errors: check the data paths before continuing!"
if status_counts.get("timeout"):
    print("⚠️ gold timeouts:", [q["idx"] for q, r in zip(QUESTIONS, gold_results) if r["status"] == "timeout"],
          "→ these will be scored 'gold_failed'. Re-run this cell; if it persists, raise GOLD_TIMEOUT_S.")

# %% [markdown]
# ---
# ## STEP 6: Turn each question into a prompt
#
# The model never sees the database itself, only a **text description of its schema** inside the prompt. How you write that
# description ("schema serialization") has a big effect on accuracy. We use the **OmniSQL format** (Li et al., 2025), which
# both OmniSQL and Arctic-Text2SQL-R1 were trained on. For each table it gives:
# - a `CREATE TABLE` statement with every column and its type,
# - a SQL comment with a readable column description (from BIRD's `dev_tables.json`) plus **2 example values** (helps with formats like `'2012/12/1'` vs `'2012-12-01'`),
# - `PRIMARY KEY` and `FOREIGN KEY` lines, so the model knows how tables join.
#
# The evidence is put in front of the question (`evidence + "\n" + question`), as OmniSQL does.
#
# **Rule of thumb from the literature:** use the prompt format a model was trained with. A fine-tuned model can lose many
# points under an unfamiliar prompt, and then you would be measuring the prompt, not the model. If you design your own prompt
# for your thesis, run it as a separate, clearly named experiment.
#
# *Not included (yet):* OmniSQL's paper setup also retrieves question-relevant database values with a BM25 index. Without it, expect our
# scores to be a few points lower than the papers' numbers. Adding value retrieval is a good experiment for later.

# %%
SQL_RESERVED = set("""ABSOLUTE ACCESS ADD AFTER ALL ALTER ANALYZE AND AS ASC AVG BEFORE BEGIN BETWEEN BIGINT BINARY BIT BLOB
BOOLEAN BOTH BREAK BY CALL CASCADE CASE CAST CHAR CHARACTER CHECK CLOB COLLATE COLUMN COMMENT COMMIT CONDITION CONNECT
CONSTRAINT CONTINUE CONVERT COUNT CREATE CROSS CURRENT CURRENT_DATE CURRENT_TIME CURRENT_TIMESTAMP CURSOR DATABASE DATE DAY
DEC DECIMAL DECLARE DEFAULT DELETE DESC DESCRIBE DISTINCT DO DOUBLE DROP EACH ELSE ELSEIF END ESCAPE EXCEPT EXEC EXECUTE
EXISTS EXIT EXPLAIN EXTEND EXTERNAL FALSE FETCH FILTER FLOAT FOR FOREIGN FROM FULL FUNCTION GENERATED GLOBAL GRANT GROUP
HAVING HOLD HOUR IDENTIFIED IF IGNORE ILIKE IN INDEX INNER INOUT INSENSITIVE INSERT INT INTEGER INTERSECT INTERVAL INTO IS
ITERATE JOIN KEY LANGUAGE LEADING LEAVE LEFT LIKE LIMIT LOCAL LOCALTIME LOCALTIMESTAMP LOCK LOOP MATCH MAX MERGE MIN MINUTE
MODIFIES MODULE MONTH NATURAL NCHAR NEW NO NOT NULL NUMERIC OF OFFSET OLD ON ONLY OPEN OR ORDER OUT OUTER OVER PARTITION
PERCENT PLACING POSITION PRECISION PRIMARY PROCEDURE RANGE READS REAL RECURSIVE REFERENCES REGEXP RELEASE RENAME REPEAT
REPLACE RESIGNAL RESTRICT RETURN REVOKE RIGHT RLIKE ROLE ROLLBACK ROW ROWNUM ROWS SAVEPOINT SCHEMA SELECT SENSITIVE SESSION
SET SIGNAL SMALLINT SOME SPECIFIC SQL SQLEXCEPTION SQLSTATE SQLWARNING START STATIC SUBSTRING SUM SYSDATE TABLE TERMINATED
THEN TIME TIMESTAMP TO TOP TRAILING TRIGGER TRUE UNDO UNION UNIQUE UNKNOWN UNLOCK UPDATE USAGE USER USING VALUE VALUES
VARCHAR VARYING VIEW WHEN WHERE WHILE WITH WITHIN YEAR ZONE""".split())

def q_ident(name):
    """Backtick-quote identifiers that are reserved words or contain spaces/special characters."""
    return f"`{name}`" if name.upper() in SQL_RESERVED or re.search(r"[^a-zA-Z0-9_]", name) else name

TABLES = {t["db_id"]: t for t in json.load(open(MINIDEV_DIR / "dev_tables.json", encoding="utf-8"))}

def sample_values(conn, table, column, k):
    sql = (f"SELECT `{column}` FROM (SELECT DISTINCT `{column}` FROM `{table}` "
           f"WHERE `{column}` IS NOT NULL AND `{column}` != '') LIMIT {k}")
    try:
        vals = [r[0] for r in conn.execute(sql).fetchall()]
    except sqlite3.Error:
        return []
    return [v[:40] if isinstance(v, str) else v for v in vals]

def schema_ddl(db_id, k=2):
    """OmniSQL-style schema: CREATE TABLE statements with column comments, example values, PKs and FKs."""
    info = TABLES[db_id]
    cols, descs, types = info["column_names_original"], info["column_names"], info["column_types"]
    pk_idx = set()
    for pk in info["primary_keys"]:
        pk_idx.update(pk if isinstance(pk, list) else [pk])
    conn = _connect_ro(db_file(db_id))
    ddls = []
    for t_idx, table in enumerate(info["table_names_original"]):
        lines, pks, fks = [], [], []
        for c_idx, ((ti, col), (_, desc), ctype) in enumerate(zip(cols, descs, types)):
            if ti != t_idx:
                continue
            vals = sample_values(conn, table, col, k) if k > 0 else []
            plain = desc.strip() == "" or col.lower() in {desc.lower(), desc.lower().replace(" ", "_"), desc.lower().replace(" ", "")}
            line = f"    {q_ident(col)} {ctype},"
            if plain:
                line += f" -- example: {vals}" if vals else ""
            else:
                line += f" -- {desc}" + (f", example: {vals}" if vals else "")
            lines.append(line)
            if c_idx in pk_idx:
                pks.append(q_ident(col))
            for src, tgt in info["foreign_keys"]:
                if src == c_idx:
                    t_tab = info["table_names_original"][cols[tgt][0]]
                    fk = (f"    CONSTRAINT fk_{table.lower().replace(' ', '_')}_{col.lower().replace(' ', '_')} "
                          f"FOREIGN KEY ({q_ident(col)}) REFERENCES {q_ident(t_tab)} ({q_ident(cols[tgt][1])}),")
                    if fk not in fks:
                        fks.append(fk)
        if pks:
            lines.append(f"    PRIMARY KEY ({', '.join(pks)}),")
        body = "\n".join(lines + fks).rstrip(",")
        ddls.append(f"CREATE TABLE {q_ident(table)} (\n{body}\n);")
    conn.close()
    return "\n\n".join(ddls)

OMNISQL_TEMPLATE = """Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
SQLite

Database Schema:
{schema}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
In your answer, please enclose the generated SQL query in a code block:
```sql
-- Your SQL query
```

Take a deep breath and think step by step to find the correct SQL query.
"""

# Arctic-Text2SQL-R1 was trained with a system prompt, a <think>/<answer> output format, and a pre-filled
# assistant start (copied from Snowflake's ArcticTraining bird_eval/infer.py).
ARCTIC_SYSTEM = ("You are a data science expert. Below, you are provided with a database schema and a natural language "
                 "question. Your task is to understand the schema and generate a valid SQL query to answer the question.")
ARCTIC_USER_TEMPLATE = """Database Engine:
SQLite

Database Schema:
{schema}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
Please provide a detailed chain-of-thought reasoning process and include your thought process within `<think>` tags. Your final answer should be enclosed within `<answer>` tags.

Ensure that your SQL query follows the correct syntax and is formatted as follows:

```sql
-- Your SQL query here
```

Example format:
<think> Step-by-step reasoning, including self-reflection and corrections if necessary. [Limited by 4K tokens] </think>
<answer> Summary of the thought process leading to the final SQL query. [Limited by 1K tokens]

```sql
Correct SQL query here
```
</answer>"""
ARCTIC_ASSISTANT_PREFIX = "Let me solve this step by step. \n<think>"

SCHEMAS = {db_id: schema_ddl(db_id, CFG["values_per_column"]) for db_id in tqdm(sorted(TABLES), desc="schemas")}

def build_prompt(q):
    """Returns (chat messages, text to pre-fill the assistant answer with)."""
    question = q["question"]
    if CFG["use_evidence"] and q["evidence"].strip():
        question = q["evidence"] + "\n" + question
    schema = SCHEMAS[q["db_id"]]
    if MODEL["prompt_style"] == "omnisql":
        return [{"role": "user", "content": OMNISQL_TEMPLATE.format(schema=schema, question=question)}], ""
    if MODEL["prompt_style"] == "arctic":
        return ([{"role": "system", "content": ARCTIC_SYSTEM},
                 {"role": "user", "content": ARCTIC_USER_TEMPLATE.format(schema=schema, question=question)}],
                ARCTIC_ASSISTANT_PREFIX)
    raise ValueError(MODEL["prompt_style"])

# Which questions this run uses: the first n, or a fixed stratified sample when n < 500 (same seed → same questions).
if CFG["n_questions"] >= len(QUESTIONS):
    SUBSET = list(QUESTIONS)
else:
    rng = random.Random(CFG["seed"])
    SUBSET = sorted(rng.sample(QUESTIONS, CFG["n_questions"]), key=lambda q: q["idx"])
PROMPTS = {q["idx"]: build_prompt(q) for q in SUBSET}

chars = pd.Series({db: len(s) for db, s in SCHEMAS.items()}).sort_values()
print(f"{len(SUBSET)} prompts built. Schema length per database (characters; ≈ 3.5 characters per token):")
print(chars.to_string())
print("\n================ EXAMPLE PROMPT (idx", SUBSET[0]["idx"], ") ================")
print(PROMPTS[SUBSET[0]["idx"]][0][-1]["content"][:3000], "\n[... truncated for display ...]")

# %% [markdown]
# ---
# ## STEP 7: Generate SQL with the model
#
# What happens in the next cell (for `backend="vllm"`):
# 1. vLLM downloads the model from Hugging Face (a 7B model is about 15 GB, which takes a few minutes) and splits it over both T4s.
# 2. Each prompt is wrapped in the model's **chat template** (the special tokens that mark *user* and *assistant* turns).
# 3. All prompts are generated in one batched call with greedy decoding.
# 4. The raw outputs are written to `generations.jsonl` **immediately**, before any scoring, so a crash later never costs you the GPU time.
#
# We record per question: prompt length, output length (tokens) and whether the output was cut off (`finish_reason="length"`).
# For the whole run we record wall-clock time and throughput. Efficiency matters for a practical thesis, not only accuracy.
#
# **Reading the vLLM log:** you should see the model loading on 2 workers (TP rank 0 and 1), a line with the KV-cache size,
# then a progress bar over the prompts. *Out of memory?* Lower `max_model_len` (e.g. 8192) or `gpu_memory_utilization` (e.g. 0.85).

# %%
def chat_text(tokenizer, messages, assistant_prefix):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + assistant_prefix

def generate_oracle(subset):
    # No model: "predict" the gold SQL. Tests extraction + execution + scoring end-to-end.
    return {q["idx"]: {"raw_output": f"```sql\n{q['SQL']}\n```", "finish_reason": "stop",
                       "prompt_tokens": 0, "output_tokens": 0, "status": "ok"} for q in subset}, {}

def generate_vllm(subset):
    import torch
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL["model_id"], dtype="float16", tensor_parallel_size=max(1, torch.cuda.device_count()),
              max_model_len=CFG["max_model_len"], gpu_memory_utilization=CFG["gpu_memory_utilization"],
              enforce_eager=CFG["enforce_eager"], disable_custom_all_reduce=True, seed=CFG["seed"],
              trust_remote_code=True)
    tok = llm.get_tokenizer()
    gens, todo_texts, todo_params, todo_idx = {}, [], [], []
    for q in subset:
        text = chat_text(tok, *PROMPTS[q["idx"]])
        n_prompt = len(tok(text, add_special_tokens=False).input_ids)
        budget = min(MODEL["max_new_tokens"], CFG["max_model_len"] - n_prompt - 8)
        if budget < 128:  # the prompt alone (almost) fills the context window
            gens[q["idx"]] = {"raw_output": "", "finish_reason": "prompt_too_long", "prompt_tokens": n_prompt,
                              "output_tokens": 0, "status": "prompt_too_long"}
            continue
        todo_texts.append(text)
        todo_idx.append(q["idx"])
        todo_params.append(SamplingParams(temperature=CFG["temperature"], max_tokens=budget, seed=CFG["seed"]))
    t0 = time.time()
    outputs = llm.generate(todo_texts, todo_params)
    seconds = time.time() - t0
    for i, out in zip(todo_idx, outputs):
        o = out.outputs[0]
        gens[i] = {"raw_output": o.text, "finish_reason": o.finish_reason, "prompt_tokens": len(out.prompt_token_ids),
                   "output_tokens": len(o.token_ids), "status": "ok"}
    return gens, {"generation_seconds": seconds}

def generate_hf(subset):
    import torch, transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL["model_id"])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype_kw = "dtype" if tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 56) else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(MODEL["model_id"], device_map="auto", **{dtype_kw: torch.float16})
    model.eval()
    items = sorted(((q["idx"], chat_text(tok, *PROMPTS[q["idx"]])) for q in subset), key=lambda t: -len(t[1]))
    gens, bs = {}, CFG["hf_batch_size"]
    t0 = time.time()
    for b in tqdm(range(0, len(items), bs), desc="generating"):
        batch = items[b:b + bs]
        enc = tok([t for _, t in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MODEL["max_new_tokens"], pad_token_id=tok.pad_token_id,
                                 do_sample=CFG["temperature"] > 0,
                                 **({"temperature": CFG["temperature"]} if CFG["temperature"] > 0 else {}))
        for (i, _), row, mask in zip(batch, out[:, enc.input_ids.shape[1]:], enc.attention_mask):
            ids = [t for t in row.tolist() if t != tok.pad_token_id]
            gens[i] = {"raw_output": tok.decode(ids, skip_special_tokens=True),
                       "finish_reason": "length" if len(ids) >= MODEL["max_new_tokens"] else "stop",
                       "prompt_tokens": int(mask.sum()), "output_tokens": len(ids), "status": "ok"}
    return gens, {"generation_seconds": time.time() - t0}

def environment_info():
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "on_kaggle": ON_KAGGLE}
    for mod in ("torch", "vllm", "transformers"):
        if importlib.util.find_spec(mod):
            info[mod] = getattr(__import__(mod), "__version__", "?")
    if shutil.which("nvidia-smi"):
        info["gpus"] = sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")
    if CFG["backend"] != "oracle":
        try:
            from huggingface_hub import model_info
            info["model_revision"] = model_info(MODEL["model_id"]).sha  # exact weights version, for reproducibility
        except Exception as e:
            info["model_revision"] = f"unknown ({type(e).__name__})"
    return info

GENERATE = {"oracle": generate_oracle, "vllm": generate_vllm, "hf": generate_hf}[CFG["backend"]]
started = dt.datetime.now().isoformat(timespec="seconds")
GENS, GEN_STATS = GENERATE(SUBSET)
RUN_META = {"run_id": RUN_ID, "started": started, "config": CFG, "model": MODEL, "environment": environment_info(),
            "n_questions": len(SUBSET), **GEN_STATS}
save_json(RUN_META, RUN_DIR / "run_config.json")
with open(RUN_DIR / "generations.jsonl", "w", encoding="utf-8") as f:
    for q in SUBSET:
        f.write(json.dumps({"idx": q["idx"], **GENS[q["idx"]]}, ensure_ascii=False) + "\n")

print(f"Generated {len(GENS)} outputs → {RUN_DIR / 'generations.jsonl'}")
if GEN_STATS:
    out_tok = sum(g["output_tokens"] for g in GENS.values())
    s = GEN_STATS["generation_seconds"]
    print(f"Generation time {s/60:.1f} min | {len(GENS)/(s/60):.1f} questions/min | {out_tok/s:.0f} output tokens/s")
print("finish reasons:", dict(Counter(g["finish_reason"] for g in GENS.values())))
print("\n--- raw output of the first question ---\n", GENS[SUBSET[0]["idx"]]["raw_output"][:2000])

# %% [markdown]
# ---
# ## STEP 8: Score every prediction
#
# For each question we:
# 1. **Extract the SQL** from the model's text: the *last* ```` ```sql ```` code block (models often draft a query, then fix it). If there is no code block but the text itself is SQL, we use the text.
# 2. **Execute** it read-only with the time limit and compare it with the gold result → EX and Soft-F1.
# 3. **Label the outcome** with an automatic error category, so you can see *why* it failed:
#
# | Category | Meaning | Typical cause |
# |---|---|---|
# | `correct` | EX = 1 | |
# | `no_sql_found` / `no_sql_truncated` | no SQL could be extracted (truncated = hit the token limit) | rambling, wrong output format |
# | `prompt_too_long` | prompt did not fit in `max_model_len` | huge schema |
# | `syntax_error` | SQLite could not parse it | broken SQL |
# | `schema_error` | `no such column/table`, `ambiguous column` | **schema-linking** hallucination |
# | `function_error` | `no such function` | MySQL/PostgreSQL habits (e.g. `YEAR()`, `DATEDIFF`) in SQLite |
# | `timeout` | exceeded the time limit | missing join condition → huge cross product |
# | `wrong_empty_result` | ran, returned nothing | wrong filter value / format |
# | `wrong_column_count` | ran, different number of columns | selected too much or too little |
# | `wrong_row_count` | right columns, different number of rows | missing `DISTINCT`, `LIMIT`, `GROUP BY` |
# | `wrong_values` | same shape, different values | wrong logic, aggregation, or join |
#
# These categories are *automatic* and only coarse. Step 9 also draws a random sample of failures for **manual** analysis,
# which is what papers report (e.g. DIN-SQL's error study, NL2SQL-Bugs' taxonomy).
#
# Finally, R-VES timing runs **one query at a time**, because parallel queries would slow each other down and distort the timings.

# %%
def extract_sql(text):
    blocks = re.findall(r"```(?:sql|sqlite)?[ \t]*\n?(.*?)```", text or "", flags=re.S | re.I)
    blocks = [b.strip() for b in blocks if b.strip()]
    if blocks:
        return blocks[-1], "code_block"
    t = (text or "").strip()
    if re.match(r"^(WITH|SELECT)\b", t, flags=re.I):
        return t, "raw_text"
    return "", "none"

def categorize(gen, pred_sql, pred, gold, ex):
    if gold["status"] != "ok":
        return "gold_failed"
    if ex:
        return "correct"
    if gen["status"] == "prompt_too_long":
        return "prompt_too_long"
    if not pred_sql:
        return "no_sql_truncated" if gen["finish_reason"] == "length" else "no_sql_found"
    if pred["status"] == "timeout":
        return "timeout"
    if pred["status"] == "error":
        msg = pred["error"].lower()
        if any(s in msg for s in ("no such column", "no such table", "ambiguous column")):
            return "schema_error"
        if "no such function" in msg:
            return "function_error"
        if any(s in msg for s in ("syntax error", "incomplete input", "unrecognized token")):
            return "syntax_error"
        return "other_exec_error"
    pred_set, gold_set = set(pred["rows"]), set(gold["rows"])
    if not pred_set and gold_set:
        return "wrong_empty_result"
    if pred["ncols"] != gold["ncols"]:
        return "wrong_column_count"
    if len(pred_set) != len(gold_set):
        return "wrong_row_count"
    return "wrong_values"

def sql_features(sql):
    """Simple complexity features of the gold SQL, used to break accuracy down by query type."""
    s = re.sub(r"'[^']*'", "''", sql).upper()  # ignore string literals
    return {
        "n_joins": len(re.findall(r"\bJOIN\b", s)),
        "has_join": bool(re.search(r"\bJOIN\b", s)),
        "has_subquery": len(re.findall(r"\bSELECT\b", s)) > 1,
        "has_aggregate": bool(re.search(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", s)),
        "has_group_by": "GROUP BY" in s,
        "has_order_by": "ORDER BY" in s,
        "has_case_iif": bool(re.search(r"\b(CASE|IIF)\b", s)),
        "has_cast": bool(re.search(r"\bCAST\s*\(", s)),
        "has_distinct": "DISTINCT" in s,
    }

def preview(rows, k=3):
    return "" if rows is None else repr(rows[:k])[:300]

PRED_ROWS = {}  # kept in memory for Step 9's corrected-label analysis

def score_one(q):
    i, gen = q["idx"], GENS[q["idx"]]
    gold = GOLD.get(i) or run_sql(db_file(q["db_id"]), q["SQL"], GOLD_TIMEOUT_S)
    pred_sql, how = extract_sql(gen["raw_output"]) if gen["status"] == "ok" else ("", "none")
    pred = run_sql(db_file(q["db_id"]), pred_sql) if pred_sql else {"status": "no_sql", "rows": None, "ncols": 0,
                                                                     "seconds": 0.0, "error": ""}
    ex = ex_score(pred["rows"], gold["rows"])
    PRED_ROWS[i] = pred["rows"]
    return {
        "idx": i, "question_id": q["question_id"], "db_id": q["db_id"], "difficulty": q["difficulty"],
        "question": q["question"], "evidence": q["evidence"], "gold_sql": q["SQL"], "pred_sql": pred_sql,
        "extraction": how, "finish_reason": gen["finish_reason"], "prompt_tokens": gen["prompt_tokens"],
        "output_tokens": gen["output_tokens"], "pred_status": pred["status"], "pred_error": pred["error"][:300],
        "pred_seconds": round(pred["seconds"], 4), "gold_seconds": round(gold["seconds"], 4),
        "pred_n_rows": None if pred["rows"] is None else len(pred["rows"]), "gold_n_rows": len(gold["rows"] or []),
        "pred_ncols": pred["ncols"], "gold_ncols": gold["ncols"],
        "pred_preview": preview(pred["rows"]), "gold_preview": preview(gold["rows"]),
        "ex": ex, "soft_f1": round(soft_f1_score(pred["rows"], gold["rows"]), 4),
        "category": categorize(gen, pred_sql, pred, gold, ex),
        **sql_features(q["SQL"]),
    }

records = parallel_map(score_one, SUBSET, desc="scoring")
df = pd.DataFrame(records).sort_values("idx").reset_index(drop=True)

# R-VES: time correct predictions against gold, sequentially.
df["ves_reward"] = 0.0
if CFG["ves_iterations"] > 0:
    for row in tqdm(list(df[df.ex == 1].itertuples()), desc="R-VES timing"):
        df.loc[row.Index, "ves_reward"] = ves_reward(db_file(row.db_id), row.pred_sql, row.gold_sql, CFG["ves_iterations"])
df["rves_points"] = np.sqrt(df["ves_reward"]) * 100

print(f"EX = {df.ex.mean()*100:.2f}%  |  Soft-F1 = {df.soft_f1.mean()*100:.2f}  |  R-VES = {df.rves_points.mean():.2f}"
      f"  (n = {len(df)})")
display(df.category.value_counts().to_frame("questions"))

# %% [markdown]
# ---
# ## STEP 9: Report this run
#
# ### 9a: The headline table, with uncertainty
#
# The first table has the same layout as the official BIRD output, so you can compare with the leaderboard and papers.
#
# **Why the "±" matters:** 500 questions is a *sample*. If you re-ran on 500 *other* similar questions, the score would move.
# The 95 % confidence interval (CI) tells you how much. Two details follow Miller (2024), *"Adding Error Bars to Evals"*:
# - The naive standard error is √(p(1−p)/n). With n = 500 and EX ≈ 50 %, that is about ±4.4 points (95 %).
# - Questions about the **same database are not independent** (if a model misunderstands the `financial` schema, it fails many
#   of those questions together). The **clustered** standard error (clustered by `db_id`) accounts for this and is usually larger.
#   We report the clustered interval. With only 11 databases it is a rough estimate, but it is the honest one.
#
# **Practical consequence:** if two models differ by 2 points on Mini-Dev, that alone is *not* evidence that one is better.
# Step 10 shows the proper paired test.

# %%
LEVELS = ["simple", "moderate", "challenging", "total"]

def clustered_ci(values, clusters):
    """Mean, naive SE, and SE clustered by `clusters` (Miller 2024, Eq. for clustered standard errors)."""
    x = np.asarray(values, dtype=float)
    n, m = len(x), x.mean()
    se_naive = x.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    resid = pd.Series(x - m).groupby(np.asarray(clusters)).sum()
    se_clu = math.sqrt((resid ** 2).sum()) / n
    return m, se_naive, se_clu

def headline_table(d):
    rows = {}
    for lvl in LEVELS:
        s = d if lvl == "total" else d[d.difficulty == lvl]
        rows[lvl] = {"count": len(s), "EX": s.ex.mean() * 100, "Soft-F1": s.soft_f1.mean() * 100,
                     "R-VES": s.rves_points.mean()}
    return pd.DataFrame(rows).T[["count", "EX", "Soft-F1", "R-VES"]]

HEADLINE = headline_table(df)
display(HEADLINE.round(2))

ex_m, ex_se_naive, ex_se_clu = clustered_ci(df.ex, df.db_id)
print(f"EX = {ex_m*100:.1f}%  95% CI (clustered by database): [{(ex_m-1.96*ex_se_clu)*100:.1f}, {(ex_m+1.96*ex_se_clu)*100:.1f}]"
      f"   (naive CI would be ±{1.96*ex_se_naive*100:.1f})")
print(f"Output tokens per question: mean {df.output_tokens.mean():.0f} | truncated outputs: {(df.finish_reason == 'length').sum()}"
      f" | prompt tokens mean {df.prompt_tokens.mean():.0f}, max {df.prompt_tokens.max()}")

# %% [markdown]
# ### 9b: Where does it fail? Breakdowns
#
# Three views that papers commonly report (BIRD, NL2SQL360, the NL2SQL survey by Liu et al. 2024):
# accuracy **per database (domain)**, the **distribution of failure types**, and accuracy **by SQL complexity**
# (does the model struggle with joins? with nested queries?). Read the charts with the counts in mind: a database with
# 30 questions has a wide error bar of its own.

# %%
import matplotlib
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2, "text.color": INK,
                     "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left"})
FIG_DIR = RUN_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

def tidy(ax, xlabel=""):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_xlabel(xlabel)

def hbar(ax, labels, values, color, fmt="{:.0f}", xmax=None):
    y = np.arange(len(labels))
    ax.barh(y, values, height=0.6, color=color)
    ax.set_yticks(y, labels)
    for yi, v in zip(y, values):
        ax.text(v, yi, " " + fmt.format(v), va="center", fontsize=9, color=INK2)
    if xmax:
        ax.set_xlim(0, xmax)

per_db = df.groupby("db_id").agg(EX=("ex", "mean"), n=("ex", "size")).sort_values("EX")
cats = df.category.value_counts().sort_values()
feat_cols = ["has_join", "has_subquery", "has_aggregate", "has_group_by", "has_order_by", "has_case_iif", "has_cast"]
feat = pd.DataFrame({f: {"with": df[df[f]].ex.mean() * 100 if df[f].any() else np.nan,
                         "without": df[~df[f]].ex.mean() * 100 if (~df[f]).any() else np.nan,
                         "n_with": int(df[f].sum())} for f in feat_cols}).T

fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"wspace": 0.55})
hbar(axes[0], [f"{d} (n={n})" for d, n in zip(per_db.index, per_db.n)], per_db.EX * 100, SERIES[0], "{:.0f}%", 110)
axes[0].set_title("EX by database")
tidy(axes[0], "EX (%)")
hbar(axes[1], list(cats.index), cats.values, SERIES[0])
axes[1].set_title("Outcome categories")
tidy(axes[1], "questions")
y = np.arange(len(feat))
axes[2].barh(y + 0.18, feat["with"], height=0.34, color=SERIES[0], label="gold SQL has feature")
axes[2].barh(y - 0.18, feat["without"], height=0.34, color=SERIES[1], label="gold SQL lacks feature")
axes[2].set_yticks(y, [f"{f} (n={int(n)})" for f, n in zip(feat.index, feat.n_with)])
for yi, (w, wo) in enumerate(zip(feat["with"], feat["without"])):
    for off, v in ((0.18, w), (-0.18, wo)):
        if not np.isnan(v):
            axes[2].text(v, yi + off, f" {v:.0f}%", va="center", fontsize=8, color=INK2)
axes[2].set_xlim(0, 110)
axes[2].legend(loc="lower right", frameon=False, fontsize=8)
axes[2].set_title("EX by SQL feature")
tidy(axes[2], "EX (%)")
fig.suptitle(f"{RUN_ID}", x=0.01, ha="left", fontsize=9, color=MUTED)
fig.savefig(FIG_DIR / "breakdowns.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 9c: How much does label noise move the score?
#
# Jin et al. (PVLDB 2026) audited Mini-Dev and released corrected gold SQL (**Arcwise-Plat-SQL**). Some corrections also reword the
# question or evidence. Those items are no longer the same task, so we use only the items where **question and evidence
# are unchanged**. For those, we re-score *your same predictions* against the original and the corrected gold SQL.
# The difference is pure label noise, not model behaviour. This makes a strong paragraph for the "threats to validity" section of a thesis.

# %%
CORRECTED = None
if PLAT_FILE.exists():
    plat = {str(p["question_id"]): p for p in json.load(open(PLAT_FILE, encoding="utf-8"))}
    norm = lambda s: (s or "").strip()
    rows_c = []
    for q in SUBSET:
        p = plat.get(str(q["question_id"]))
        if not p or norm(p["question"]) != norm(q["question"]) or norm(p.get("evidence")) != norm(q["evidence"]):
            continue
        new_gold = run_sql(db_file(q["db_id"]), p["SQL"], GOLD_TIMEOUT_S)
        if new_gold["status"] != "ok":
            continue
        rows_c.append({"idx": q["idx"], "ex_original": int(df.loc[df.idx == q["idx"], "ex"].iloc[0]),
                       "ex_corrected": ex_score(PRED_ROWS.get(q["idx"]), new_gold["rows"]),
                       "gold_changed": norm(p["SQL"]) != norm(q["SQL"])})
    CORRECTED = pd.DataFrame(rows_c)
    if len(CORRECTED):
        print(f"Items with unchanged question+evidence: {len(CORRECTED)} (of which gold SQL corrected: {CORRECTED.gold_changed.sum()})")
        print(f"EX on these items with ORIGINAL gold : {CORRECTED.ex_original.mean()*100:.1f}%")
        print(f"EX on these items with CORRECTED gold: {CORRECTED.ex_corrected.mean()*100:.1f}%")
        print("Outcome flips (original → corrected):",
              dict(Counter(zip(CORRECTED.ex_original, CORRECTED.ex_corrected))))
else:
    print("Corrected labels not available; skipping.")

# %% [markdown]
# ### 9d: Save everything
#
# Files written to `runs/<run_id>/`:
#
# | File | What it is | Use it for |
# |---|---|---|
# | `run_config.json` | all settings + software versions + GPU + exact model revision | reproducibility (thesis appendix) |
# | `generations.jsonl` | raw model outputs | re-scoring without re-generating |
# | `per_question.csv` | one row per question: SQL, outcome, category, tokens, features | all analysis |
# | `summary.json` | headline metrics, CIs, category counts, timing | comparing runs |
# | `predict_official_format.json` | predictions in the official BIRD format | cross-checking with the official scripts |
# | `manual_review.csv` | 30 random failures with empty columns to fill in | your manual error analysis |
# | `figures/*.png` | charts | the thesis |
#
# A one-line summary is also appended to `runs/leaderboard.csv`.

# %%
df.to_csv(RUN_DIR / "per_question.csv", index=False)

official = {str(r.idx): f"{r.pred_sql}\t----- bird -----\t{r.db_id}" for r in df.itertuples()}
save_json(official, RUN_DIR / "predict_official_format.json")

SUMMARY = {
    "run_id": RUN_ID, "model_key": CFG["model_key"] if CFG["backend"] != "oracle" else "oracle",
    "model_id": MODEL["model_id"] if CFG["backend"] != "oracle" else "gold SQL", "backend": CFG["backend"],
    "prompt_style": MODEL["prompt_style"], "use_evidence": CFG["use_evidence"], "n": len(df),
    **{f"EX_{l}": round(HEADLINE.loc[l, "EX"], 2) for l in LEVELS},
    **{f"SoftF1_{l}": round(HEADLINE.loc[l, "Soft-F1"], 2) for l in LEVELS},
    **{f"RVES_{l}": round(HEADLINE.loc[l, "R-VES"], 2) for l in LEVELS},
    "EX_ci95_clustered": [round((ex_m - 1.96 * ex_se_clu) * 100, 2), round((ex_m + 1.96 * ex_se_clu) * 100, 2)],
    "EX_se_naive": round(ex_se_naive * 100, 2), "EX_se_clustered": round(ex_se_clu * 100, 2),
    "EX_corrected_subset": (round(CORRECTED.ex_corrected.mean() * 100, 2) if CORRECTED is not None and len(CORRECTED) else None),
    "EX_original_same_subset": (round(CORRECTED.ex_original.mean() * 100, 2) if CORRECTED is not None and len(CORRECTED) else None),
    "corrected_subset_n": (len(CORRECTED) if CORRECTED is not None else 0),
    "categories": df.category.value_counts().to_dict(),
    "mean_prompt_tokens": round(df.prompt_tokens.mean(), 1), "mean_output_tokens": round(df.output_tokens.mean(), 1),
    "truncated_outputs": int((df.finish_reason == "length").sum()),
    "generation_minutes": round(GEN_STATS.get("generation_seconds", 0) / 60, 2),
    "ves_iterations": CFG["ves_iterations"], "note": CFG["note"], "finished": dt.datetime.now().isoformat(timespec="seconds"),
}
save_json(SUMMARY, RUN_DIR / "summary.json")

lb_path = RUNS_DIR / "leaderboard.csv"
lb_row = pd.DataFrame([{k: v for k, v in SUMMARY.items() if not isinstance(v, (dict, list))}])
lb_row.to_csv(lb_path, mode="a", header=not lb_path.exists(), index=False)

wrong = df[df.ex == 0]
review = wrong.sample(min(30, len(wrong)), random_state=CFG["seed"]) if len(wrong) else wrong
review = review[["idx", "db_id", "difficulty", "question", "evidence", "gold_sql", "pred_sql", "category",
                 "gold_preview", "pred_preview"]].copy()
review["error_type"] = ""      # fill in: schema_linking | join | filter_value | aggregation_grouping | ordering_limit |
                               #          nested_logic | projection | evidence_misuse | dialect | ambiguous_question
review["gold_is_wrong"] = ""   # fill in: yes / no / unsure (about half of Mini-Dev labels have issues!)
review["notes"] = ""
review.to_csv(RUN_DIR / "manual_review.csv", index=False)

print("Saved:", *sorted(p.name for p in RUN_DIR.iterdir()), sep="\n  ")

# %% [markdown]
# ---
# ## STEP 10: Compare runs
#
# A single number means little on its own; the thesis is about **comparisons** (model A vs model B, with vs without evidence, ...).
# This cell looks for every finished run it can see: runs from this session in `/kaggle/working/runs`, plus any earlier run folders
# you attached as **Input** (a Kaggle Dataset or a previous notebook version's output).
#
# **The right test for "is A better than B?"** Both models answered the *same* 500 questions, so compare them **question by question**
# (a *paired* comparison). Only questions where exactly one of them is correct carry information:
#
# |  | B correct | B wrong |
# |---|---|---|
# | **A correct** | both (no information) | **A only** = b |
# | **A wrong** | **B only** = c | neither (no information) |
#
# **McNemar's exact test** asks: if A and B were equally good, how surprising is this split of b vs c? A p-value below 0.05 is the usual
# threshold. With many comparisons, be stricter (e.g. Bonferroni: divide 0.05 by the number of comparisons).

# %%
def find_runs():
    roots = [RUNS_DIR] + ([Path("/kaggle/input")] if ON_KAGGLE else [])
    found = {}
    for root in roots:
        for s in root.rglob("summary.json") if root.exists() else []:
            if (s.parent / "per_question.csv").exists():
                summ = json.load(open(s))
                found[summ["run_id"]] = s.parent
    return found

def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)

def compare(run_a, run_b, runs):
    a = pd.read_csv(runs[run_a] / "per_question.csv")[["idx", "ex", "db_id", "difficulty"]]
    b = pd.read_csv(runs[run_b] / "per_question.csv")[["idx", "ex"]]
    m = a.merge(b, on="idx", suffixes=("_a", "_b"))
    only_a = int(((m.ex_a == 1) & (m.ex_b == 0)).sum())
    only_b = int(((m.ex_a == 0) & (m.ex_b == 1)).sum())
    diff, _, se = clustered_ci(m.ex_a - m.ex_b, m.db_id)
    print(f"A = {run_a}\nB = {run_b}\n  shared questions: {len(m)}")
    print(f"  EX A = {m.ex_a.mean()*100:.1f}% | EX B = {m.ex_b.mean()*100:.1f}% | difference A−B = {diff*100:+.1f} points "
          f"(95% CI {(diff-1.96*se)*100:+.1f} … {(diff+1.96*se)*100:+.1f}, clustered)")
    print(f"  both correct {int(((m.ex_a==1)&(m.ex_b==1)).sum())} | A only {only_a} | B only {only_b} | "
          f"neither {int(((m.ex_a==0)&(m.ex_b==0)).sum())}")
    print(f"  McNemar exact p-value = {mcnemar_exact(only_a, only_b):.4f}")

RUNS = find_runs()
LEADERBOARD = pd.DataFrame([json.load(open(p / "summary.json")) for p in RUNS.values()])
cols = ["run_id", "model_key", "prompt_style", "use_evidence", "n", "EX_simple", "EX_moderate", "EX_challenging",
        "EX_total", "EX_ci95_clustered", "SoftF1_total", "RVES_total", "mean_output_tokens", "generation_minutes"]
display(LEADERBOARD[cols].sort_values("EX_total", ascending=False).reset_index(drop=True))

real = LEADERBOARD[(LEADERBOARD.backend != "oracle") & (LEADERBOARD.n == LEADERBOARD.n.max())]
if len(real) >= 2:
    a_id, b_id = real.sort_values("EX_total", ascending=False).run_id.iloc[:2]  # change these to compare any two runs
    compare(a_id, b_id, RUNS)
else:
    print("Need at least two non-oracle runs with the same n to compare. Come back after your second model run.")

# %%
# EX by difficulty for the (up to 4) most recent runs.
show = LEADERBOARD.sort_values("finished").tail(4)
if len(show):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    width = 0.8 / len(show)
    x = np.arange(len(LEVELS))
    for k, (_, r) in enumerate(show.iterrows()):
        vals = [r[f"EX_{l}"] for l in LEVELS]
        ax.bar(x + (k - (len(show) - 1) / 2) * width, vals, width=width * 0.92, color=SERIES[k], label=r["run_id"])
        for xi, v in zip(x, vals):
            ax.text(xi + (k - (len(show) - 1) / 2) * width, v + 1, f"{v:.0f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, LEVELS)
    ax.set_ylim(0, 105)
    ax.set_ylabel("EX (%)")
    ax.set_title("EX by difficulty")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.savefig(RUNS_DIR / "compare_by_difficulty.png", dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ---
# ## STEP 11: Keep your results
#
# Everything in `/kaggle/working` disappears when an **interactive** session ends. Three ways to keep it:
# 1. **Recommended for real runs:** set `CFG`, then click **Save Version → "Save & Run All (Commit)"**. Kaggle runs the whole notebook
#    in the background (you can close the browser; the limit is 12 h). The finished version keeps `/kaggle/working` as its **Output**, and
#    the version history records exactly which code and config produced which numbers.
# 2. **Download** the zip created below (Output panel on the right → `…` → Download) and store it in your thesis folder under `results/`.
# 3. To compare against earlier runs in a new version: **Add Input** → *Your Work* → pick this notebook's earlier version or a dataset
#    with your run folders. Step 10 finds them automatically.

# %%
archive = shutil.make_archive(str(WORK_DIR / RUN_ID), "zip", root_dir=RUNS_DIR, base_dir=RUN_ID)
print("Archive:", archive)
print("\nRun finished. Headline:", f"EX {SUMMARY['EX_total']:.2f}% | Soft-F1 {SUMMARY['SoftF1_total']:.2f} | "
      f"R-VES {SUMMARY['RVES_total']:.2f} | n = {SUMMARY['n']}")
