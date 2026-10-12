"""Run every CPU-only part of the notebook locally, on the real BIRD Mini-Dev data.

Usage:
    py dev/test_local.py <folder that contains minidev/MINIDEV> <work dir>

What it checks:
  1. "oracle" run end-to-end (gold SQL as predictions)  -> EX and Soft-F1 must be exactly 100
  2. two "fake model" runs with deliberately broken SQL -> every error category is detected,
     run comparison (Step 10) works
  3. SQL extraction on tricky model outputs
Optional (set environment variables):
  T2S_TOKENIZER_JSON=<tokenizer.json>   -> report prompt lengths in tokens
  T2S_OFFICIAL_EVAL_DIR=<dir>           -> re-score fake run A with the official BIRD scripts and compare
"""
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebook import SRC, parse_cells  # noqa: E402

data_root, work_dir = Path(sys.argv[1]), Path(sys.argv[2])
os.environ.update({"T2S_DATA_DIR": str(data_root), "T2S_WORK_DIR": str(work_dir), "MPLBACKEND": "Agg",
                   "T2S_OVERRIDES": json.dumps({"backend": "oracle", "ves_iterations": 2})})

cells = [src for kind, src in parse_cells(SRC.read_text(encoding="utf-8")) if kind == "code"]
ns = {"__name__": "__main__"}


def run_cells(start=0):
    for i in range(start, len(cells)):
        t0 = time.time()
        exec(compile(cells[i], f"<cell {i}>", "exec"), ns)
        print(f"--- cell {i} done in {time.time() - t0:.1f}s\n")


# 1. Oracle run ---------------------------------------------------------------------------------------------
run_cells()
s = ns["SUMMARY"]
assert s["n"] == 500, s["n"]
assert s["EX_total"] == 100.0 and s["SoftF1_total"] == 100.0, (s["EX_total"], s["SoftF1_total"])
print("ORACLE OK:", {k: s[k] for k in ("EX_total", "SoftF1_total", "RVES_total", "EX_corrected_subset",
                                       "EX_original_same_subset", "corrected_subset_n")})

# 3. Extraction cases ---------------------------------------------------------------------------------------
extract_sql = ns["extract_sql"]
cases = {
    "draft then final\n```sql\nSELECT 1;\n```\nfix:\n```sql\nSELECT 2\n```": "SELECT 2",
    "```\nSELECT 3\n```": "SELECT 3",
    "SELECT 4 FROM t": "SELECT 4 FROM t",
    "<think>```sql\nSELECT a\n```</think><answer>ok\n```sql\nSELECT b\n```</answer>": "SELECT b",
    "```SQL\nselect 5\n```": "select 5",
    "I cannot answer that.": "",
    "```sql\nSELECT unfinished": "",
}
for text, want in cases.items():
    got = extract_sql(text)[0]
    assert got == want, (text, got, want)
print("EXTRACTION OK")

# 2. Fake model runs ----------------------------------------------------------------------------------------
step8 = next(i for i, c in enumerate(cells) if "def extract_sql" in c)
QUESTIONS = ns["QUESTIONS"]
MUTATIONS = {
    "bad_column": lambda s: re.sub(r"^SELECT\s+", "SELECT nonexistent_col, ", s, count=1, flags=re.I),
    "mysql_function": lambda s: "SELECT YEAR('2020-01-01')",
    "syntax": lambda s: s + " WHERE WHERE",
    "empty": lambda s: f"SELECT * FROM ({s}) WHERE 1 = 0",
    "extra_column": lambda s: f"SELECT *, 1 FROM ({s})",
    "limit_1": lambda s: f"SELECT * FROM ({s}) LIMIT 1",
    "wrong_value": lambda s: f"SELECT * FROM ({s}) LIMIT 1 OFFSET 1",
}


def fake_gens(rate, seed):
    rnd = random.Random(seed)
    gens = {}
    for q in QUESTIONS:
        out, finish = f"Reasoning...\n```sql\n{q['SQL']}\n```", "stop"
        if rnd.random() < rate:
            kind = rnd.choice(list(MUTATIONS) + ["no_sql", "truncated", "raw_text"])
            if kind == "no_sql":
                out = "I am not sure how to write this query."
            elif kind == "truncated":
                out, finish = "Let me think step by step ... ```sql\nSELECT", "length"
            elif kind == "raw_text":
                out = q["SQL"]
            else:
                out = f"```sql\n{MUTATIONS[kind](q['SQL'])}\n```"
        gens[q["idx"]] = {"raw_output": out, "finish_reason": finish, "prompt_tokens": 1000, "output_tokens": 50,
                          "status": "ok"}
    # two guaranteed timeouts (an endless recursive CTE)
    for i in (0, 1):
        gens[i]["raw_output"] = "```sql\nWITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT MAX(x) FROM c\n```"
    return gens


for name, rate, seed in (("fake-a", 0.45, 1), ("fake-b", 0.30, 2)):
    ns["CFG"].update({"backend": "fake", "model_key": name, "ves_iterations": 1})
    ns["RUN_ID"] = f"test_{name}"
    ns["RUN_DIR"] = ns["RUNS_DIR"] / ns["RUN_ID"]
    ns["RUN_DIR"].mkdir(parents=True, exist_ok=True)
    ns["GENS"], ns["GEN_STATS"] = fake_gens(rate, seed), {"generation_seconds": 60.0}
    run_cells(step8)
    cats = ns["SUMMARY"]["categories"]
    print(name, "EX", ns["SUMMARY"]["EX_total"], cats)

expected = {"correct", "schema_error", "function_error", "syntax_error", "no_sql_found", "no_sql_truncated",
            "wrong_empty_result", "wrong_column_count", "wrong_row_count", "timeout"}
missing = expected - set(cats)
assert not missing, f"categories never produced: {missing}"
print("FAKE RUNS OK")

# Optional: prompt lengths in tokens ------------------------------------------------------------------------
tok_path = os.environ.get("T2S_TOKENIZER_JSON")
if tok_path:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    for style in ("omnisql", "arctic"):
        ns["MODEL"]["prompt_style"] = style
        lens = []
        for q in QUESTIONS:
            msgs, prefix = ns["build_prompt"](q)
            lens.append(sum(len(tok.encode(m["content"]).ids) for m in msgs) + 30)  # +30 ≈ chat-template tokens
        lens.sort()
        print(f"prompt tokens ({style}): median {lens[len(lens)//2]}, p95 {lens[int(len(lens)*.95)]}, max {lens[-1]}")

# Optional: cross-check with the official scripts ------------------------------------------------------------
official_dir = os.environ.get("T2S_OFFICIAL_EVAL_DIR")
if official_dir:
    import pandas as pd
    minidev = ns["MINIDEV_DIR"]
    run_a = ns["RUNS_DIR"] / "test_fake-a"
    diff_jsonl = work_dir / "mini_dev_sqlite.jsonl"
    diff_jsonl.write_text("\n".join(json.dumps(q) for q in json.load(open(minidev / "mini_dev_sqlite.json",
                                                                           encoding="utf-8"))), encoding="utf-8")
    ours = pd.read_csv(run_a / "per_question.csv")
    for script, col in (("evaluation_ex.py", "ex"), ("evaluation_f1.py", "soft_f1")):
        cmd = [sys.executable, script, "--predicted_sql_path", str(run_a / "predict_official_format.json"),
               "--ground_truth_path", str(minidev / "mini_dev_sqlite_gold.sql"),
               "--db_root_path", str(minidev / "dev_databases") + "/", "--num_cpus", "4", "--meta_time_out", "30",
               "--diff_json_path", str(diff_jsonl), "--sql_dialect", "SQLite",
               "--output_log_path", str(work_dir / "official_log.txt")]
        out = subprocess.run(cmd, cwd=official_dir, capture_output=True, text=True, env={**os.environ})
        print(out.stdout[-900:], out.stderr[-1500:])
        print(f"OURS {col}: simple {ours[ours.difficulty=='simple'][col].mean()*100:.2f} | "
              f"moderate {ours[ours.difficulty=='moderate'][col].mean()*100:.2f} | "
              f"challenging {ours[ours.difficulty=='challenging'][col].mean()*100:.2f} | total {ours[col].mean()*100:.2f}")
print("ALL LOCAL TESTS PASSED")
