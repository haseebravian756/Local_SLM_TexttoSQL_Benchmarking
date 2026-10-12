# My Knowledge Base: Text-to-SQL Thesis

A place to re-read explanations whenever I'm confused. New entries get added at the bottom.
Each section ends with **📚 Read more**: the papers and docs the explanation is based on.

**Contents**
1. [The BIRD Mini-Dev databases](#1-the-bird-mini-dev-databases)
2. [Running an LLM: where models are stored vs. the inference engine](#2-running-an-llm-where-models-are-stored-vs-the-inference-engine)
3. [How vLLM works inside (and why it matters for my results)](#3-how-vllm-works-inside-and-why-it-matters-for-my-results)
4. [A story: one question from prompt to answer (context window vs. KV cache), and many questions at once](#4-a-story-one-question-from-prompt-to-answer-and-many-questions-at-once)
5. [Prompts: the layers, the ingredients, classical formats, and the OmniSQL / Arctic formats](#5-prompts-the-layers-the-ingredients-classical-formats-and-the-omnisql--arctic-formats)
6. [Model families: reading model names, Qwen, text-to-SQL specialists, distillation, explaining results, tool calling & MCP](#6-model-families-reading-model-names-qwen-text-to-sql-specialists-distillation-explaining-results-tool-calling--mcp)
7. [Big schemas: when the full CREATE TABLE schema is too much, and what to do instead](#7-big-schemas-when-the-full-create-table-schema-is-too-much-and-what-to-do-instead)

**Decisions so far**
- Database to start with: **superhero** (small and simple, but most questions still need joins)
- Inference engine: **vLLM** suggested, with **Hugging Face Transformers** as the fallback

---

## 1. The BIRD Mini-Dev databases

### How the databases work

There are **11 databases**. Each one is a single SQLite file (one `.sqlite` file holds the whole database), about 1.4 GB in total.
Every question belongs to exactly one database, so the model only ever sees one database's structure at a time.
Eleven is a manageable number.

A few terms:
- **Table**: like one spreadsheet, for example `drivers` or `races`.
- **Column**: one field in a table, for example `driver_name`.
- **Foreign key (FK)**: a link between tables, for example `results.driverId` points to `drivers.driverId`.
- **JOIN**: combining linked tables in one query. More joins make a question harder.

### The 11 databases compared

The join numbers come from the gold (correct) SQL of each database's questions.

| Database | Tables | Columns | Widest table | Links (FKs) | Questions | Avg joins | Max joins | Questions needing a join | Size |
|---|---|---|---|---|---|---|---|---|---|
| **european_football_2** | 7 | **199** | Match (**115 cols**) | **29** | 51 | 0.84 | 2 | 78 % | **598 MB** |
| card_games | 6 | 115 | cards (74) | 4 | 52 | 0.77 | 2 | 69 % | 262 MB |
| **formula_1** | **13** | 94 | results (18) | 19 | 66 | 1.15 | **5** | 82 % | 22 MB |
| california_schools | 3 | 89 | schools (49) | 2 | 30 | 0.87 | 2 | 83 % | 11 MB |
| codebase_community | 8 | 71 | posts (21) | 13 | 49 | 1.00 | 2 | 80 % | 481 MB |
| thrombosis_prediction | 3 | 64 | Laboratory (44) | 2 | 50 | 0.90 | 2 | 82 % | 7 MB |
| financial | 8 | 55 | district (16) | 8 | 32 | **1.53** | 3 | **94 %** | 71 MB |
| student_club | 8 | 48 | member (9) | 8 | 48 | 1.10 | 4 | 81 % | 3 MB |
| superhero | 10 | 31 | superhero (12) | 11 | 52 | 1.52 | 3 | 92 % | <1 MB |
| debit_card_specializing | 5 | 21 | transactions_1k (9) | 1 | 30 | 0.93 | 2 | 77 % | 35 MB |
| toxicology | 4 | 11 | atom (3) | 5 | 40 | 1.02 | 2 | 80 % | 3 MB |

### Which is the most complex?

**european_football_2 has the most complex structure overall.**
- It has the most columns (199), one table with 115 columns (`Match`), the most links between tables (29), and the most data.
- For a model, that means a very long schema description to read, and many similar-looking column names to choose from (e.g. `home_player_1` … `home_player_11`).
- 24 % of its questions are rated *challenging*.

**formula_1 is the most complex for joins.**
- It has the most tables (13), and it contains the hardest query in all of Mini-Dev, with 5 joins in one query.

**Two surprises:**
- **financial** and **superhero** are small, but almost every question needs joins (avg 1.5 per question).
- **toxicology** is the smallest database (11 columns), yet it has the highest share of challenging questions (45 %). Difficulty doesn't only come from structure.

### Why superhero is a good first database

It has 10 tables, 31 columns and 52 questions, and the file is under 1 MB. Its schema is short, so prompts stay small and queries run
instantly. Yet 92 % of its questions still need joins, so it's simple without being trivial.

### 📚 Read more
- **BIRD benchmark paper:** Li et al., *Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs*, NeurIPS 2023. [arXiv:2305.03111](https://arxiv.org/abs/2305.03111) · [website](https://bird-bench.github.io/)
- **BIRD Mini-Dev (the 500-question set, data and official scripts):** [github.com/bird-bench/mini_dev](https://github.com/bird-bench/mini_dev)
- *The table above was computed by us* from Mini-Dev's `dev_tables.json` (tables, columns, foreign keys) and the gold SQL (join counts).

---

## 2. Running an LLM: where models are stored vs. the inference engine

### Three separate pieces

People often mix these up, and Ollama happens to bundle two of them:

1. **Where the model is stored:** usually **Hugging Face Hub**, which works like "GitHub for models". Every model we discussed (Qwen-Coder, OmniSQL, Arctic) lives there.
2. **The software that runs the model** (the "inference engine"). This is the real choice: Ollama, vLLM, Transformers, and so on.
3. **The hardware:** Kaggle's 2× T4 GPUs.

### The options on Kaggle

| Option | What it is | Strengths | Weaknesses for *my* project |
|---|---|---|---|
| **Ollama** | Easy model runner (built on llama.cpp), runs as a small server | Very easy; same commands on a laptop | Uses **compressed (4-bit) model versions** by default, so scores won't be comparable with papers. Few text-to-SQL models in its catalogue. Slow when you send many questions. Possible on Kaggle, but awkward (a server running in the background). |
| **llama.cpp** | The engine underneath Ollama | More control than Ollama | Same compression issue; more manual setup |
| **Hugging Face Transformers** | The basic Python library for loading and running models | Already installed on Kaggle; you see every step; nothing extra to set up | Slow: questions are processed one small batch at a time |
| **vLLM** | High-speed engine built for running many prompts | Runs models **straight from Hugging Face at full quality**; uses **both T4s together**, so 7B models fit; processes all questions at once; it's what the OmniSQL and Arctic authors used in their own evaluation code | Takes 3-5 minutes to install on Kaggle, and that install is the most likely thing to break |
| **Paid APIs** (OpenAI, Anthropic, Groq, …) | Someone else runs the model; you send text over the internet | No GPU needed; strong models | Costs money; closed models; not "your own" deployment |

### Why vLLM (with Transformers as the backup)

1. **Fair results.** vLLM runs the model in its original form, so the numbers can be compared with published papers. Ollama's compressed versions can lose accuracy.
2. **Changing models is one line.** Swap the Hugging Face name, e.g. `"Qwen/Qwen2.5-Coder-7B-Instruct"` → `"seeklhy/OmniSQL-7B"`.
3. **It uses the hardware properly.** A 7B model needs about 15 GB, which is more than one T4 can hold with room to work. vLLM splits it across both T4s.
4. **It's fast.** All 52 superhero questions should take roughly a minute or two once the model is loaded (an estimate, to be measured).

**The honest risk:** the vLLM install on Kaggle. If it fails, fall back to Transformers, which is already installed. It's slower but
always works, and the rest of the code stays almost the same.

**In one sentence:** Ollama is great for chatting with a model on a laptop; for **measuring** models in a thesis, vLLM fits better.

*Nuance:* Ollama also offers full-precision versions (e.g. the `qwen2.5-coder:7b-instruct-fp16` tag, 15 GB). The default
`qwen2.5-coder:7b` tag, however, is the 4-bit `q4_K_M` version (4.7 GB). Even at full precision, it isn't built to push many prompts through at once.

### 📚 Read more
- **vLLM paper (PagedAttention):** Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) · [vLLM docs](https://docs.vllm.ai/)
- **vLLM supports the T4** (needs compute capability ≥ 7.5): [vLLM GPU installation docs](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
- **Ollama's quantized default versions:** [ollama.com/library/qwen2.5-coder/tags](https://ollama.com/library/qwen2.5-coder/tags) · [llama.cpp](https://github.com/ggml-org/llama.cpp)
- **Hugging Face Transformers:** [docs](https://huggingface.co/docs/transformers)
- **The papers' authors evaluate with vLLM:** OmniSQL [`infer.py`](https://github.com/RUCKBReasoning/OmniSQL/blob/main/train_and_evaluate/infer.py) · Arctic-Text2SQL-R1 [`bird_eval`](https://github.com/snowflakedb/ArcticTraining/tree/main/projects/arctic_text2sql_r1)
- **vLLM on Kaggle's 2× T4 (community project with tests):** [github.com/kaggle-vllm/kaggle-vllm](https://github.com/kaggle-vllm/kaggle-vllm)

---

## 3. How vLLM works inside (and why it matters for my results)

This follows the order a question travels through vLLM. At each stage: what it means when analysing results.
Running examples: **Qwen2.5-Coder-1.5B** (first try) and **7B** (later).

### 3.0 One question's journey

```
your prompt (text)
   │  1. tokenizer: text → token IDs           "SELECT" → 4858
   ▼
[ token IDs ]
   │  2. PREFILL: read the whole prompt at once, store its "memory" (KV cache)
   ▼
[ KV cache ] ──► 3. DECODE loop: produce 1 token, add it to the KV cache, repeat …
   │
   │  4. stop when: end-of-turn token (finish_reason = "stop")
   │                 or token limit reached (finish_reason = "length")
   ▼
generated token IDs → tokenizer → text → we extract the SQL
```

### 3.1 Tokens and the context window

- Models don't read characters, they read **tokens** (word pieces). For English and SQL, **1 token ≈ 3-4 characters**.
- The **context window** is the most tokens a model can handle at once, **prompt + answer combined**. Qwen2.5-Coder was built for 32,768 tokens.
- In vLLM you set `max_model_len`, your own cap (e.g. 8,192). It saves memory, but **prompt + answer must fit under it**.

**For analysis:** if a prompt is too long, or the answer hits the token limit (`max_tokens`), the answer gets **cut off**. You'll see
`finish_reason = "length"`, and usually no complete SQL, so it is scored as wrong for a *technical* reason, not because the model
"didn't know". **Always count truncations separately.** The superhero schema is small, so prompts should be roughly 1-2k tokens (to be measured).

### 3.2 Prefill vs. decode: why generating is slow

| Phase | What happens | Speed |
|---|---|---|
| **Prefill** | All prompt tokens are processed in parallel | Fast: thousands of tokens per second |
| **Decode** | The answer is produced **one token at a time**, and every token needs a full pass through the model | Slow |

**Why decode is slow on a T4 (a back-of-envelope estimate):** to produce *one* token, the GPU must read **all** model weights from memory.
- A T4 reads about **320 GB/s**.
- 7B model in float16 = about 15 GB of weights, so at best about 50 ms per token on one GPU, or about 20 tokens/s.
- 1.5B model = about 3 GB, so at best about 10 ms per token, or about 100 tokens/s.

**The trick:** one read of the weights can serve **many questions at the same time**. That's why vLLM runs all 52 questions together (§3.5).
It costs barely more than running one.

**For analysis:** generation time depends mainly on **how many tokens the model writes**. A "reasoning" model like Arctic-R1 thinks out
loud for hundreds of tokens, so it's slower. **Always record output tokens per question** next to accuracy.

### 3.3 The KV cache: the model's short-term memory

While reading the prompt, each layer of the model computes two vectors per token, called **K** (key) and **V** (value). The next token needs
to "look back" at all previous tokens, so these are **stored rather than recomputed**. That store is the **KV cache**. It is the main
consumer of GPU memory after the weights.

**How big it is** (float16; *layers × 2 (K and V) × KV heads × head size × 2 bytes*):

| Model | Weights | KV cache per token | One 1,500-token question |
|---|---|---|---|
| Qwen2.5-Coder-**1.5B** | about 3.1 GB | about **28 KB** | about 42 MB |
| Qwen2.5-Coder-**7B** | about 15.2 GB | about **56 KB** | about 84 MB |

(Qwen uses "grouped-query attention": many query heads share a few K/V heads. That makes its KV cache much smaller than older designs.)

GPU memory is split like this (a rough estimate for 7B across both T4s):
```
per T4: 15 GB × 0.90 (gpu_memory_utilization)  ≈ 13.5 GB
        − 7.6 GB  half of the 7B weights (other half on the other GPU)
        − ~1 GB   working memory (activations, CUDA)
        = ~5 GB   KV cache  →  roughly 175,000 tokens
```
That's enough for **all 52 superhero questions at once**, many times over.

**For analysis:** at startup vLLM prints the real numbers, for example `GPU KV cache size: … tokens` and
`Maximum concurrency for 8192 tokens per request: …x`. Note them down; they explain your speed. If the cache is too small, vLLM refuses to
start ("max seq len is larger than the maximum number of tokens that can be stored in KV cache"). The fix is to lower `max_model_len`.

### 3.4 PagedAttention: how vLLM manages that memory

vLLM's core idea, borrowed from how operating systems manage RAM:
- Older systems **reserved** memory for the *maximum* answer length of every request, so most of it sat empty.
- vLLM cuts the KV cache into small **blocks of 16 tokens** ("pages") and hands them out **only as needed**.

The result is very little wasted memory, so many more questions fit at the same time.

### 3.5 Continuous batching and the scheduler

- **Old way (static batching):** start 8 questions together, wait until the *longest* answer finishes, then start the next 8. Short answers leave the GPU idle.
- **vLLM (continuous batching):** after **every token step** the scheduler removes finished questions and adds waiting ones, so the GPU stays busy.
- **If memory runs out:** vLLM temporarily pauses ("preempts") some questions and resumes them later. That is slower but still correct, and you'll see a warning in the log.
- **Chunked prefill:** very long prompts are read in pieces, mixed with decode steps, so nothing stalls.

**For analysis:** in batch mode, all questions run *together*, so a "time per question" isn't meaningful. Report **total time** and
**throughput** (questions per minute, output tokens per second) instead.

### 3.6 Prefix caching: a free speed-up for our prompts

All 52 superhero prompts start with the **same text**: instructions plus the superhero schema. Only the question at the end differs:

```
[ instructions + superhero schema ......... ][ question 1 ]
[ instructions + superhero schema ......... ][ question 2 ]   ← identical start
```

vLLM notices identical starting blocks and **reuses their KV cache** instead of recomputing them ("automatic prefix caching").
It's one reason to **put the question at the end** of the prompt.

### 3.7 Tensor parallelism: using both T4s

With `tensor_parallel_size=2`, every layer's weight matrices are **split in half**, one half per GPU. After each layer, the two GPUs exchange partial results.
- On Kaggle, the T4s talk over **PCIe** (no fast NVLink bridge), so the exchange costs time.
- **7B:** it doesn't fit comfortably on one T4, so TP=2 is required.
- **1.5B:** it fits on one T4, and TP=2 may even be *slower* because of the exchange overhead. A community benchmark saw TP=2 slower than one GPU for a tiny model.

**For analysis:** speed comparisons between models are only fair if you note the **TP setting** for each. Record it with every run.

### 3.8 Sampling: how the next token is chosen

At each step the model gives a probability for every possible next token. Then:

| Setting | Meaning | What we use |
|---|---|---|
| `temperature=0` | **Greedy decoding:** always take the most likely token | ✅ standard for benchmarks |
| `temperature>0` | Pick randomly according to the probabilities (higher = more random) | later: "majority voting" experiments |
| `max_tokens` | Hard limit on answer length | about 1024 for normal models |
| `seed` | Fixes the random choices | always set it |
| `n` | Answers per question | 1 |

The **chat template** matters too. Qwen expects special markers around each turn (`<|im_start|>user … <|im_end|>`). vLLM's tokenizer adds
them for us. With the **wrong template**, a model produces garbage or never stops.

### 3.9 Determinism: an honest caveat

Even with `temperature=0`, two runs can differ on **a question or two**. vLLM's own docs say it *does not guarantee reproducible results by
default, for the sake of performance*. The main cause (He / Thinking Machines, 2025) is that GPU kernels are not "batch-invariant": a question's
numbers change slightly depending on **how many other questions share its batch**. That can flip a near-tie between two tokens, and from then on
the answer goes a different way. Even with the extra settings, reproducibility only holds on the same hardware and vLLM version.

**For analysis:** small differences (±1-2 questions out of 52) between runs of the *same* setup are noise, not a real change. That's why we
save raw outputs and settings for every run. With only 52 questions, **1 question = about 2 percentage points**.

### 3.10 Cheat sheet: the settings we'll use

```python
LLM(
  model="Qwen/Qwen2.5-Coder-1.5B-Instruct",  # which model (from Hugging Face)
  dtype="float16",                 # T4 has no bfloat16
  tensor_parallel_size=1 or 2,     # 1 GPU or split across 2
  max_model_len=8192,              # cap on prompt + answer tokens
  gpu_memory_utilization=0.90,     # share of GPU memory vLLM may use (rest = safety margin)
  enforce_eager=True,              # skip "CUDA graphs": simpler and less memory, a bit slower
  seed=42,
)
SamplingParams(temperature=0, max_tokens=1024, seed=42)
```

### 3.11 What to record for each run, and why

| Record | Why it helps the analysis |
|---|---|
| prompt tokens per question | longer prompts → is accuracy lower? (big schemas) |
| output tokens per question | speed, and "does thinking longer help?" |
| `finish_reason` | separates *truncated* failures from *wrong* answers |
| total generation time and throughput | efficiency comparison between models |
| model name and revision, dtype, TP, max_model_len, temperature, seed | reproducibility: anyone can re-run the exact setup |
| KV cache size from the log | explains speed and memory limits |

### 📚 Read more
- **Transformers, attention, keys and values (the origin of "K" and "V"):** Vaswani et al., *Attention Is All You Need*, NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- **Tokens (byte-pair encoding):** Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units*, ACL 2016. [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)
- **Qwen2.5-Coder (the model family we start with):** Hui et al., *Qwen2.5-Coder Technical Report*, 2024. [arXiv:2409.12186](https://arxiv.org/abs/2409.12186)
- **Grouped-query attention (why Qwen's KV cache is small):** Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, EMNLP 2023. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
- **Prefill vs. decode, and why decode is memory-bound:** Pope et al., *Efficiently Scaling Transformer Inference*, MLSys 2023. [arXiv:2211.05102](https://arxiv.org/abs/2211.05102)
- **PagedAttention / vLLM:** Kwon et al., SOSP 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- **Continuous (iteration-level) batching:** Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models*, OSDI 2022. [paper](https://www.usenix.org/conference/osdi22/presentation/yu)
- **Chunked prefill:** Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*, 2024. [arXiv:2403.02310](https://arxiv.org/abs/2403.02310)
- **Prefix caching:** [vLLM design doc: automatic prefix caching](https://docs.vllm.ai/en/latest/design/prefix_caching.html) · related idea: Zheng et al., *SGLang* (RadixAttention), 2023. [arXiv:2312.07104](https://arxiv.org/abs/2312.07104)
- **Tensor parallelism:** Shoeybi et al., *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*, 2019. [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)
- **Sampling, temperature, top-p:** Holtzman et al., *The Curious Case of Neural Text Degeneration*, ICLR 2020. [arXiv:1904.09751](https://arxiv.org/abs/1904.09751)
- **Why temperature 0 is still not perfectly repeatable:** He / Thinking Machines Lab, *Defeating Nondeterminism in LLM Inference*, Sept 2025. [blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) · [vLLM docs: reproducibility](https://docs.vllm.ai/en/latest/usage/reproducibility.html)
- **T4 hardware specs (16 GB, 320 GB/s):** [NVIDIA T4 page](https://www.nvidia.com/en-us/data-center/tesla-t4/)
- **TP=2 slower than one GPU for a tiny model on Kaggle:** [kaggle-vllm benchmarks](https://github.com/kaggle-vllm/kaggle-vllm)

---

## 4. A story: one question from prompt to answer, and many questions at once

### Part 1: The journey of one question

**The question (idx 239, superhero):** *"Please list all the superpowers of 3-D Man."*

The two ideas the story keeps coming back to:
- 📄 **The context window is a sheet of paper.** One sheet per question, with room for **8,192 tokens** (our `max_model_len`). The prompt *and* the answer must both fit on this one sheet.
- 🗂️ **The KV cache is a drawer of note cards.** Reading a token properly is expensive: it passes through all 28 layers of the model. So the first time the model reads a token, it writes a summary card for it (the K and V vectors) and files it in the drawer, which lives in GPU memory. Later, instead of re-reading the text, it just flips through the cards. Cards are kept in **folders of 16** (the "blocks" from PagedAttention).

#### Chapter 1: Building the prompt (our Python code, on the CPU)
We glue together the instructions, the superhero schema (`CREATE TABLE superhero …`, `hero_power …`, `superpower …`), the evidence
(*"3-D Man refers to superhero_name = '3-D Man'; superpowers refers to power_name"*) and the question. The result is one long text of a few thousand characters.

#### Chapter 2: Cutting into tokens (the tokenizer)
The text becomes a list of numbers, about **1,200 tokens** (an estimate: the superhero schema alone is about 2,900 characters, roughly 850 tokens).
A word like *superpower* may become two pieces, `super` + `power`. Then the **chat template** wraps it in Qwen's markers:
`<|im_start|>user … <|im_end|> <|im_start|>assistant`. That last marker tells the model it's its turn to write.

#### Chapter 3: The paper check (context window)
1,200 prompt tokens + up to 1,024 answer tokens = 2,224, which is under 8,192. ✅ Plenty of room.
*(If the prompt were 7,500 tokens, only 692 would be left for the answer, and it might get cut off.)*

#### Chapter 4: Prefill: reading the whole prompt at once
All 1,200 tokens go through the 28 layers **in parallel**, in one big burst. For every token, at every layer, a card (K and V) is written:
- 1,200 tokens × about 28 KB ≈ **34 MB** of cards, in **75 folders** of 16.

At the end of the burst, the model scores every possible first word of its answer, and greedy decoding picks the most likely one, say *"To"*.

#### Chapter 5: Decode: writing one token at a time
For each new token:
1. Only the **new** token goes through the 28 layers (not the whole prompt again).
2. At every layer, its attention **flips through all the cards in the drawer** to decide what's relevant: the card for `'3-D Man'`, the cards for `hero_power`, the cards for `power_name`…
3. It picks the next token and **files one new card** for itself.

| Moment | Tokens on the paper | Folders (blocks) | Card memory (KV cache) |
|---|---|---|---|
| after prefill | 1,200 | 75 | about 34 MB |
| after 16 answer tokens | 1,216 | 76 (a new folder opened) | about 34 MB |
| end of answer (150 tokens) | 1,350 | 85 | about 38 MB |

**Why keep cards at all?** Without the drawer, writing token #1,350 would mean re-reading all 1,349 earlier tokens through every layer.
Over 150 answer tokens that's about **190,000** token passes instead of about **1,350**. The KV cache trades **memory for speed**.

#### Chapter 6: The ending
The model writes a little reasoning, then:
```sql
SELECT T3.power_name FROM superhero AS T1
INNER JOIN hero_power AS T2 ON T1.id = T2.hero_id
INNER JOIN superpower AS T3 ON T2.power_id = T3.id
WHERE T1.superhero_name = '3-D Man'
```
Then it writes its end-of-turn token `<|im_end|>`, so `finish_reason = "stop"` ✅.
*Alternative ending:* if it keeps rambling past 1,024 tokens, it's cut off mid-sentence and `finish_reason = "length"` ❌.

#### Chapter 7: Cleanup
The 85 folders go back to the shared pool. The schema folders may be kept, because the next superhero question starts with the same text
(prefix caching). The text answer goes to our code, which extracts the SQL, runs it on `superhero.sqlite`, and compares the rows with the gold answer: **EX = 1 or 0**.

#### Context window vs. KV cache, side by side

| | 📄 Context window | 🗂️ KV cache |
|---|---|---|
| What it is | a **limit** (a rule) | **memory** (actual storage) |
| Belongs to | **each question** separately | **the GPU**, shared by all questions |
| Measured in | tokens | bytes (folders of 16 tokens) |
| Set by | the model's design + our `max_model_len` | GPU memory left after the weights (`gpu_memory_utilization`) |
| When it runs out | the answer is cut off (`length`) or the prompt is rejected | vLLM won't start, or pauses questions (slower, still correct) |

In one line: **the context window says how long one question's paper *may* be; the KV cache is the shelf space where all the papers
currently being written keep their notes.**

### Part 2: Many questions, and the "two chats" picture

#### How ChatGPT/Claude-style chats really work
The model itself has **no memory between messages**. Every time you hit send, the app sends the **entire conversation so far** (every earlier
message plus your new one) as one big prompt. Anthropic's API docs say it directly: the API is stateless, so *"you always send the full conversational history"*.

So in a chat app:
- The "memory" of a chat = its **history re-sent every turn**, written on that chat's paper (context window).
- A very long chat eventually **fills the paper**, so apps start dropping or summarising old messages.
- **Two chats = two separate papers.** They never see each other. (App "memory" features just paste saved notes into the prompt as text.)

#### Our benchmark = 52 brand-new, one-message chats
Each question is its own request: **no history, no knowledge of the other 51 questions.** That's exactly what we want, because each question
must be judged on its own. If we put all 52 in one long chat, earlier answers could help or confuse later ones, and the history would eat up the paper.

#### How vLLM serves 52 questions at the same time
Think of a GPU **"tick"**: in each step, the model writes **one next token for every active question at once**, in one big calculation.
The weights (§3.2) are read **once** per tick and used 52 times. That's why batching is nearly free.

```
tick:   1          2     3     ...   40          41    ...   90
Q1   [prefill]    tok   tok   ...   <end> ✔ (folders freed)
Q2   [prefill]    tok   tok   ...   tok         tok   ...   <end> ✔
Q3   [prefill]    tok   tok   ...   tok         <end> ✔
...
Q53  (waiting) ......................[prefill]   tok   ...      ← joins as soon as space frees up
```
(With 52 questions everything fits at once. With 500 later, some wait in line. That's **continuous batching**.)

**What's separate and what's shared:**
- 🔒 **Separate:** each question's own folders of cards. When Q7 "flips back", its attention only reads **Q7's** folders. No content leaks between questions.
- 🤝 **Shared:** the model weights (read-only), the GPU's time, and, with prefix caching, the **identical schema folders**. These are like photocopies of the same instruction sheet: identical text for everyone, so nothing is revealed.
- ⚠️ **The one side effect:** batch-mates cause tiny rounding differences (§3.9), so a re-run can differ by a question or so.

#### The link back to ChatGPT
Commercial services do **exactly this** on their servers. Your chat and thousands of other people's chats are batched together on the same GPUs
(continuous batching), and your conversation history gets reused via prefix caching. Your batch-mates are also why the same question can get
slightly different answers there.

In our code, the whole of Part 2 is **one line**: `llm.generate([52 prompts])`. vLLM schedules everything and gives back 52 answers in the same order.

### 📚 Read more
- **KV cache blocks and PagedAttention:** Kwon et al., SOSP 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- **Continuous batching:** Yu et al., *Orca*, OSDI 2022. [paper](https://www.usenix.org/conference/osdi22/presentation/yu)
- **Prefix caching:** [vLLM design doc](https://docs.vllm.ai/en/latest/design/prefix_caching.html)
- **Keys and values in attention:** Vaswani et al., *Attention Is All You Need*, 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- **Batch-mates and nondeterminism:** He / Thinking Machines Lab, 2025. [blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- **Chat APIs are stateless (full history re-sent each turn):** [Claude docs: Using the Messages API](https://platform.claude.com/docs/en/build-with-claude/working-with-messages)

---

## 5. Prompts: the layers, the ingredients, classical formats, and the OmniSQL / Arctic formats

**The prompt is the text we send to the model, before it's tokenized.** With chat models it's built in **layers**:

```
Layer 1  CONTENT        what we write: instructions + schema + evidence + question
   ▼
Layer 2  MESSAGES       content put into "roles":  [{"role": "user", "content": ...}]
   ▼
Layer 3  CHAT TEMPLATE  the model family's special markers wrap the messages into one string
   ▼
Layer 4  TOKENS         the tokenizer turns that string into numbers (§4, chapter 2)
```

### 5.1 Layers 2-3: what Qwen actually receives

We hand over a list of messages. Each one has a **role**:
- **system:** standing instructions (who the model is, general rules).
- **user:** the request.
- **assistant:** the model's own turns.

The chat template turns that list into this exact string for Qwen:
```
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
……our prompt text (layer 1)……<|im_end|>
<|im_start|>assistant
```
(Qwen inserts that default system line when we don't give one.) Every model family uses **different markers**; Llama's look completely
different, for example. That's why we never type them by hand: the tokenizer's `apply_chat_template()` does it for whichever model is
loaded. So switching models doesn't break the prompt.

### 5.2 The ingredients of a text-to-SQL prompt (layer 1)

| Ingredient | Example | Why |
|---|---|---|
| Task instruction | "Generate a valid SQL query…" | tells the model the job |
| SQL dialect | "Database Engine: SQLite" | SQLite ≠ MySQL (e.g. no `YEAR()` function) |
| **Schema** | tables, columns, types, keys, example values | the model never sees the database itself, only this description |
| Evidence | "3-D Man refers to superhero_name = '3-D Man'" | BIRD's hint |
| Question | "Please list all the superpowers of 3-D Man." | |
| (optional) Examples | 1-5 solved question→SQL pairs ("few-shot") | shows the expected style |
| Reasoning instruction | "think step by step" | thinking before answering helps ("chain of thought") |
| Output format | "put the SQL in a ```sql block" | so our code can find the SQL |

### 5.3 "Classical" prompts: what the research found

How to write the schema was studied a lot, mostly on Spider, an older and easier benchmark:

1. **Rajkumar et al. (2022)** compared styles for OpenAI Codex. **`CREATE TABLE` statements plus 3 example rows per table** worked best (67 % on Spider).
2. **Chang & Fosler-Lussier (2023):**
   - `CREATE TABLE` works well.
   - **Adding foreign keys helps** (Codex 71.8 → 73.1 %).
   - **Showing real values helps**, either first rows or distinct values per column.
3. **DAIL-SQL (Gao et al., 2023)** compared 5 styles. The "OpenAI Demonstration" style was best for GPT-3.5 (75.5 %), but GPT-4 preferred the
   plainest one. Lesson: **the best format depends on the model.**

A classical prompt for our question, in the OpenAI Demonstration style from DAIL-SQL:
```
### Complete sqlite SQL query only and with no explanation
### SQLite SQL tables, with their properties:
#
# superhero(id, superhero_name, full_name, gender_id, ..., weight_kg)
# hero_power(hero_id, power_id)
# superpower(id, power_name)
#
### Please list all the superpowers of 3-D Man.
SELECT
```
It's short and simple: no types, no keys, no example values, no reasoning.

### 5.4 What "OmniSQL" and "Arctic" are

| | **OmniSQL** | **Arctic-Text2SQL-R1** |
|---|---|---|
| Who | Renmin University of China (Li et al., 2025) | Snowflake (Yao et al., 2025) |
| What | a family of text-to-SQL models (7B/14B/32B) | a text-to-SQL model (7B public) |
| Built how | took **Qwen2.5-Coder** and trained it further on **SynSQL-2.5M**, about 2.5 million generated question→SQL examples | took **Qwen2.5-Coder-7B-Instruct** and trained it with **reinforcement learning (GRPO)**: write SQL, run it, get rewarded if the result is correct. It learned to "think" before answering. |
| Prompt | published its own prompt layout ("OmniSQL format") | **adapted the OmniSQL layout**, plus a system message and a `<think>…</think><answer>…</answer>` structure |

So "prompt style: omnisql / arctic" means **which of these two published layouts we fill in**. They're names of research projects whose prompts we copy.

### 5.5 The real OmniSQL prompt for the 3-D Man question

The schema is generated from the real superhero database; 7 small tables are folded here to save space (the full schema is about 2,900 characters):

````
Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question.
Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
SQLite

Database Schema:
CREATE TABLE superhero (
    id integer, -- example: [1, 2]
    superhero_name text, -- example: ['3-D Man', 'A-Bomb']
    full_name text, -- example: ['Charles Chandler', 'Richard Milhouse Jones']
    gender_id integer, -- example: [1, 2]
    ...  (7 more columns)
    weight_kg integer, -- example: [90, 441]
    PRIMARY KEY (id),
    CONSTRAINT fk_superhero_gender_id FOREIGN KEY (gender_id) REFERENCES gender (id),
    ...  (6 more foreign keys)
);

CREATE TABLE superpower (
    id integer, -- example: [1, 2]
    power_name text, -- example: ['Agility', 'Accelerated Healing']
    PRIMARY KEY (id)
);

CREATE TABLE hero_power (
    hero_id integer, -- example: [1, 2]
    power_id integer, -- example: [1, 18]
    CONSTRAINT fk_hero_power_hero_id FOREIGN KEY (hero_id) REFERENCES superhero (id),
    CONSTRAINT fk_hero_power_power_id FOREIGN KEY (power_id) REFERENCES superpower (id)
);
...  (alignment, attribute, colour, gender, publisher, race, hero_attribute)
This schema describes the database's structure, including tables, columns, primary keys, foreign keys,
and any relevant relationships or constraints.

Question:
3-D Man refers to superhero_name = '3-D Man'; superpowers refers to power_name
Please list all the superpowers of 3-D Man.

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific
  column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
In your answer, please enclose the generated SQL query in a code block:
```sql
-- Your SQL query
```

Take a deep breath and think step by step to find the correct SQL query.
````

**Where the research shows up in it:**
- **`CREATE TABLE` + types** → Rajkumar 2022 and Chang 2023: the best schema format.
- **`FOREIGN KEY` lines** → Chang 2023: helps the model find the join path superhero → hero_power → superpower.
- **`-- example: [...]` values** → Chang 2023: real values help. Here, the model sees that names are written like `'3-D Man'`.
- **Evidence placed before the question** → OmniSQL's convention.
- **"Only output what is asked… nothing more"** → EX compares rows exactly (§4, chapter 7), so **one extra column makes a correct answer score 0**.
- **"Think step by step"** → chain-of-thought prompting (Wei et al., 2022).
- **"```sql" code block** → our code takes the last ```sql block as the answer.

### 5.6 Arctic's version: same content, different packaging

1. The first sentence becomes a **system** message.
2. The output format asks for `<think> reasoning </think> <answer> … ```sql … ``` </answer>`.
3. The assistant's turn is **pre-filled** with `Let me solve this step by step. <think>`, so the model starts directly in thinking mode.

### 5.7 Why use a model's own format?

A fine-tuned model learned on **one layout**. Give it a different one and it can lose accuracy, and then you're measuring the prompt, not
the model. The OmniSQL paper also evaluated *other* models with this same prompt; Qwen2.5-Coder-7B scored 50.9 % on BIRD dev. That gives us
published numbers to compare against.

**For the first step:** OmniSQL prompt + Qwen2.5-Coder-1.5B. Later, each ingredient becomes an experiment knob: evidence on/off,
0/2/5 example values, with or without few-shot examples, and so on. One change per run.

### 📚 Read more
- **Prompt formats compared:** Rajkumar, Li, Bahdanau, *Evaluating the Text-to-SQL Capabilities of Large Language Models*, 2022. [arXiv:2204.00498](https://arxiv.org/abs/2204.00498)
- **Database representation in prompts (CREATE TABLE, foreign keys, values):** Chang & Fosler-Lussier, *How to Prompt LLMs for Text-to-SQL: A Study in Zero-shot, Single-domain, and Cross-domain Settings*, NeurIPS 2023 TRL Workshop. [arXiv:2305.11853](https://arxiv.org/abs/2305.11853)
- **Five prompt representations compared (DAIL-SQL):** Gao et al., *Text-to-SQL Empowered by Large Language Models: A Benchmark Evaluation*, 2023. [arXiv:2308.15363](https://arxiv.org/abs/2308.15363)
- **OmniSQL:** Li et al., *OmniSQL: Synthesizing High-quality Text-to-SQL Data at Scale*, 2025. [arXiv:2503.02240](https://arxiv.org/abs/2503.02240) · [repo with the prompt template](https://github.com/RUCKBReasoning/OmniSQL)
- **Arctic-Text2SQL-R1:** Yao et al., *Arctic-Text2SQL-R1: Simple Rewards, Strong Reasoning in Text-to-SQL*, 2025. [arXiv:2505.20315](https://arxiv.org/abs/2505.20315) · [eval code with its prompt](https://github.com/snowflakedb/ArcticTraining/tree/main/projects/arctic_text2sql_r1)
- **Chain-of-thought prompting:** Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*, NeurIPS 2022. [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)
- **Few-shot prompting:** Brown et al., *Language Models are Few-Shot Learners* (GPT-3), NeurIPS 2020. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
- **GRPO (the reinforcement-learning method Arctic used):** Shao et al., *DeepSeekMath*, 2024. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- **Chat templates and roles:** [Hugging Face docs: chat templating](https://huggingface.co/docs/transformers/chat_templating)

---

## 6. Model families: reading model names, Qwen, text-to-SQL specialists, distillation, explaining results, tool calling & MCP

### 6.1 Who is Qwen, and how to read a name like `Qwen2.5-Coder-7B-Instruct`

**Qwen** is the LLM family made by **Alibaba Cloud** (China). It's a *general* model family, not a text-to-SQL family. The name decodes like this:

```
Qwen2.5  -  Coder  -  7B  -  Instruct
   │          │       │        │
   │          │       │        └─ "Instruct" = trained further to follow instructions / chat.
   │          │       │            Without it ("base") = raw autocomplete model, bad at chatting.
   │          │       └─ size: 7 billion parameters (the numbers inside the model)
   │          └─ specialisation: extra training on code. Others: Math, VL (vision), Omni (audio/video)
   └─ generation (version)
```

**Generations so far** (as of October 2026):

| Generation | When | Note |
|---|---|---|
| Qwen | 2023 | first |
| Qwen2 | Jun 2024 | |
| **Qwen2.5** | Sep 2024 | ← **our models are built on this one** (Coder variant: Nov 2024) |
| Qwen3 | Apr 2025 | big step for reasoning and tool use; Qwen3-Coder in Jul 2025 |
| Qwen3.5 / 3.6 | Feb / Apr 2026 | |
| Qwen3.8 | Aug 2026 | newest open models (e.g. 27B dense); Qwen4 is announced as "in training" |

**Why use an "old" 2024 model?** Because the text-to-SQL specialists (OmniSQL, Arctic) and their published scores are all built on
Qwen2.5-Coder. That makes our numbers directly comparable with papers. Newer Qwens are a great *later* comparison, if they fit on 2× T4
(a 27B model in float16 needs about 54 GB, so it would need 4-bit compression).

### 6.2 Is there a "text-to-SQL model" category? Who trained on the 2.5 million examples?

**Not an official category.** Any LLM can write SQL. "Text-to-SQL model" informally means **someone took a general model and trained it
further for SQL**. There are two very different training stages, done by **different people**:

| | Qwen2.5-Coder-7B-Instruct | OmniSQL-7B | Arctic-Text2SQL-R1-7B |
|---|---|---|---|
| Made by | **Alibaba (Qwen team)** | **Renmin University of China** | **Snowflake** |
| Training | **5.5 trillion tokens** of code and text: all kinds of code, SQL is a small part | took Qwen2.5-Coder-7B-Instruct and fine-tuned it on **SynSQL-2.5M** (2.5 million question→SQL examples over 16,583 *generated* databases) + Spider and BIRD training data | took Qwen2.5-Coder-7B-Instruct and trained it with **reinforcement learning**: reward when the SQL returns the right result |
| Goal | general coding assistant | SQL specialist | SQL specialist that "thinks" first |
| Hugging Face name | `Qwen/Qwen2.5-Coder-7B-Instruct` | `seeklhy/OmniSQL-7B` | `Snowflake/Arctic-Text2SQL-R1-7B` |

So **yes, these are separate models**: separate files with different numbers inside, made by different teams. But they share the
**same architecture and size**, so they need the same GPU memory and run at the same speed. Only their skills differ. Think of one car
model sold by three different tuning shops.

Scale check: the Qwen team's training was 5.5 *trillion* tokens; OmniSQL's extra training was 2.5 *million* examples. Fine-tuning is tiny
by comparison, but very focused.

### 6.3 "DeepSeek-R1-Distill-Qwen-7B" and similar names

**DeepSeek** is another Chinese AI company. They built a huge reasoning model, **DeepSeek-R1**. Then they **distilled** it:
- **Distillation** = a big "teacher" model writes lots of answers (with its reasoning), and a small "student" model is trained to imitate them. The student inherits part of the teacher's skill.
- DeepSeek generated **800k** samples with R1 and trained small **Qwen** and **Llama** models on them.

**The naming pattern:** `[who trained it]-[teacher/method]-Distill-[base family]-[size]`.
So `DeepSeek-R1-Distill-Qwen-7B` = DeepSeek's student, built on Qwen2.5-**Math**-7B.

Everything in one family tree:
```
Alibaba Qwen team
└── Qwen2.5 (Sep 2024)
     ├── Qwen2.5-Coder ── 0.5B … 32B (base + Instruct)
     │     └── Qwen2.5-Coder-7B-Instruct
     │           ├── OmniSQL-7B               (Renmin Univ.: + 2.5M SQL examples)
     │           └── Arctic-Text2SQL-R1-7B    (Snowflake: + reinforcement learning)
     ├── Qwen2.5-Math-7B
     │     └── DeepSeek-R1-Distill-Qwen-7B    (DeepSeek: + 800k R1 reasoning samples)
     └── Qwen2.5-14B / 32B
           └── DeepSeek-R1-Distill-Qwen-14B / 32B
```
(Many companies build on Qwen because it's open, small enough and strong. Even Alibaba itself has a SQL fine-tune, **XiYan-SQL**.)

### 6.4 Can they explain the results, or only write SQL?

- **Qwen2.5-Coder-Instruct (general):** yes. It's a chat model, so it can explain a query or describe the rows in words. A 1.5B model will do it simply; a 7B noticeably better.
- **OmniSQL / Arctic (specialists):** their papers only test SQL writing. The OmniSQL paper says nothing about other abilities. Heavy training on one task usually **weakens other skills** (known as "catastrophic forgetting"), so expect them to be worse at explaining.
- **A common design:** a *specialist writes the SQL → the database runs it → a general model turns the rows into a sentence.* An architecture option to keep in mind for the thesis.

### 6.5 Tool calling and MCP

**Tool calling** means the model doesn't touch the database itself. It writes a structured request, the app executes it, and the result goes back to the model:
```
model:  {"tool": "run_sql", "arguments": {"query": "SELECT ... "}}    ← just text, in a fixed format
app:    runs it on the DB → sends back the rows as a new message
model:  reads the rows → answers, or calls another tool (e.g. fixes an error)
```

**MCP (Model Context Protocol)** is an open **standard plug** for such tools, published by Anthropic in Nov 2024, like USB-C for AI tools.
A "database MCP server" offers tools like `list_tables`, `describe_table`, `run_query`, and any MCP-capable app can plug into it.

**How this relates to our benchmark:**
- **Our setup is "one-shot":** the whole schema goes into the prompt, the model writes SQL **once**, and we run it.
- **The agentic setup:** the model **explores** the database with tools (look at tables, check values, try a query, read the error, fix it) over several turns. Newer benchmarks like Spider 2.0 basically require this, and top BIRD systems use multi-step pipelines.

**Which models can do it?**
- Qwen2.5-Instruct chat templates support tool calls; a 1.5B model is weak at it, though.
- **Qwen3 and later** were trained much more for tool use and agents.
- **OmniSQL/Arctic** were *not* trained for tool calling, so don't rely on it.
- vLLM supports it in server mode (`--enable-auto-tool-choice --tool-call-parser hermes` for Qwen).

"One-shot vs. agentic with DB tools" could be a strong later chapter (see open_questions_todos.md, row 5).

### 📚 Read more
- **Qwen generations and licences:** [Wikipedia: Qwen](https://en.wikipedia.org/wiki/Qwen) · background on 2026 releases (blogs): [codersera guide](https://codersera.com/blog/qwen-3-5-complete-guide-2026/) · [Qwen timeline](https://www.scriptbyai.com/qwen-timeline/)
- **Qwen2.5-Coder (5.5 trillion tokens, sizes 0.5B-32B):** Hui et al., *Qwen2.5-Coder Technical Report*, 2024. [arXiv:2409.12186](https://arxiv.org/abs/2409.12186)
- **OmniSQL (built on Qwen2.5-Coder-Instruct; SynSQL-2.5M with 16,583 databases):** Li et al., 2025. [arXiv:2503.02240](https://arxiv.org/abs/2503.02240) · [OmniSQL-7B model card](https://huggingface.co/seeklhy/OmniSQL-7B)
- **Arctic-Text2SQL-R1:** Yao et al., 2025. [arXiv:2505.20315](https://arxiv.org/abs/2505.20315) · [model card](https://huggingface.co/Snowflake/Arctic-Text2SQL-R1-7B)
- **DeepSeek-R1 and its distilled models:** DeepSeek-AI, *DeepSeek-R1*, 2025. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) · [Distill model card (base models, 800k samples)](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
- **Distillation (the original idea):** Hinton, Vinyals, Dean, *Distilling the Knowledge in a Neural Network*, 2015. [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
- **Catastrophic forgetting:** Kirkpatrick et al., *Overcoming catastrophic forgetting in neural networks*, PNAS 2017. [arXiv:1612.00796](https://arxiv.org/abs/1612.00796)
- **MCP:** [Anthropic: Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) (Nov 2024) · [MCP documentation](https://modelcontextprotocol.io)
- **Tool calling with Qwen and vLLM:** [Qwen function-calling docs](https://qwen.readthedocs.io/en/latest/framework/function_call.html) · [vLLM tool calling docs](https://docs.vllm.ai/en/latest/features/tool_calling.html)
- **Agentic text-to-SQL benchmark:** Lei et al., *Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows*, 2024. [arXiv:2411.07763](https://arxiv.org/abs/2411.07763)

---

## 7. Big schemas: when the full CREATE TABLE schema is too much, and what to do instead

**Short answer:** for our benchmark the full schema is fine. For a big database (e.g. 62 tables with tens of columns each), pasting the whole
schema stops working well.

### 7.1 One clarification: the template doesn't need the *whole* database

The OmniSQL and Arctic templates need *a* schema in `CREATE TABLE` format, but **not necessarily the whole database**. You can give them only
the relevant tables and columns, in the same format.

OmniSQL was even **trained that way on purpose**. Its data-preparation code (`process_dataset.py`) builds training prompts with the keys and
the columns the gold SQL uses, plus a *random* share of the other columns, capped at 225 columns. So the model is used to seeing **partial schemas**.

### 7.2 Why the full schema becomes a problem

Rough numbers, based on the real superhero schema (2,897 characters for 31 columns, about **25-30 tokens per column** including keys and example values):

| Database | Columns | Schema in prompt (estimate) | Fits our 8,192 limit? |
|---|---|---|---|
| superhero | 31 | ~850 tokens | ✅ easily |
| european_football_2 (our largest) | 199 | ~5,000-6,000 tokens | ✅ yes |
| **example: 62 tables × ~30 columns** | **~1,860** | **~45,000 tokens** | ❌ no. That's even beyond Qwen2.5's native 32k window |

Three separate problems appear:
1. **It doesn't fit** (the context window, §4). Even if you raise `max_model_len`, every question carries about 45k tokens of schema.
2. **Accuracy drops.** The model must find 3 relevant columns among 1,860, a needle in a haystack. *Lost in the Middle* (Liu et al.) showed
   that models use information at the start or end of a long input much better than information in the middle. Small models like our 1.5B/7B suffer most.
3. **Cost and speed.** Prefill gets slow, and each question uses about 1.3 GB of KV cache (1.5B model), so fewer questions run at once.
   Prefix caching (§3.6) softens this, because all questions on the same database share the schema part, but it doesn't fix problems 1 and 2.

### 7.3 What to do instead: the toolbox from the literature

**A. Schema linking: pick the relevant part first.** This is the classic answer, and it turns generation into a two-step pipeline:
```
question ──► [Step 1: SELECT relevant tables/columns] ──► small schema ──► [Step 2: write SQL with OmniSQL prompt]
```
There are three ways to do step 1:

| Way | How | Example systems |
|---|---|---|
| Search | embed or keyword-match the question against table and column names and descriptions; keep the top matches + their foreign-key neighbours | retrieval steps in many pipelines |
| Trained ranker | a small model scores how relevant each column is | RESDSQL, CodeS |
| Ask the LLM | "Which tables and columns are needed?", then a second call with the pruned schema | DIN-SQL, MAC-SQL, CHESS (its Schema Selector cut tokens **×5** and *gained* about 2 % accuracy) |

**⚠️ The big risk is recall.** If step 1 drops a column the answer needs, step 2 *cannot* be right. Maamari et al. ("The Death of Schema
Linking?", 2024) argue that with strong models you should **skip** pruning whenever the full schema fits. Their system ranked first on BIRD that way.

**B. Make the schema shorter.** Drop the example values, shorten comments, or use more compact formats (e.g. XiYan-SQL's "M-Schema").
Chang & Fosler-Lussier also found that normalising the schema (lowercase, standard spacing) cuts tokens.

**C. Use a long-context model.** Chung et al. (Google, VLDB 2025) put whole schemas plus extra context into Gemini 1.5 Pro and found that
long-context models *"are robust and do not get lost"*. That's a big model with a huge window, though. On our T4s with 1.5B/7B models, A is more realistic.

**D. Let the model explore** (the agentic / MCP idea from §6.5). The model calls `list_tables`, looks at only the tables it wants, and tries
queries. Spider 2.0, with databases of **over 1,000 columns**, was built to test exactly this.

**E. Bring the right values in.** OmniSQL retrieves database values that match words in the question (BM25 search), so the model sees e.g.
`'3-D Man'` even if it's not among the two example values. This complements A.

### 7.4 Rule of thumb

| Situation | Approach |
|---|---|
| Schema fits comfortably, decent model | **full schema** (simplest; no recall risk) ← our BIRD Mini-Dev case |
| Schema too big, or small model struggling | **prune (A)** with high recall: keep generously + foreign-key neighbours |
| Huge enterprise database (hundreds or thousands of columns) | **agentic exploration (D)**, or prune + retrieve values (A + E) |

**How to measure it in a thesis:** schema linking is evaluated separately with **recall** = the share of the gold SQL's tables and columns
that step 1 kept. Then compare EX with the full schema vs. the pruned one, along with prompt tokens and time. A neat later experiment would be
"full vs. pruned schema" on european_football_2, our largest database (see open_questions_todos.md, row 6).

### 📚 Read more
- **Long inputs and lost information:** Liu et al., *Lost in the Middle: How Language Models Use Long Contexts*, TACL 2024. [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)
- **Schema linking with a trained ranker:** Li et al., *RESDSQL: Decoupling Schema Linking and Skeleton Parsing for Text-to-SQL*, AAAI 2023. [arXiv:2302.05965](https://arxiv.org/abs/2302.05965) · Li et al., *CodeS: Towards Building Open-source Language Models for Text-to-SQL*, SIGMOD 2024. [arXiv:2402.16347](https://arxiv.org/abs/2402.16347)
- **Schema linking by asking the LLM:** Pourreza & Rafiei, *DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction*, NeurIPS 2023. [arXiv:2304.11015](https://arxiv.org/abs/2304.11015) · Wang et al., *MAC-SQL: A Multi-Agent Collaborative Framework for Text-to-SQL*, 2023. [arXiv:2312.11242](https://arxiv.org/abs/2312.11242) · Talaei et al., *CHESS: Contextual Harnessing for Efficient SQL Synthesis*, 2024 (Schema Selector: ×5 fewer tokens). [arXiv:2405.16755](https://arxiv.org/abs/2405.16755)
- **Against pruning when the schema fits:** Maamari et al., *The Death of Schema Linking? Text-to-SQL in the Age of Well-Reasoned Language Models*, 2024. [arXiv:2408.07702](https://arxiv.org/abs/2408.07702)
- **Long context instead of pruning:** Chung et al., *Is Long Context All You Need? Leveraging LLM's Extended Context for NL2SQL*, VLDB 2025. [arXiv:2501.12372](https://arxiv.org/abs/2501.12372)
- **Compact schema format (M-Schema):** Gao et al., *A Preview of XiYan-SQL: A Multi-Generator Ensemble Framework for Text-to-SQL*, 2024. [arXiv:2411.08599](https://arxiv.org/abs/2411.08599)
- **Schema normalisation:** Chang & Fosler-Lussier, 2023. [arXiv:2305.11853](https://arxiv.org/abs/2305.11853)
- **Huge schemas and agents:** Lei et al., *Spider 2.0*, ICLR 2025 (databases with over 1,000 columns). [arXiv:2411.07763](https://arxiv.org/abs/2411.07763)
- **Value retrieval and partial-schema training:** [OmniSQL paper](https://arxiv.org/abs/2503.02240) · [OmniSQL `process_dataset.py`](https://github.com/RUCKBReasoning/OmniSQL/blob/main/train_and_evaluate/process_dataset.py)
