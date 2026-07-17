# Qwen3-Embedding / Qwen3-Reranker 的并行机制，以及 benchmark 里 `-bs/--batch_size` 到底是什么并行

日期：2026-07-06。基于对 OpenVINO GenAI C++ 源码
（`benchmark/openvino.genai/src/cpp/src/rag/`）、官方 `llm_bench` 工具
（`tools/llm_bench/`）、本仓库 Python benchmark（`benchmark/tests/`）与 C++ 版
`model_bench` 的逐行阅读。所有结论都附代码出处。

---

## 0. 结论速览（TL;DR）

1. **embedding / reranker 都支持并行，但只有一种：**「**同一次推理内的 batch 维张量并行**」
   （intra-inference batched-tensor parallelism）。即把 N 条文本/N 个 (query,doc) 对
   pad 成一个 `[N, L]` 的张量，**一次 `infer` 全部算完**，由底层插件在 batch 维上并行。
   **没有**多 InferRequest、请求池、`AsyncInferQueue`、`NUM_STREAMS`、`THROUGHPUT` hint、
   线程池——pipeline 自身一个都不用（§1、§2）。

2. **能达到什么效果**：靠**摊薄固定开销**（tokenize、dispatch、kernel launch、权重读取）
   提升吞吐（docs/s、tokens/s），代价是 batch 内所有行 pad 到最长序列。它优化的是
   **吞吐**，不是单条**延迟**；且是"一次算 N 条"，不是"N 条同时各跑各的"（§3）。

3. **官方/脚本里的 `-bs/--batch_size` 算不算并行？分场景：**
   - **embedding**：算——它把同一条文本**复制 N 份**塞进一次 `embed_documents()`，
     形成 batch=N 的真张量并行（§4.3、§4.5）。
   - **reranker**：本仓库脚本**根本没有 `-bs`**，候选集大小用 `--num-documents` 控制；
     官方 llm_bench **强制 reranker batch_size=1**。reranker 的"并行"来自它**天生**就是
     "一个 query 的所有候选文档一次前向打分"，与 `-bs` 无关（§4.4、§4.5）。
   - **LLM（qwen3）/ VLM（qwen3.5）**：本仓库脚本里 `-bs` **被解析但没有被使用**，
     纯粹是写进结果 CSV 的元数据，`generate()` 每次只喂**一条** prompt——**不是任何形式
     的并行**（§4.1、§4.2）。官方 llm_bench 的 `-bs` 才真的会把 prompt 复制 N 份做 batch。

一句话：**embedding/reranker 的并行 = "batch 维张量并行，单请求一次算完"**；
`-bs` 在 embedding 里就是这个 batch 维的大小，在本仓库 LLM 脚本里是**无效元数据**，
在本仓库 reranker 脚本里**不存在**（改用 `--num-documents`）。

---

## 1. TextEmbeddingPipeline 的并行实现

源码：`src/cpp/src/rag/text_embedding_pipeline.cpp`，头
`src/cpp/include/openvino/genai/rag/text_embedding_pipeline.hpp`。类是 PIMPL，逻辑在
`TextEmbeddingPipelineImpl`。

### 1.1 API：sync 就是 async 的 start+wait（不是线程）

```cpp
// text_embedding_pipeline.cpp:135
EmbeddingResults embed_documents(const std::vector<std::string>& texts) {
    start_embed_documents_async(texts);   // 一批文档
    return wait_embed_documents();
}
```
公共方法：`embed_documents(vector<string>)` / `embed_query(string)`，及对应
`start_..._async` + `wait_...`（头 162–187）。头里明确注释：**"Only one method of async
family can be active"**——即底层**只有一个** InferRequest，异步族一次只能有一个在飞。
`src/rag/` 全目录**无** `std::thread / std::async / future / thread pool / AsyncInferQueue`。

### 1.2 batching：整批 N 条文本 → 一个张量 → 一次 `start_async()`

```cpp
// text_embedding_pipeline.cpp:229-243
const auto encoded = m_tokenizer.encode(texts, m_tokenization_params);  // 整个 vector 一次 tokenize
m_request.set_tensor("input_ids", encoded.input_ids);        // shape = [N, L]
m_request.set_tensor("attention_mask", encoded.attention_mask);
...
m_request.start_async();                                     // 对全部 N 条只发一次推理
```
- **没有逐文档 for 循环调用 infer**；唯一的按文档循环是①加 instruction 前缀
  `format_texts`（254–266）、②推理后把 `[N, hidden]` 输出切回 N 个向量
  `to_embedding_result`（276–293，读 `shape[0]` 作 batch）。
- batch 维由 tokenizer 一次 encode 出来（`input_ids.shape[0]==N`），batch 内 pad 到最长
  （或 `max_length`）。
- **只有一个** `ov::InferRequest m_request`（成员 173，构造处 131
  `compiled_model.create_infer_request()`）。

### 1.3 无多请求 / 无 streams

pipeline 自身**不读** `optimal_number_of_infer_requests`、**不设** `NUM_STREAMS` /
`PERFORMANCE_HINT(THROUGHPUT)`。`core.compile_model(model, device, properties)`（129）
只透传用户传入的 `properties`。要设备级多流并行，只能由调用方经 `properties` 传给插件，
pipeline 不代劳。（另有一个 `m_post_request` 成员 174，仅 NPU 动态路径用于 pooling/
normalize 后处理，CPU/GPU 上为空、直接短路 `if(!m_post_request) return input;`，不是并行手段。）

### 1.4 Config：可选把 batch 维/序列维「定形」

头 53–119 的 `Config`：
- `batch_size`（69–75）：**"Batch size of embedding model. Useful for database
  population. If set, the pipeline will fix model shape ... Number of documents passed
  should be equal to batch_size. For query embeddings, set to 1 or not set."**
- `max_length`、`pad_to_max_length`、`padding_side`、`pooling_type`(CLS/MEAN/LAST_TOKEN)、
  `normalize`、`query_instruction`/`embed_instruction`。

设了 `batch_size`（和/或 `max_length`）会 `reshape_model` 把 batch/seq 维定成静态
（125–127），之后每次调用**必须**正好传 `batch_size` 条，否则 `OPENVINO_ASSERT` 报错
（220–227）。不设则 batch 维动态，N 可逐次变化。这里的 `batch_size` 是**定形/优化**用途，
本质仍是 §1.2 的单请求 batch 张量并行。

### 1.5 NPU 路径（`src/rag/npu/text_embedding_pipeline.cpp`）

仍然**单请求**（`create_infer_request()`，npu:41）。区别：静态 `[batch_size, max_length]`
走 reshape+编译；动态则走 NPU 专用 decoder 编译（NPUW 分块 prefill）+ 把 pooling/normalize
拆成单独的 **CPU** 后处理模型（同步 `m_post_request.infer()`）。动态 NPU 只在 MEAN pooling
下允许 left-padding，否则报错要求用 `batch_size+max_length+pad_to_max_length` 定形。
NPU 后处理模型 batch 维固定为 1——NPU embedding 更偏向 batch=1/定形用途，不适合大动态 batch。

---

## 2. TextRerankPipeline 的并行实现

源码：`src/cpp/src/rag/text_rerank_pipeline.cpp`，头
`.../rag/text_rerank_pipeline.hpp`，样例 `samples/cpp/rag/text_rerank.cpp`。

### 2.1 API

```cpp
// text_rerank_pipeline.cpp:194
std::vector<std::pair<size_t,float>> rerank(const std::string& query,
                                            const std::vector<std::string>& texts) {
    start_rerank_async(query, texts);
    return wait_rerank();
}
```
返回 `(doc_index, score)` 列表；另有 `start_rerank_async` + `wait_rerank`（头 91–102）。
**没有单文档重载**——每次必须传整个候选 `texts`。同样注释"only one async method active"。

### 2.2 batching：N 个 (query,doc) 对 → 一个张量 → 一次 `start_async()`

```cpp
// text_rerank_pipeline.cpp:271-283  —— 把一个 query 对齐 N 个候选，凑成 batch=N
TokenizedInputs tokenize(const std::string& query, const std::vector<std::string>& texts) {
    if (m_tokenizer.supports_paired_input())
        return m_tokenizer.encode({query}, texts, m_tokenization_params);   // 成对编码，N 行
    std::vector<std::string> concatenated;                                  // 否则拼 query+text
    for (auto& text : texts) concatenated.push_back(query + text);
    return m_tokenizer.encode(concatenated, m_tokenization_params);         // 仍是 N 行的 batch
}
```
```cpp
// text_rerank_pipeline.cpp:199-223
m_request.set_tensor("input_ids", encoded.input_ids);      // [N, L]
m_request.set_tensor("attention_mask", encoded.attention_mask);
const size_t batch_size = encoded.input_ids.get_shape()[0]; // == N
...
m_request.start_async();                                    // N 个 pair 一次推理
```
输出 `[N, ...]` logits → 每文档一个分数 → `partial_sort` 取 `top_n`（225–261）。
**definitively 一次 batched 推理**，非逐文档循环、非分块。**单个** `InferRequest m_request`
（成员 265，构造 191）。同样无请求池/streams/hint。

### 2.3 `top_n` 不影响并行度

`top_n`（Config，头 17）只在推理后 `partial_sort`+`resize` 截断返回条数——**所有 N 个候选
永远都会被打分**，`top_n` 不减少推理量。

### 2.4 静态 reshape（NPU）：`reshape_reranker_static.py`

`benchmark/tests/reshape_reranker_static.py` 把 reranker IR reshape 成静态
`[num_candidates, max_length]`（`ov.PartialShape([num_candidates, max_length])`），并在文
档串里点明本质："**the reranker scores ALL candidate documents for a query in ONE forward
pass, so the batch dimension equals the number of candidate texts**"。因为 NPU 不接受动态
`[?,?]`，必须配 `pad_to_max_length=True` 定形；且**一个静态模型只对应一个固定的候选数**。
它还记录 NPU 编译时间随 `max_length`/batch 陡增（`[2,512]≈132s`、`[1,2048]≈462s`、
`[2,2048]>15min`），建议 NPU 上用小 `max_length`(256/512)。

---

## 3. 这种并行能达到什么效果？

**它是「batch 维张量并行」：一次推理把 N 条塞进去，靠插件在 batch 维上并行算完。**

- **收益 = 摊薄固定开销**：tokenize、host↔device dispatch、kernel launch、权重/常量读取
  等固定成本被 N 条分摊 → **吞吐（docs/s、tokens/s）随 N 上升**，直到算力/带宽饱和。
- **代价与边界**：
  - batch 内所有行 pad 到最长序列（或 `max_length`），**一条超长文本会拖累整批**的计算量。
  - **无自动子分批**：传多大 N 就是多大的单张量，N 过大 = 一次超大推理（显存/内存峰值高）。
  - **优化吞吐不优化延迟**：单条 query 的响应时间不会因为 batch 更快，反而可能更慢
    （要等整批）。
  - **无跨请求流水**：单 InferRequest + "一次只能一个 async" → 同一 pipeline 实例**不能**
    让两批同时在飞。要设备级并发，得开**多个 pipeline 实例**或经 `properties` 给插件设
    `NUM_STREAMS`/`THROUGHPUT`（pipeline 自己不设）。
- **reranker 的天然并行**：即便不谈 `-bs`，reranker 每次 `rerank(query, docs)` 本身就是
  "一个 query 的全部候选一次前向"——候选数 = batch 维，这就是它的并行来源。

---

## 4. benchmark 脚本里的 `-bs/--batch_size` 逐脚本拆解

### 4.1 `benchmark_qwen3.py`（LLM）：`-bs` 被解析但**未使用**（无效元数据）

- 解析：`lib/cli.py:78`，`default=1`，help 原文即 **"Batch size (informational; pipeline
  is invoked once per prompt)."**
- 运行时：每轮只对**一条** `prompt_text` 调 `run_llm_iteration`（`lib/genai_runner.py:368`
  `result = pipe.generate(inputs, gen_config, streamer=...)`，`inputs` 是单条 encode 结果）。
- `args.batch_size` **全脚本只出现一次**——写进结果 dataclass（`benchmark_qwen3.py:175`
  `batch_size=args.batch_size`）→ 落 CSV。它**从不**设张量 shape[0]、从不复制 prompt、
  从不传进 `generate()`。吞吐 `= output_size/(e2e/1000)`，**不含** batch_size 因子。
- ⇒ 对 LLM：**(a)/(b)/(c)/(d) 都不是**，就是个报告字段。**改 `-bs` 对 LLM 结果无任何影响。**

### 4.2 `benchmark_qwen3_5.py`（VLM）：同上，`-bs` 未使用

同一套 `add_common_gen_args`（`-bs` default 1）。只在 `benchmark_qwen3_5.py:355` 写进结果记录。
`run_vlm_iteration` 每次生成单条 `prompt_text`(+images)。无复制、无 batch 维、不影响吞吐。

### 4.3 `benchmark_embedding.py`（embedding）：`-bs` = **把同一文本复制 N 份**塞进一次调用

- 解析：`lib/cli.py:145`，help **"Number of texts encoded per embed_documents() call ...
  The benchmark replicates the resolved text(s) up to this batch size."**
- 运行时：
  ```python
  # benchmark_embedding.py:171-174
  if mode == "documents":
      batch_texts = [raw_text] * max(1, args.batch_size)   # 同一文本重复 N 次
  else:
      batch_texts = [raw_text]                             # query 模式恒为 1
  # lib/embed_runner.py:89
  result = pipe.embed_documents(texts)                     # texts==batch_texts, len==N
  ```
  即"把同一 prompt 复制 N 份"（llm_bench 经典做法）**且**这 N 份正是交给
  `embed_documents()` 的文档列表 → 落到 §1.2 的单请求 batch=N 张量并行。
- 吞吐随 `batch_size` 线性放大：`docs_per_s = batch/(lat/1000)`、
  `tokens_per_s = total_tokens/(lat/1000)`（`embed_runner.py:104-105`，`total_tokens =
  batch_size × per_text_tokens`）。
- ⇒ 对 embedding：`-bs` 就是 batch 维大小，**是真并行（batch 张量并行）的开关**。
  C++ 版 `model_bench` 完全一致：`embedding.cpp:210-211`
  `batch_texts.assign(bs, raw_text); ... pipe->embed_documents(batch_texts);`。

### 4.4 `benchmark_reranker.py`（reranker）：**没有 `-bs`**，用 `--num-documents`

- `add_reranker_args`（`lib/cli.py:160-176`）只有 `--texts/--texts-file/--num-documents/
  --top-n`，**无 `-bs/--batch-size`**；`RerankBenchmarkResult` 也没有 batch_size 字段。
- 候选集大小由 `--num-documents` 复制/裁剪（`benchmark_reranker.py:104-110`）。
  `pipe.rerank(query, documents)`（`embed_runner.py:143`）里 **`len(documents)` 就是 batch
  维**（§2.2），吞吐用 `n_docs`。
- C++ 版一致：`reranker.cpp` 用 `--num-documents` 复制/裁剪文档，`pipe->rerank(q, documents)`
  一次打分全部候选。
- ⇒ reranker 的"并行度"= 候选文档数，经 `--num-documents` 控制；不存在 `-bs`。

### 4.5 对照：官方 `llm_bench` 的 `-bs` 才真的做 batch（并印证以上）

- LLM：`task/text_generation.py:281` `input_text_list = [input_text]*args["batch_size"]`
  → encode 成 `[N, seq]` → `model.generate(...)`。CB pipeline 下拆成 N 个请求
  （`:217-218`）。指标乘 batch：`in_size = num_input_tokens*batch_size`（:411）、
  `estimate_throughput(latency, bs)`（`output_json.py`）。arg 在 `benchmark.py:195`。
- embedding：`task/text_embeddings.py:93,120` 同样 `[input_text]*batch_size` →
  `embed_documents(list)`——与本仓库 embedding 脚本行为一致。
- **VLM 与 reranker：官方强制 batch_size=1**（`visual_language_generation.py:57-59`；
  `text_reranker.py:221-223` 打印 **"Only batch size 1 available for Text Reranker
  Pipeline"** 并把 `self.batch_size=1`）。

> 关键差异：本仓库的 qwen3/qwen3_5 脚本是从官方 llm_bench 精简来的，但**删掉了**
> `[prompt]*batch_size` 那步复制，因此本地 LLM/VLM 的 `-bs` 退化为纯元数据。

---

## 5. "batch 并行" vs 其它可能被误认为并行的东西

| 名目 | 是不是"并行"？ | 说明 |
|---|---|---|
| embedding/reranker 的 batch 张量（N 条一次 infer） | ✅ 设备内 batch 维并行 | 唯一真并行；单 InferRequest 一次算完 |
| embedding `-bs` / 官方 llm_bench `-bs` | ✅（复制 N 份→上面那个） | 复制同一输入凑 batch，测吞吐 |
| 本仓库 LLM/VLM 脚本的 `-bs` | ❌ | 仅写 CSV，`generate()` 每次单条 |
| reranker `--num-documents` | ✅（=候选数=batch 维） | reranker 天生"一次算全部候选" |
| `--num-iters (-n)` | ❌ | 串行重复测量取统计，不重叠 |
| `top_n` | ❌ | 推理后排序截断，不减推理量 |
| 多 InferRequest / NUM_STREAMS / AsyncInferQueue | ⛔ 未使用 | 这三个 pipeline 都不用；要用需调用方经 properties 传插件或开多实例 |
| CB(PagedAttention) 的 continuous batching | (仅 LLM) | 官方 llm_bench 的 CB 路径会把 batch 拆成 N 个并发请求；embedding/reranker 不走 CB |

---

## 6. 实操建议

- **想给 embedding/reranker 提吞吐**：加大 batch（embedding 用 `-bs` 或
  `Config::batch_size`；reranker 用 `--num-documents` / 直接传更大候选集），并尽量让 batch
  内序列长度接近以减少 pad 浪费；必要时设 `max_length + pad_to_max_length` 定形（NPU 必需）。
- **想要设备级并发（多批同时在飞）**：pipeline 层面不支持，需**开多个 pipeline 实例**，
  或经 `properties` 给插件设 `NUM_STREAMS`/`PERFORMANCE_HINT=THROUGHPUT`（GPU）。
- **测 LLM 的"批"吞吐**：本仓库 qwen3/qwen3_5 脚本的 `-bs` 无效，应改用官方 llm_bench，
  或在脚本里显式把 prompt 列表喂给支持批的 pipeline（普通 `LLMPipeline.generate` 对
  `StringInputs` 列表 / CB pipeline 才有意义）。
- **别被 `-bs` 的存在误导**：在本仓库跑 qwen3/qwen3_5 时改 `-bs` 不会改变性能，只改 CSV 里
  那一列数字。

---

## 附：关键代码位置

| 主题 | 文件:行 |
|---|---|
| embedding sync=async | `src/cpp/src/rag/text_embedding_pipeline.cpp:135` |
| embedding 单请求 batch infer | 同上 `:229-243`（+ 单请求 131/173） |
| embedding Config.batch_size 定形 | 头 `text_embedding_pipeline.hpp:69-75`；cpp `:125-127,220-227` |
| embedding NPU 路径 | `src/cpp/src/rag/npu/text_embedding_pipeline.cpp` |
| reranker tokenize 成对/拼接 | `src/cpp/src/rag/text_rerank_pipeline.cpp:271-283` |
| reranker 单请求 batch infer | 同上 `:199-223`（batch=shape[0]）；单请求 191/265 |
| reranker 静态 reshape 说明 | `benchmark/tests/reshape_reranker_static.py` |
| `-bs` 定义（gen / embed） | `benchmark/tests/lib/cli.py:78, 145` |
| LLM `-bs` 仅写元数据 | `benchmark/tests/benchmark_qwen3.py:175`；`benchmark_qwen3_5.py:355` |
| embedding `-bs` 复制 N 份 | `benchmark/tests/benchmark_embedding.py:171-174`；`lib/embed_runner.py:89,104-105` |
| reranker 用 `--num-documents` | `benchmark/tests/benchmark_reranker.py:104-110`；`lib/embed_runner.py:143` |
| 官方 llm_bench `-bs` 真复制 | `tools/llm_bench/task/text_generation.py:281`；`text_embeddings.py:93,120` |
| 官方 reranker/VLM 强制 bs=1 | `tools/llm_bench/task/text_reranker.py:221-223`；`visual_language_generation.py:57-59` |
| C++ 版对应实现 | `model_bench/src/tasks/embedding.cpp:210-211`、`reranker.cpp:78-90,211` |
