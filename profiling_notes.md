# Profiling Notes: PyTorch / SGLang / vLLM

本文整理原生 PyTorch Profiler、SGLang 與 vLLM 如何錄製特定 profiling 區間。

## 1. 原生 PyTorch Profiler

原生 PyTorch Profiler 適合用在自寫 benchmark、minimal reproduction，或要修改框架 profiler 內部實作時。

### 1.1 context manager用法
使用torch.profiler.profile 直接包住要錄製的範圍

```python
import torch

activities = [
    torch.profiler.ProfilerActivity.CPU,
    torch.profiler.ProfilerActivity.CUDA,
]

with torch.profiler.profile(
    activities=activities,
    record_shapes=False,
    with_stack=True,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./torch_profile"),
) as prof:
    for i in range(20):
        run_one_iteration()
```


### 1.2 手動 `start()` / `stop()`

手動控制 lifecycle，使用profile.start() / profile.stop() 包住要錄製的範圍

```python
import torch

prof = torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    with_stack=True,
)

prof.start()

run_workload()

prof.stop()
prof.export_chrome_trace("trace.json")
```

### 1.3 `torch.profiler.schedule`

PyTorch 原生 profiler 可以用 `schedule` 控制 wait / warmup / active：

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(
        wait=5,
        warmup=2,
        active=10,
        repeat=1,
    ),
    record_shapes=False,
    with_stack=True,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./torch_profile"),
) as prof:
    for i in range(100):
        run_one_iteration()
        prof.step()
```

語意：
- `prof.step()`: 推進 profiler schedule 的 step；使用 `schedule=...` 時是必要的 step control。
- `wait`: 完全不錄，跳過 steps。
- `warmup`: profiler 啟動但不保存正式 trace，讓 profiler / CUDA 狀態穩定。
- `active`: 真正錄製的 steps。
- `repeat`: 重複幾輪。

例如：

```python
torch.profiler.schedule(wait=5, warmup=2, active=10, repeat=1)
```

表示先跳過 5 次 `prof.step()`，接著 warmup 2 次 `prof.step()`，最後正式錄製 10 次 `prof.step()`。


## 2. SGLang
以下指令使用 SGLang 0.1.dev1+g2637c0557.d20260419 版本測試，若使用其他版本，需要注意可能會有指令上的不同。

### 2.1 Server Mode 使用 client端 benchmark 控制 profiler

Server mode 指的是先啟動 SGLang HTTP server，再從 client 端送 requests 或 benchmark workload。

啟動 server：

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct
```

送 client benchmark，讓 `bench_serving` 自己控制 profiler：

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --num-prompts 100 \
  --profile \
  --profile-start-step 5 \
  --profile-steps 10
```


在較新的 SGLang 版本中，`bench_serving` 支援：

- `--profile-start-step`: 會傳成 `/start_profile` JSON body 裡的 `start_step`。
- `--profile-steps`: 會傳成 `/start_profile` JSON body 裡的 `num_steps`。
- `--profile-num-steps`: 舊/相容用法，也會影響 `/start_profile` 的 `num_steps`，但不表達 start step。

### 2.2 Server Mode 使用 start_profile / stop_profile 控制

典型順序是：

1. 啟動 server
2. 等 server ready
3. 先呼叫 `/start_profile`
4. 再啟動 client benchmark / workload
5. 如果沒有指定 `num_steps`，最後手動呼叫 `/stop_profile`

範例：

```bash
# Terminal 1: start server
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct
```

```bash
# Terminal 2: arm profiler before sending workload
curl -X POST http://127.0.0.1:30000/start_profile \
  -H "Content-Type: application/json" \
  -d '{
    "output_dir": "/tmp/profiles",
    "start_step": 5,
    "num_steps": 10,
    "activities": ["CPU", "GPU"]
  }'
```

```bash
# Terminal 3: send workload
python -m sglang.bench_serving \
  --backend sglang \
  --num-prompts 100
```

如果 `num_steps` 有指定，通常 profiler 會自動停止。如果沒有指定 `num_steps`，需要手動停止：

```bash
curl -X POST http://127.0.0.1:30000/stop_profile
```

注意：SGLang 的 start_step 不是「從 /start_profile 開始再等 N 步」的相對值，而是對 scheduler.forward_ct 的目標值。
如果 server 已經跑過一些 request，例如目前 forward_ct 已經是 100，這時再設定 start_step=5，不會再等 5 步，也不會回到第 5 步；因為第 5 步早就過去了。SGLang 會改成至少從下一個 forward step，也就是 forward_ct=101 時開始 profile。

常見做法是先下 `/start_profile`，再立刻跑 client benchmark。

#### `/start_profile` 參數

- `output_dir`: trace 輸出目錄。如果沒有指定，會使用 `SGLANG_TORCH_PROFILER_DIR`，再 fallback 到 `/tmp`。
- `start_step`: 從第幾個 step 開始錄，常用來跳過 warmup。
- `num_steps`: 錄製幾個 steps。有指定時通常會自動停止。
- `activities`: profiler activity，例如 `["CPU", "GPU"]`。
- `merge_profiles`: distributed profiling 時是否合併 traces。
- `bench_serving --profile-start-step`: client 端 shortcut，會傳成 `/start_profile` 的 `start_step`。
- `bench_serving --profile-steps`: client 端 shortcut，會傳成 `/start_profile` 的 `num_steps`。
- `bench_serving --profile-num-steps`: 舊/相容 shortcut，也用於控制 `/start_profile` 的 `num_steps`，但不控制 `start_step`。


## 3. vLLM

以下指令使用 vLLM 0.19.2rc1.dev228+gebf862c35.rocm722 版本測試，若使用其他版本，需要注意可能會有指令上的不同。
vLLM 新版主要用 `ProfilerConfig.delay_iterations/max_iterations` 控制錄製範圍。

### 3.1 Server Mode 使用 client端 benchmark 控制 profiler 的啟動
vLLM Client端不像 SGLang 一樣有辦法傳送 --profile-start-step/--profile-steps 給 start_profile，只能在 Server端指定

vLLM server mode 指的是用 OpenAI-compatible server：

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --profiler-config '{
    "profiler": "torch",
    "torch_profiler_dir": "./vllm_profile",
    "delay_iterations": 5,
    "max_iterations": 10
  }'
```

Client benchmark

```bash
vllm bench serve  --model meta-llama/Llama-3.1-8B-Instruct \
  ... \
  --profile
```

語意：

- `delay_iterations`: 呼叫 `/start_profile` 後，先跳過幾個 engine iterations。
- `max_iterations`: 最多錄幾個 engine iterations。`0` 通常表示不限制。
- `torch_profiler_dir`: trace 輸出目錄。

### 3.2 Server Mode 使用 start_profile / stop_profile 控制 profiler 的啟動
vLLM start_profile 不像 SGLang 一樣有辦法傳送 JSON body，只能在 Server端指定 Iteration 區間。

啟動 profiling：

```bash
curl -X POST http://localhost:8000/start_profile
```

若沒有指定max_iterations, 手動停止 profiling：

```bash
curl -X POST http://localhost:8000/stop_profile
```