# S2 FastAPI 接口

## 当前范围

S2 提供健康检查、图片上传、模型路线调用、任务状态/结果查询和产物下载，并将已有 Gradio 工作台挂载在同一服务的 `/ui/`。`deskvision/service.py` 是两种入口共用的执行层；它调用第二阶段已有模型模块，页面保留阈值、分路结果、数量对比和 ZIP 导出。

通俗解释：网页是操作窗口，API 是接单规则，执行层负责把同一张原图交给两条路线。网页回调直接调用共用执行层，不绕回本机 HTTP；网页任务暂不进入 API 的任务索引，两者在 S3 接入统一数据库。

当前任务暂存于单进程内存；服务重启后任务索引会消失，S3 将接入 SQLite。当前没有登录和用户隔离，因此仅监听 `127.0.0.1`，不可直接开放给局域网或公网；身份认证安排在 S4。

## 启动

在项目 Python 3.12 环境、FDSF 根目录运行：

```console
python -m deskvision.api
```

默认监听 `127.0.0.1:8000`。浏览器访问 `http://127.0.0.1:8000/ui/`，根路径会跳转到工作台。调试界面是 `/docs`，健康检查是 `/health`。启动和访问页面不触发识别；模型在首次识别时加载。

也可使用 `python -m uvicorn deskvision.api:create_app --factory --host 127.0.0.1 --port 8000`。停止服务按 `Ctrl+C`。应在 S1 的 GPU Python 环境运行；本阶段没有更换 PyTorch，也不安装 CPU 版。

页面操作：上传原图 → 按需修改识别设置 → 运行本地 YOLO + SAM → 核对检测图、数量和详细报告 → 下载 ZIP。若选择 DeepSeek 或两路识别，还须勾选云端调用确认；勾选本身不触发请求。替换原图或本地参数后，旧结果失效规则沿用原工作台。

## 请求流程

1. `GET /health`：返回 `{"status":"ok"}`，不加载模型。
2. `POST /api/v1/runs`：使用 multipart/form-data 上传 `image`，返回 HTTP 202 和任务号。路线 `route` 可选 `yolo`、`deepseek`、`both`，默认 `yolo`；后两种必须显式传 `allow_remote=true`，否则返回 422，且不创建任务、不调用模型。
3. `GET /api/v1/runs/{run_id}`：读取 `queued`、`running`、`succeeded`、`partial_failed` 或 `failed` 状态和已完成路线结果。
4. 使用结果里的 `artifact_urls` 下载检测图、分割图或大模型文本产物。接口不提供内部 `result.json`，避免输出服务端本地路径。

上传图片上限 32 MiB，须为可解码的单帧 JPEG、PNG、WebP 或 GIF，单边最长 8192 像素。任务 ID 使用 UUID；两条路线各自读取上传原图，成功结果 SHA256 必须匹配上传内容。不匹配返回路线错误 `input_mismatch`。损坏图、空文件和参数错误返回 422，超大图片返回 413。这里限制的是图片文件；公网请求体上限在 S7 的代理中另设。

| 表单字段 | 默认值 | 含义/范围 |
|---|---|---|
| `image` | 必填 | 原图文件 |
| `route` | `yolo` | `yolo` / `deepseek` / `both` |
| `allow_remote` | `false` | 明确同意发送原图至 DeepSeek 并产生 API 用量 |
| `mode` | `finetuned` | 八类微调模型；`baseline` 使用现有基线 |
| `confidence` | `0.25` | 检测置信度，严格大于 0、小于 1 |
| `iou` | `0.7` | NMS 阈值，严格大于 0、小于 1 |
| `image_size` | `640` | 32～2048，且为 32 的倍数 |
| `device` | `auto` | `auto` / `cuda:0` / `cpu`；GPU 环境仍可选择 CPU 设备，不代表安装 CPU 版 Torch |
| `segment` | `true` | 是否使用 SAM 生成实例掩膜 |

任务状态依次为 `queued → running → succeeded / partial_failed / failed`。进程内共用执行槽一次处理一个任务；双路线内部可并行。一路异常仍保留另一路结果，失败不写成数量零；成功的空检测则保留真实的零。`GET` 返回时间、参数、原图摘要、路线结果、数量对比和下载地址，不返回服务端结果目录。

下载只接受该路线报告列出的产物名，并检查任务目录、路径边界和符号链接；猜测 `result.json`、未列出的文件或越界路径均返回 404。网页 ZIP 延用已有导出功能，包含本次原始报告与图片；这是本机导出文件，不能当作已经脱敏的公开附件。

示例：

```bash
curl -F image=@desk.jpg -F route=yolo -F device=cuda:0 http://127.0.0.1:8000/api/v1/runs
curl http://127.0.0.1:8000/api/v1/runs/<run_id>
```

获得明确同意后才运行真实 DeepSeek 联调；请求需额外传 `-F route=both -F allow_remote=true`。本轮新增真实请求为 0 次。离线双路线测试模拟模型响应，不得当成真实大模型的 HTTP 验收。

## 检查与记录

在项目环境运行离线检查：

```console
python environment/verify_stage3_s2.py
```

该脚本在当前系统运行完整测试，保存本机原始日志，并生成 `reports/stage3-S2-<system>-checks.json` 和脱敏文本日志。

真实本地 GPU 检查须显式提供已有样图；会启动临时回环服务、上传、轮询、下载检测图/掩膜，并在结束时停止服务：

```console
python environment/verify_stage3_s2_http.py --image data/desk8-train-v1/images/test/loco_test_000.png --check-ui
```

`--check-ui` 额外通过 Gradio HTTP 客户端执行上传和本地识别回调，检查 ZIP 内容；会再运行一次本地模型。它不操作真实浏览器，也不选择外部模型路线。报告见 [Linux HTTP 联调](../reports/stage3-S2-linux-http.json)，完整过程见 [开发记录](../第三阶段开发记录.md)。

## 当前限制

- 队列与任务状态只在 API 进程内存；进程重启后不能恢复。生产 worker 在 S5 实现。
- 本版本未加入账号和 JWT；只能本机访问。S7 配置反向代理前须先完成 S4 权限。
- API 的 `BackgroundTasks` 适合当前演示串联，不是持久任务队列。
- 已做真实本地 GPU HTTP 联调；真实 DeepSeek 双路线联调与浏览器人工视觉验收仍待完成。
- 本机 Windows 与 Ubuntu 的 FastAPI 已安装版本分别为 0.159.1、0.162.0，报告如实记录。依赖清单固定 0.159.1；本轮未重装环境，S6 构建将按固定依赖重新验证。
- 离线检查仍有 Gradio/asyncio 的未关闭事件循环 ResourceWarning；未将其描述为已修复。
