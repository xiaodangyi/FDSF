# FDSF · 桌面物品识别 Web 部署项目

第三阶段的代码、计划、运行记录与验收说明统一保存在本仓库。目标是在 Linux + Docker 中部署可供老师、同学临时访问的双路线识别应用：上传同一张原图，独立运行 YOLOv8 + SAM 2 与 DeepSeek，并排展示结果。

## 阅读入口

- [PDF 要求逐项解读](docs/第三阶段需求与交付目标.md)：原文要求、学习点、项目选择和完成标准。
- [分阶段实施路线](docs/第三阶段分阶段实施路线.md)：每阶段做什么、为什么做、交付什么、怎么验收。
- [第三阶段开发与运行记录](第三阶段开发记录.md)：只记录实际发生的操作、结果、失败与待办。
- [交付清单](docs/第三阶段交付清单.md)：代码、环境、视频、说明和最终提交检查。
- [运行记录模板](docs/第三阶段运行记录模板.md)：后续每阶段沿用。
- [S2 API 接口说明](docs/第三阶段S2-API接口说明.md)：本地启动方式、请求字段、状态与当前限制。
- [S4 登录与权限说明](docs/第三阶段S4-登录与权限说明.md)：管理员如何预建账号、本机密钥、退出和任务隔离。
- [S5 队列与恢复说明](docs/第三阶段S5-队列与恢复说明.md)：Ubuntu 三个终端启动 Redis、worker、Web，排队/超时/补投规则及证据。
- [S6 Docker 与容器说明](docs/第三阶段S6-Docker与容器说明.md)：GPU 镜像、Compose 启停、独立容器账号、命名卷和本机验收。
- [第二阶段原项目说明](第二阶段项目说明.md)：已有模型、数据和实验的历史背景。

## 当前状态

2026-10-26：S0～S6 本机实现完成。S6 Ubuntu 完整回归 120 项全部通过；Windows 120 项运行、119 项通过、1 项 symlink 权限跳过。GPU Docker/Compose 实际构建启动、四服务健康、真实 YOLO/SAM、Redis/worker 故障恢复、整体重建保留历史与 ZIP、真实网页控件共 9 组容器检查通过。严格执行 1 次授权的 DeepSeek 双路请求，两路输入原图哈希一致，用量 1,874 tokens；YOLO 螺丝 2、螺母 2、垫圈 2，DeepSeek 螺丝 3、螺母 2、垫圈 2，差异保留，不代表精度验收。当前 Nginx 只供本机 HTTP 入口，S7 云端、HTTPS、异地访问和 S8 最终交付仍待完成。证据见 `reports/stage3-S6-*.json`。

后续按 S0～S8 推进。S0 是本轮需求与代码基线整理；每阶段完成后报告改动、检查、限制和提交，再讲解下一阶段。

## 代码来源与历史

复用自己的第二阶段项目 [xiaodangyi/123](https://github.com/xiaodangyi/123)，基准提交 `47bceeb0c402a2518d62901322a2c268d14b0e3a`。迁入包含原工作区未提交的 `deskvision/web/` 展示层及配套检查；来源与 SHA256 见[导入清单](reports/stage3-source-baseline.json)。旧提交历史保留在 123；FDSF 从明确的基线迁入开始持续记录第三阶段，不伪造旧历史。

`docs/开发路线.md`、`docs/开发记录.md` 和 `reports/stage4-*`～`stage8-*` 是第二阶段内部八步的历史材料，不是第三阶段部署完成证据。第三阶段以 S0～S8 文档为准。

## 离线检查与已有入口

在已安装项目依赖的 Python 3.12 环境中，从本仓库根目录运行：

```console
python environment/verify_stage3_baseline.py
```

此命令运行已有离线检查，原始日志保存到忽略的 `runs/stage3/S0/`，摘要写入 `reports/stage3-S0-checks.json`。不启动训练、模型推理或真实 DeepSeek 请求。请使用自己的解释器，不把本机路径写进公开配置。

S6 容器工作台启动（Ubuntu，已按 S6 说明准备私有配置和已有模型）：

```console
sudo docker compose -f compose.yaml -f compose.gpu.yaml build web
sudo docker compose -f compose.yaml -f compose.gpu.yaml up -d redis
sudo docker compose -f compose.yaml -f compose.gpu.yaml run --rm --no-deps bootstrap
sudo docker compose -f compose.yaml -f compose.gpu.yaml up -d --wait
# 本机 WSL：保持此命令/Ubuntu 终端运行，防止发行版空闲退出。
sudo docker compose -f compose.yaml -f compose.gpu.yaml up --no-color
```

登录 `http://127.0.0.1:8001/login`。最后一条命令保持前台连接并显示日志，Ctrl+C 会停止此 Compose 会话。容器使用独立数据库卷和 `.local/s6/accounts.json` 中的随机密码，旧本机密码/历史不自动迁入。详细准备、启停、故障恢复及本机代理边界见 S6 说明。只绑定本机，不代表已经云端部署。

S5 原生工作台启动（Ubuntu，同一个项目环境，分别开三个终端；不要与容器同时占用 8001）：

```console
python environment/start_stage3_s5.py redis
python environment/start_stage3_s5.py worker
python environment/start_stage3_s5.py web --port 8001
```

通过 `python environment/manage_users.py init-local` 生成忽略的本机 JWT 配置，再用 `create-user <账号>` 预建账号；本机已有账号不必重建。当前启动器地址 `http://127.0.0.1:8001/login`，只监听本机。队列依赖、启停和补投详见 S5 文档。完整识别另需八类 `best.pt`、SAM 2.1 tiny、`.local/config.json` 和有效 DeepSeek 密钥/余额；参考 `configs/local.example.json`、`configs/vlm.example.json`、`environment/README.md`。没有明确同意时不要运行付费云端路线。

原模块入口仍可用，但 S5 默认需要队列环境变量及独立 Redis/worker，不能只启动 Web：

```console
python -m deskvision.api
```

这个原入口默认端口 8000；当前 S5 启动器指定 8001，两者均是同一受保护应用。`DESKVISION_QUEUE_MODE=inline` 只供明确选择的离线回归/旧调试，不代表独立队列。历史数据库位于忽略目录 `runs/stage3.sqlite3`，启动自动迁移到 schema v3；也可运行 `python environment/migrate_stage3_db.py`。接口与历史细节见 S2/S3 说明，最新启动和恢复规则以 S5 为准。

权重、原始数据、密钥、上传图片和缓存不通过普通 Git 提交。新机器只拉代码不能重现完整推理，须按私有资源清单补齐。

## 实施顺序

S0 要求与基线 → S1 Ubuntu 与资源核验 → S2 API 与双路线 → S3 数据库与历史 → S4 登录/JWT → S5 消息队列 → S6 Docker/Compose → S7 Nginx/云端访问 → S8 验收与交付。

少量演示先用一台服务器；SQLite 和 Redis 可同机运行。GPU、域名和独立云数据库不是 PDF 明文要求。用户已选择 GPU 版 PyTorch，本轮移除 CPU 构建；云端规格另按预算和需求决定。S1 已通过 `/etc/os-release` 确认本机 `Ubuntu` WSL2 实例是 Ubuntu 26.04.1 LTS，真实推理结果以 S1 报告为准。
