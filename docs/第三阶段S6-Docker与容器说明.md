# S6：Docker、GPU 与持久化

更新：2026-10-26。S6 在本机 Ubuntu/WSL2 实施；实际验收结果以 `reports/stage3-S6-*.json` 为准。S7 云服务器、HTTPS 和异地访问另做。

## 这一步做什么

把依赖、代码和启动方式装进镜像。Compose 同时管理四个服务：Nginx 接收网页请求，Web 检查账号并保存任务，Redis 排队，worker 使用 GPU 识别。模型从宿主只读挂入；数据库、原图、结果、缓存保存在 `runs` 命名卷，Redis AOF 在 `redis` 命名卷。删除重建容器不等于删除数据卷。

镜像仍使用 GPU 版 `torch==2.7.1+cu128` / `torchvision==0.22.1+cu128`。只用 `compose.yaml` 时设备可选 CPU，不安装另一份 CPU PyTorch；使用 GPU 时加 `compose.gpu.yaml`，只给 worker 分配一块 GPU。当前验证硬件为 RTX 5070 Laptop GPU。新服务器的架构、显卡驱动和内存仍需现场验证。

## 当前电脑的启动命令

打开 Ubuntu 终端，进入仓库：

```bash
cd "/mnt/e/新建文件夹 (9)/FDSF"
```

第一次需要 Docker Engine、Compose、Buildx；使用 NVIDIA GPU 还需宿主可用的驱动与 NVIDIA Container Toolkit。当前电脑已安装，不重复安装。`environment/setup_container_gpu.sh` 是本轮使用的 root 安装脚本；它配置容器运行时，不安装或替换 WSL 显示驱动。不要把普通用户加入 docker 组当成无权限操作，Docker 管理权限相当于宿主高权限。

先在已有项目环境生成忽略的本机配置：

```bash
"$HOME/.venvs/deskvision-s1/bin/python" environment/prepare_stage3_s6.py
```

该命令只检查已有三份模型，不下载；生成 `.env`、`.local/s6/config.json`、JWT、两个随机账号，并私下复用已有 DeepSeek 密钥。重复运行保留既有账号密码和 `.env`。新机器可参照 `.env.example` 手工填写自己的 Linux 模型路径与私有目录，不能照抄本机绝对路径。

实际构建、初始化和启动：

```bash
sudo docker compose -f compose.yaml -f compose.gpu.yaml build web
sudo docker compose -f compose.yaml -f compose.gpu.yaml up -d redis
sudo docker compose -f compose.yaml -f compose.gpu.yaml run --rm --no-deps bootstrap
sudo docker compose -f compose.yaml -f compose.gpu.yaml up -d --wait
# 本机 WSL 还需保持一个前台进程，运行后保持终端打开：
sudo docker compose -f compose.yaml -f compose.gpu.yaml up --no-color
```

登录：`http://127.0.0.1:8001/login`；工作台：`http://127.0.0.1:8001/ui/`。仅本机可访问，启动本身不请求 DeepSeek。此前 S5 原生服务不能同时占用 8001。

**当前 WSL 的常驻条件：**systemd/Docker 服务本身未能让本机发行版持续运行。只有 `up -d` 后结束全部 WSL 用户进程，Ubuntu 会空闲退出，Docker 随之正常关闭；下次进入 WSL，自动重启的 Nginx 可能比 Web 更快，短暂返回 502。`restart: unless-stopped` 不能让已关闭的 Ubuntu 常驻。上面最后的前台 `up` 保持 WSL 活跃，保持该终端打开；Ctrl+C 会停止 Compose 服务。恢复时先运行 `up -d --wait`，等待健康后再保持前台连接。关机、休眠或主动 `wsl --shutdown` 仍会中断服务。

本轮为方便直接访问，额外启动了一个隐藏的 Windows WSL 前台 Compose 进程，不是计划任务或开机自动启动。PID 记录在忽略的 `.local/s6/gateway-windows.pid`，输出在同目录 `gateway-foreground*.log`。用下文正常停止命令关闭栈后，该进程会结束；下次按上述命令在自己的 Ubuntu 终端启动即可。

容器使用新的独立账号/数据库，旧 S4/S5 的密码和历史不会自动迁入。账号为 `teacher01` / `student01`；对应随机密码仅查看本机 `.local/s6/accounts.json`，不要截图公开或提交。该初始化命令只创建缺失账号，不擅自覆盖已有密码。管理员可交互重设密码：

```bash
sudo docker compose -f compose.yaml -f compose.gpu.yaml exec web python environment/manage_users.py --help
```

## 查看、停止和恢复

```bash
sudo docker compose -f compose.yaml -f compose.gpu.yaml ps
sudo docker compose -f compose.yaml -f compose.gpu.yaml logs --tail 100 worker
sudo docker compose -f compose.yaml -f compose.gpu.yaml run --rm --no-deps worker gpu-info
sudo docker compose -f compose.yaml -f compose.gpu.yaml exec worker python environment/manage_queue.py recover-expired
sudo docker compose -f compose.yaml -f compose.gpu.yaml exec worker python environment/manage_queue.py redeliver <原任务号>
```

`gpu-info` 实际执行 CUDA 矩阵运算；不能用 `nvidia-smi` 单独代替模型验收。worker 并发为 1，prefork 父进程不初始化 CUDA。Web、worker、Redis、Nginx 都有健康检查；健康不代表识别精度或外部 API 可用。

正常停止先停止入口接新任务，再等待 worker 完成，最后关闭所有容器：

```bash
sudo docker compose -f compose.yaml -f compose.gpu.yaml stop nginx web
sudo docker compose -f compose.yaml -f compose.gpu.yaml stop worker
sudo docker compose -f compose.yaml -f compose.gpu.yaml down
```

worker 默认留 32 分钟退出时间，覆盖单任务硬限制。正常 `down` 保留两个命名卷；**不要加 `-v`，该参数会删除持久化数据**。再次 `up -d --wait` 会复用数据。若修改 `COMPOSE_PROJECT_NAME`，卷名也会变，表现为一套新数据。Redis AOF everysec 仍有突然断电约最后一秒的写入窗口，进程重建测试不等于断电零丢失。

Redis 不可用时接口返回 503 和已保存的任务号；恢复后只对尚未执行的 queued 任务按原号补投。DeepSeek 已开始但结果未知时不得补投以产生第二次付费调用，规则沿用 S5。

## 私有资源与入口边界

- `.dockerignore` 默认排除全部，仅白名单包含代码、类别配置、依赖和必要管理脚本；Dockerfile 明确 COPY，不复制整个仓库。
- 模型只读挂载，Web/worker 以 UID 10001 运行、根文件系统只读、临时目录使用 tmpfs，持久数据只写共享卷。
- auth 只挂给需要的服务，DeepSeek secret 只挂给 worker；账号明文只挂给一次性 bootstrap。密钥不写入镜像环境变量。
- 只有 Nginx 映射宿主 `127.0.0.1:8001`，Web/Redis 不发布端口。当前 Nginx 覆盖转发客户端地址为 loopback，专门适配本机 HTTP 登录检查；内部 Web 信任代理头。
- **当前代理配置只用于本机 S6，不能直接改成公网监听后交付。** S7 必须重新核对可信代理、HTTPS Cookie、SSH/安全组和异地权限访问。
- 命名卷保留数据不等于备份。迁移云端前还需一致性备份 SQLite 与结果，并私下传递模型、密钥和账号；本轮没有自动上传它们。

## 验收命令与证据

离线回归不联网、不调用模型：

```bash
python environment/verify_stage3_s6.py
```

实际容器验收需要本机 Docker 管理权限和已有资源，运行真实 YOLO/SAM、worker/Redis 故障恢复、整体重建与原会话/历史/ZIP 保留：

```bash
sudo "$HOME/.venvs/deskvision-s1/bin/python" environment/verify_stage3_s6_containers.py --browser
```

看到 `BROWSER_READY` 后，在有 Playwright/Edge 的 Windows Node 环境执行 `node environment/verify_stage3_s6_browser.cjs`。它操作真实文件控件、预览、运行按钮、自动结果、下载、跨账号和退出，只点击本地路线。截图和原始日志保存在忽略目录。

只有明确授权才可给容器验收加 `--allow-one-remote`。该参数在最后提交一次真实双路请求；SDK 自动重试关闭，独占创建 `.local/s6/remote-attempt-used.json` 阻止脚本重跑时再次用量。**不要为重试删除此标记。** 超时或失败仍需核对供应商实际用量，不宣称没有计费。

公开证据：`reports/stage3-S6-{windows,linux}-checks.json/.txt`、`reports/stage3-S6-container-checks.json`、`reports/stage3-S6-browser-checks.json`。配置解析和离线测试不能代替实际 Docker 成功。真实图是已有演示样例，用于部署链路检查，不新增精度或泛化结论。

## 本轮实际结果

| 检查 | 结果 |
|---|---|
| Ubuntu 完整回归 | 120 项全部通过，退出码 0 |
| Windows 完整回归 | 120 项运行、119 项通过、1 项 symlink 权限跳过，退出码 0 |
| Docker/GPU/故障/持久化 | 9 组实际检查通过；GPU PyTorch 2.7.1+cu128、CUDA 12.8；YOLO/SAM 生成 6 个实例分割提示 |
| 浏览器实际控件 | 原图上传/预览、本地运行、Timer 结果、ZIP、跨账号拒绝、历史和退出通过；未在浏览器另点云端按钮 |
| 唯一一次授权双路 | 任务 `05d05a5a-ec9f-4aaa-80d9-59bf7cf3b145` 成功；原图 SHA256 两路一致；输入 1,668、输出 206、总 1,874 tokens |
| 样例结果差异 | YOLO 螺丝 2 / 螺母 2 / 垫圈 2；DeepSeek 螺丝 3 / 螺母 2 / 垫圈 2。原样保留，不为对齐结果重新请求或调参 |

实际环境为 Docker Engine 29.1.3、Compose 2.40.3、NVIDIA Container Toolkit 1.20.1。Python 基础镜像固定 digest；SAM2 固定官方提交；本轮镜像的全部 Python 依赖快照保留在 `reports/stage3-S6-container-packages.txt`。主依赖已固定，APT 与部分传递依赖的未来解析仍可能变化；快照记录本轮环境，不声称未来所有源包均不可变。

第一次 Compose 启动因 YAML 流式列表中未给 tmpfs 字符串加引号，逗号被拆成第二项，报 `invalid mount path: 'mode=1777'`；当时未调用 DeepSeek。改为单个带引号字符串并补充配置断言，随后完整重跑两平台回归和容器验收通过。首轮构建日志显示镜像构建成功，但旧 shell 日志包装器的退出码表达式为空；改用 `set -e`、单服务缓存重建后退出码为 0，未重新下载 CUDA。两项问题均没有通过忽略退出码解决。

Gradio 仍有上传完成后 `upload_progress?upload_id=undefined` 的非阻塞 404；实际控件、预览和结果通过，报告保留 warning。Windows 的框架 ResourceWarning/弃用提示仍保留。当前四服务通过前台 WSL 进程保持运行，宿主地址为 `127.0.0.1:8001`；重启电脑后需运行上述启动命令。

2026-10-27 接续检查确认，前次收尾看到的 502 源于 WSL 退出后冷启动：journal 显示 Docker 收到 systemd 正常停止信号，容器 RestartCount 均为 0；等待健康后 HTTP 200。额外恢复检查脚本 `environment/verify_stage3_s6_recovery.py` 在 Windows 连续观察一分钟，13 次 HTTP 均为 200、WSL boot ID 不变、四容器 healthy/重启计数 0，并只读验证原付费任务和 ZIP 保留；没有创建新识别任务。证据为 `reports/stage3-S6-recovery-checks.json`。
