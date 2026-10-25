# ai-cuda 项目环境

完成日期：2026-10-22。按用户要求在原来的 Conda 环境内升级，没有保留额外 `.venv`。

## 位置与启动

解释器位置通过不入库的 `.local/config.json` 中的 `ai_cuda_root` 指定。本机继续使用原来的 ai-cuda，Python 已于 2026-10-22 原地升级；没有新增环境。

首次在其他机器使用时，参考 `configs/local.example.json` 创建 `.local/config.json`，填写已有解释器、权重及环境自检素材路径。模型路径允许绝对路径或相对项目根目录的路径，不会自动下载。

在当前项目的 PowerShell 中运行：

```powershell
. .\environment\activate-ai-cuda.ps1
```

该脚本只修改当前 PowerShell 进程，设置解释器、运行库路径、UTF-8 和 `PYTHONNOUSERSITE=1`，避免混用用户目录中的依赖。重复加载不反复叠加环境路径，没有修改系统 PATH。编辑器应选择配置目录内的 python.exe。中文脚本请使用 PowerShell 7，或在旧 Windows PowerShell 中以 UTF-8 正确加载。

旧 Anaconda base 的 Conda 22.9 存在 SSL DLL 加载问题，本次未修改 base；通过临时 Micromamba 对 ai-cuda 原地完成升级后，已清理临时工具与缓存。今后可在上述激活脚本加载后使用 `python -m pip` 管理项目 Python 依赖。

## 已安装和验证

| 组件 | 版本 |
|---|---|
| Python | 3.12.15 |
| PyTorch | 2.7.1+cu128 |
| TorchVision | 0.22.1+cu128 |
| Ultralytics（运行 YOLOv8） | 8.3.203 |
| SAM-2 包 | 1.0，模型使用 SAM 2.1 tiny |
| OpenCV | 4.11.0.86 |
| NumPy | 1.26.4，PyPI Windows wheel |
| Pillow | 11.3.0，PyPI Windows wheel |
| Pandas | 2.3.3 |
| Matplotlib | 3.9.2 |
| Gradio | 5.49.1 |
| OpenAI SDK | 2.16.0 |

也已安装 Hydra、OmegaConf、iopath、tqdm、pycocotools、HTTP 和 dotenv 等依赖。完整版本快照见 `requirements.lock.txt`。不要只按这个快照从默认源重装 torch，应使用官方 cu128 构建；SAM-2 按下面的固定提交安装。

NumPy 使用 PyPI 的同版本构建，解决 Conda MKL 与 PyTorch 在训练阶段重复加载 OpenMP 的冲突；没有设置 `KMP_DUPLICATE_LIB_OK`。Conda 原有 NumPy/Pillow 记录可能仍在，今后若用 Conda 重装这些组件，需要再次验证推理和训练。

## 模型资源与来源

- YOLOv8 沿用本机已有的 yolov8n.pt，通过 `yolo_weights` 配置，未重复下载。
- SAM 2 权重：`../models/sam2.1_hiera_tiny.pt`，156,008,466 字节。
- 权重来源：https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
- 权重下载 SHA256：`7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`（本次下载记录，不代表独立官方校验值）。
- SAM 2 来源：https://github.com/facebookresearch/sam2/commit/2b90b9f5ceec907a1c18123530e92e794ad901a4
- 模型配置：`configs/sam2.1/sam2.1_hiera_t.yaml`，随 sam2 包安装。
- PyTorch Python 3.12 wheel SHA256：`2bb8c05d48ba815b316879a18195d53a6472a03e297d971e916753f8e1053d30`。
- TorchVision wheel SHA256：`650561ba326d21021243f5e064133dc62dc64d52f79623db5cd76637a9665f96`。
- 两个 wheel 均对照 https://download.pytorch.org/whl/cu128/ 的官方索引哈希验证后安装，下载副本已清理。

安装 SAM 2 时使用 `SAM2_BUILD_CUDA=0`，没有安装完整 CUDA Toolkit/C++ 编译工具。核心 GPU 分割可用；可选的小孔填充、碎点去除 CUDA 后处理未启用。使用 `build_sam2(..., apply_postprocessing=False)` 与本次验证一致。

## 验证结果

- `python -m pip check`：无依赖冲突。
- RTX 5070：矩阵计算、TorchVision CUDA NMS、卷积反向传播通过。
- 已有 COCO8 图片：YOLOv8 检出 6 个目标，实际取检测框交给 SAM 2，生成非空 480×640 掩膜。
- COCO8 副本：4 张训练图、4 张验证图，1 个 epoch，batch=2、imgsz=320、workers=0、amp=False，训练、验证和权重保存完成。仅为环境自检，不是正式项目训练或精度结论。
- Gradio：本地应用配置接口通过 TestClient 检查。
- `verification.json`、`training-verification.json`、`interface-verification.json` 保存检查结果；测试数据副本与测试权重已清理。

第一阶段只需检查配置、文件和版本，不进行模型推理或训练：

```powershell
python .\environment\verify_project.py
```

## 下一步

已完成八类小数据集长训练和 YOLOv8 → SAM 2 分割串联，运行方式见 [第五阶段说明](../docs/第五阶段分割说明.md)。当前数据与真实桌面泛化仍有缺口。第六阶段另用已有 OpenAI SDK 2.16.0 和 httpx 0.28.1 接入 DeepSeek deepseek-flash，已完成两张验证图云端调用；参数、密钥位置与独立入口见 [第六阶段说明](../docs/第六阶段视觉大模型说明.md)。第七阶段复用 Gradio 5.49.1 完成统一界面，本机没有新增依赖或训练。激活环境后运行 `python -m deskvision.ui`，打开 http://127.0.0.1:7860；详细操作见 [第七阶段说明](../docs/第七阶段界面说明.md)。

没有安装 Grounding DINO、Florence-2、DDS、Docker 或 TorchAudio，因为当前路线是 YOLOv8 + SAM 2。

## 其他机器的环境复现

以下是准备新机器的步骤，本机不要重复执行。先准备 Python 3.12 环境及适配驱动，再使用所选解释器安装官方 CUDA 构建：

```powershell
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r environment/requirements.txt -c environment/constraints.txt
$env:SAM2_BUILD_CUDA = '0'
python -m pip install --no-build-isolation --no-deps "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"
python -m pip check
```

构建时需要环境内已有兼容的 setuptools、wheel 和 torch；已验证的 setuptools 版本见 constraints.txt。安装固定 SAM 2 提交需 Git 和网络。模型权重另按上述来源准备，禁止把下载任意包时自动带来的新版 torch 视为等价环境。requirements.lock.txt 是现有环境的完整快照，包含一些非本项目必需包，用于诊断，不能在其他平台直接盲目照装。

## 按需重测与训练终端

加载环境后，`python environment/verify_environment.py` 会检查 GPU 运算并执行已有图片的 YOLOv8 到 SAM 2 推理，不训练。`python environment/verify_training.py` 会复制本机 COCO8 并做一轮训练自检；必须先打开可见终端让用户看到进度。第 1 阶段没有执行该训练脚本。正式项目训练另在第 4 阶段实现。

脚本生成的 JSON、训练副本和日志默认忽略，不入库。仓库只保存复现说明和脚本；本地路径、旧安装快照、权重及缓存均不上传。

## 第八阶段验收环境

第八阶段复用本机环境、权重和数据，没有新增依赖或训练。固定九图实验的精确依赖版本、权重和源码哈希见 `reports/stage8-protocol.json`；已有结果可用 `python -m deskvision.acceptance summarize` 离线核查，不需要再发API请求。完整恢复边界见 [验收与复现指南](../docs/验收与复现指南.md)。

演示素材使用本机已有FFmpeg 8.1.2、Pillow和OpenCV制作并解码检查。`python -m environment.create_acceptance_media` 只使用已有浏览器截图和实验产物生成截图序列视频，不运行模型或联网；FFmpeg不是识别功能的依赖，新机器仅在重新编码视频时需要。截图须先存在 `runs/stage8-demo/`，此脚本不伪造浏览器操作。

`python -m environment.package_acceptance` 在原工作区离线打包验收证据，逐项校验协议、文件哈希、ZIP完整性和本地密钥排除；产物及整包指纹位于忽略的 `runs/delivery/`。无需安装打包依赖；完整训练集与Python环境不在交付包内。
