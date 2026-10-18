# Qwen 声音克隆部署

使用参考录音、对应台词和待生成文本进行声音克隆，无需先训练角色。桌面版支持录音检查、长文本分段、取消生成、五分钟空闲后释放模型和诊断信息导出；也提供本地网页入口。

## 一键部署与启动

在本文件所在目录打开 Windows PowerShell：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ./Setup.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File ./Start.ps1
```

网页模式与停止服务：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ./Start-Web.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File ./Stop.ps1
```

诊断环境：`./Check.ps1`。已经完整安装且下载完成后，可用 `./Setup.ps1 -Offline` 校验；首次安装需要联网。若当前会话阻止脚本，使用上面带 `-ExecutionPolicy Bypass` 的执行方式。

需要 Windows x64、兼容 CUDA 12.8 的 NVIDIA 驱动及足够显存。默认安装 PyTorch 2.9.1 + CUDA 12.8 运行库，无需另装完整 CUDA Toolkit；具体版本以 `deployment/dependencies.json` 为准。模型本身约 4.23 GiB，另有 Python 环境、约 2.67 GiB 的 PyTorch 安装包和下载缓存，建议预留 20 GiB 以上空间。具体占用取决于缓存及输出。

## 使用

选择清晰的单人参考音频，填写与录音一致的台词、目标文本和语言，然后生成。输出保存在本项目的 `outputs/`。参考音频允许 0.25～120 秒，拒绝静音、无效波形和过长录音；质量仍取决于录音、台词对齐及模型效果。长文本会拆分并在片段之间插入停顿。

默认模型为 `Qwen/Qwen3-TTS-12Hz-1.7B-Base`。`configs/runtime.json` 通过模型 ID 选择 `deployment/models.json` 中的项目相对目录。示例录音首次部署一并下载；不要把它误当成自己的角色训练数据。

## 结构与数据流

```text
Setup.ps1 -> deployment/Setup.ps1 -> dependencies.json + models.json
Start.ps1 -> desktop_app.py -> desktop_worker.py -> common/engine.py
Start-Web.ps1 -> app.py -> common/model.py
common/paths.py : 地址簿；common/io.py : 共享文件读写
common/audio.py : 参考音频校验与原子输出
common/runtime.py : 项目内缓存、离线推理和模型地址解析
```

桌面工作进程输入为逐行 JSON：`{reference, transcript, text, language}`，输出为以 `VOICE_EVENT ` 开头的事件 JSON。界面将录音与文字送入独立工作进程，处理完成后只返回输出位置和时长；模型根据需要加载。工作进程的停止与释放由界面管理。

`deployment/dependencies.json` 保存全部依赖版本和工具地址；`deployment/models.json` 保存模型文件清单、大小及 SHA256。下载使用临时文件，校验通过才发布。两个 JSON 是部署信息的来源，代码不使用固定盘符，也不导入另一个子项目。

## 仓库与验证

`.venv/`、`runtime/`、`tools/`、`models/`、`cache/`、`temp/`、`logs/`、`outputs/` 均不提交。移动整个文件夹后，请重建 `.venv`；模型与下载文件可继续复用。请单独保存录音及生成结果。

完整本地测试：

```powershell
./.venv/Scripts/python.exe -B -m unittest discover -s tests -v
```

测试包含桌面离屏状态、UTF-8 工作进程协议、音频校验、分段推理及下载中断/校验。GPU 和实际生成需另用 `Check.ps1` 与参考录音验证；GitHub 自动检查不下载 GPU 模型。

上游：[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)。本目录的部署脚本不会自动更新模型或训练权重。
