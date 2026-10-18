# 一键部署配置

从项目根目录运行 `powershell -NoProfile -ExecutionPolicy Bypass -File .\Setup.ps1`。

`dependencies.json` 是 Python、包版本、GPU wheel、uv 和 SoX 下载的唯一依赖配置。
`models.json` 固定各模型文件的下载地址、字节数与 SHA256。路径均相对于项目根目录。
模型默认随部署自动下载；已有且校验通过的文件会复用。下载中断后重新执行即可继续模型下载。
Python 由 uv 准备到项目内，虚拟环境明确通过 Python 标准库 `python -m venv` 创建。
uv 仅用于准备解释器和安装完整锁定列表；CUDA wheel 通过 venv 内 pip 安装。

维护版本时同时更新包版本与对应 wheel 地址、大小和校验和。不要把本机绝对路径写入 JSON。
本目录脚本独立于其他仓库；`-Offline` 用于已有完整本地文件的验证，默认执行联网部署。
