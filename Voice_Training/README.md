# 通用角色声音训练

本项目可单独部署，导入新角色的录音后，完成数据检查、音频编码、老师模型缓存、LoRA 蒸馏训练、模型导出和试听。使用自己的 Python 内置 venv，不依赖部署项目或 Conda。

## 安装

在本文件所在目录打开 Windows PowerShell：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ./Setup.ps1
```

部署自动下载 Python、锁定依赖、Qwen 0.6B 学生模型和 1.7B 老师模型，并检查 GPU，不会开始训练。需要 Windows x64、支持 PyTorch CUDA 12.8 的 NVIDIA 驱动；无需单独安装 CUDA Toolkit。两个基础模型合计约 6.57 GiB，还需环境、下载缓存、数据和训练产物空间，建议先预留 30 GiB 以上，训练中持续留意剩余容量和显存。未测定所有录音长度的显存下限。

依赖列表与工具地址见 `deployment/dependencies.json`，模型地址、大小及 SHA256 见 `deployment/models.json`。已安装完整时可用 `./Setup.ps1 -Offline` 复核，或运行 `./Check.ps1`。

## 导入自己的角色

把录音放在本项目 `data/raw/my_character/`，在同目录建立 UTF-8 的 `manifest.jsonl`，每行一个对象：

```jsonl
{"id":"train_001","audio":"train_001.wav","text":"与第一条录音完全一致的台词。","split":"train","reviewed":true}
{"id":"train_002","audio":"train_002.wav","text":"与第二条录音完全一致的台词。","split":"train","reviewed":true}
{"id":"validation_001","audio":"validation_001.wav","text":"只用于验证的独立录音台词。","split":"validation","reviewed":true}
```

`audio` 相对于这个 JSONL 所在目录，必须留在本项目内。`id` 用英文、数字、下划线或连字符。`reviewed` 表示是否人工听过并核对，不会代替人工检查。示例仅说明格式，不包含可直接训练的录音。

```powershell
./New-Character.ps1 -Name my_character -Manifest data/raw/my_character/manifest.jsonl -Language Chinese -ReferenceId train_001
./Train.ps1 -Stage validate -Config configs/characters/my_character.json
```

至少两条训练录音和一条独立验证录音；这是程序接受数据的最低数量，不是获得满意音色的推荐数据量。每条 0.25～30 秒，不会自动截断未对齐音频；重复录音、验证集参考音频、路径越界和覆盖已有角色会被拒绝。每个角色单独保存配置、编码、缓存、检查点及导出模型。

可选语言：Chinese、English、Japanese、Korean、German、French、Russian、Portuguese、Spanish、Italian。训练需明确指定语言。

## 训练、继续与试听

先小步验证流程，再决定正式训练参数：

```powershell
./Train.ps1 -Stage all -Config configs/characters/my_character.json -MaxSteps 1
```

`all` 依次准备数据、生成老师缓存、训练并尝试导出最佳模型。首次老师缓存可能耗时较长；`-MaxSteps` 只限制优化步数，数据编码和老师缓存仍需完成。默认导出要求存在有效训练步；未得到最佳检查点时不会伪造训练成果。

分阶段、恢复及试听：

```powershell
./Train.ps1 -Stage prepare -Config configs/characters/my_character.json
./Train.ps1 -Stage teacher -Config configs/characters/my_character.json
./Train.ps1 -Stage train -Config configs/characters/my_character.json -Resume
./Train.ps1 -Stage export -Config configs/characters/my_character.json
./Speak.ps1 -Config configs/characters/my_character.json -Text '这是一段用于试听的新台词。' -Output outputs/my_character_preview.wav
```

第一次训练不要加 `-Resume`。断点恢复会检查数据及训练配置是否一致。学习率、轮次、LoRA 等设置集中在生成的角色 JSON；模板为 `configs/training.example.json`。数据或模型发生变化时，旧缓存会被拒绝。

默认输出：`checkpoints/my_character/` 为训练检查点，`models/characters/my_character/` 为导出的角色模型，`outputs/` 为试听音频。导出是独立的 Qwen custom_voice 模型；不要直接把部署项目的 Base 模型 ID 改指向它，该项目的参考录音克隆接口与此处的 custom_voice 接口不同。

## 结构与调用关系

```text
PowerShell 入口 -> train.py -> src/voice_training/cli.py
CLI -> features/datasets.py                 导入角色
CLI -> features/preparation.py              数据编码
CLI -> features/teacher.py                  冻结老师、生成分布缓存
CLI -> features/optimization.py             训练与验证
CLI -> features/exporting.py / preview.py   合并导出与试听
common/paths.py、io.py、state.py             地址簿、文件读写、版本校验
common/lora.py、tensor_ops.py               共享模型与张量操作
```

功能模块通过 CLI 协调，只依赖公共层。JSONL 转为角色 profiles/config，再生成带数据与模型指纹的编码/老师缓存，训练生成检查点，最后合并学生权重并写入角色信息。训练使用 16 组音频单元、LoRA 和老师分布蒸馏；基础模型文件不会被训练直接覆盖。

## 验证和仓库卫生

```powershell
./.venv/Scripts/python.exe -B -m unittest discover -s tests -v
```

测试覆盖角色隔离、重复录音、路径限制、LoRA 梯度与导出校验。训练方案是实验性的；真实模型的梯度检查不能代替真实录音的完整训练、独立验证与音质试听。

环境、模型、原始录音、`configs/characters/`、训练缓存、检查点、生成音频和日志均被 Git 忽略，代码仓库不保存这些内容。请单独备份训练数据、角色配置和训练成果。移动项目后重建 `.venv`。

上游模型与接口：[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)。
