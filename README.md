# Codex CLI 指纹检测

检测本地 Codex CLI 实际调用的模型。脚本让模型凭第一反应写出约 300 个 1–355 的整数，再与参考库中的 72 个指纹类别比对。脚本只使用 Python 3 标准库。指纹数据和检测算法来自 [lm-detector](https://github.com/Ikaleio/lm-detector/tree/33fb43f26670be0ed7817ab13e5c4c4731ccb76a)。

<img src="./example.png" alt="用 GPT-6.2 Sol 测试的指纹检测结果" width="760">

## 用法

需要先安装 [Codex CLI](https://github.com/openai/codex)。

```bash
python3 codex_fingerprint.py -m gpt-6.2-sol
```

### 一键运行

以下任选其一，指纹数据会从本仓库下载：

```bash
curl -fsSL https://is.gd/codex_fingerprint | python3 - -m gpt-6.2-sol
```

```bash
wget -qO- https://is.gd/codex_fingerprint | python3 - -m gpt-6.2-sol
```

参数：

- `-m, --model`：要测试的 Codex 模型名（必填）
- `-r, --reasoning-effort`：推理强度，默认 `low`，与参考库中 GPT 样本的采集条件一致

每次运行并行发送 3 个挑战，列出参考库中得分最高的 5 个候选，并用第一名判断是否与测试模型一致。三条回答都有效时才会显示置信度。

## 致谢

- [lm-detector](https://github.com/Ikaleio/lm-detector) 提供指纹数据和检测算法，见 [MIT 许可](LICENSE.lm-detector)
- [LINUX DO](https://linux.do/) - 新的理想型社区
