#!/usr/bin/env python3
"""Codex CLI 模型指纹检测，指纹数据与算法来自 lm-detector
    python3 codex_fingerprint.py -m gpt-6.1-sol

让模型凭第一反应写出约 300 个 1–355 的整数，将数字分布与参考库比对，列出最接近的 GPT 模型。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import secrets
import shutil
import struct
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from operator import mul
from pathlib import Path
from statistics import median

# data/ 中的文件原样取自 lm-detector（https://github.com/Ikaleio/lm-detector）：
#   bundle.json          参考库模型列表（https://lm.ikale.io/data/bundle.json）
#   shared_detector.bin  检测器参数（data/shared_detector.bin）
#   templates.json       挑战提示词模板（sdk/conformance/templates.json）
# 本地运行时读取脚本同目录下的 data/，通过管道运行时从 GitHub 下载。
DATA_FILES = ("bundle.json", "shared_detector.bin", "templates.json")
REMOTE_DATA = "https://raw.githubusercontent.com/haowang02/codex-cli-fingerprint/main/data/"

# 参考库中 gpt-6.1-sol 与 gpt-6-astra 共用一个指纹，其余模型名即参考库中的模型 ID
MODEL_ALIASES = {"gpt-6.1-sol": "gpt-6-astra"}


def read_data() -> dict[str, bytes]:
    script = globals().get("__file__")
    local = Path(script).resolve().parent / "data" if script else None
    if local and all((local / name).is_file() for name in DATA_FILES):
        return {name: (local / name).read_bytes() for name in DATA_FILES}
    print("正在从 GitHub 下载指纹数据...", flush=True)
    return {name: urllib.request.urlopen(REMOTE_DATA + name, timeout=120).read() for name in DATA_FILES}


def decode_detector(data: bytes) -> dict:
    """解码 shared_detector.bin：4 字节头长度、JSON 头、4 字节对齐的 float32 数据。
    float32 按字节平面存放（先存所有值的第 1 个字节，依此类推），头中的数组记为 {"$f32": 偏移, "n": 长度}。"""
    (header_length,) = struct.unpack_from("<I", data, 0)
    start = (4 + header_length + 3) // 4 * 4
    count = (len(data) - start) // 4
    raw = bytearray(count * 4)
    for byte in range(4):
        raw[byte::4] = data[start + byte * count:start + (byte + 1) * count]
    values = struct.unpack(f"<{count}f", raw)

    def expand(value: dict):
        return list(values[value["$f32"]:value["$f32"] + value["n"]]) if "$f32" in value else value

    return json.loads(data[4:4 + header_length].decode("utf-8"), object_hook=expand)


def load_reference():
    data = read_data()
    models = json.loads(data["bundle.json"])["models"]
    detector = decode_detector(data["shared_detector.bin"])
    if [m["id"] for m in models] != detector["model_ids"]:
        raise RuntimeError("bundle.json 与 shared_detector.bin 版本不一致")
    return models, detector, json.loads(data["templates.json"])["languages"]


def generate_challenges(templates: list[dict]) -> list[dict]:
    """生成 3 个挑战：各自随机选一种语言，要求的整数个数取 292–332 中互不相同的值。"""
    available = list(range(292, 333))
    challenges = []
    for _ in range(3):
        count = available.pop(secrets.randbelow(len(available)))
        lang = secrets.choice(templates)
        request = lang["request"].replace("{action}", secrets.choice(lang["actions"])).replace("{count}", str(count))
        prompt = lang["space"].join([secrets.choice(lang["openings"]), request, lang["rules"],
                                     secrets.choice(lang["endings"]), secrets.choice(lang["separators"]), lang["start"]])
        challenges.append({"expected_count": count, "prompt": prompt})
    return challenges


# ---------------------------------------------------------------- 特征提取（移植自 shared/fingerprint-core.js、shared/positional.ts）

def parse_numbers(text: str) -> list[int]:
    """提取回答中最长的一段 1–355 的整数；两个数字之间出现字母时另起一段。"""
    best: list[int] = []
    current: list[int] = []
    previous_end = 0
    for match in re.finditer(r"[0-9]+", text):  # 只匹配 ASCII 数字
        if current and any(c.isalpha() for c in text[previous_end:match.start()]):
            best = current if len(current) > len(best) else best
            current = []
        value = int(match.group())
        if 1 <= value <= 355:
            current.append(value)
        previous_end = match.end()
    return current if len(current) > len(best) else best


def mean(values) -> float:
    return sum(values) / len(values)


def dot(a, b) -> float:
    return sum(map(mul, a, b))


def unit(values: list[float]) -> list[float]:
    scale = max(math.sqrt(dot(values, values)), 1e-12)
    return [v / scale for v in values]


def z(values: list[float]) -> list[float]:
    center = mean(values)
    scale = max(math.sqrt(mean([(v - center) ** 2 for v in values])), 1e-12)
    return [(v - center) / scale for v in values]


def column_mean(rows: list[list[float]]) -> list[float]:
    return [mean(column) for column in zip(*rows)]


def hellinger_feature(numbers: list[int]) -> list[float]:
    counts = [0] * 355
    for value in numbers:
        counts[value - 1] += 1
    total = len(numbers) + 0.5 * 355
    return [math.sqrt((c + 0.5) / total) for c in counts]


def ordered_block_feature(numbers: list[int]) -> list[float]:
    """按顺序均分为四段，每段统计 16 档取值直方图，再加上个位数直方图，均取平方根缩放。"""
    base, remainder = divmod(len(numbers), 4)
    pieces: list[float] = []
    start = 0
    for index in range(4):
        size = base + (1 if index < remainder else 0)
        bins = [0.5] * 16
        for value in numbers[start:start + size]:
            bins[min(15, math.floor((value - 1) / 355 * 16))] += 1
        pieces += [math.sqrt(b / sum(bins)) for b in bins]
        start += size
    digits = [0.5] * 10
    for value in numbers:
        digits[value % 10] += 1
    return pieces + [math.sqrt(d / sum(digits)) for d in digits]


def position_kernel(x: list[int]) -> list[float]:
    n = len(x)
    centres = [1 + i * 354 / 23 for i in range(24)]
    out = [0.0] * (34 * 3)
    for position, value in enumerate(x):
        t = (position + 0.5) * 2 / n - 1
        basis = (t, 0.5 * (3 * t * t - 1), 0.5 * (5 * t ** 3 - 3 * t))
        features = [math.exp(-0.5 * ((value - centre) / 20) ** 2) for centre in centres]
        features += [1 if digit == value % 10 else 0 for digit in range(10)]
        for k, weight in enumerate(features):
            for j, b in enumerate(basis):
                out[k * 3 + j] += weight * b / n
    return out


def centred_transitions(symbols: list[int], bins: int) -> list[float]:
    joint = [0.5] * (bins * bins)
    for a, b in zip(symbols, symbols[1:]):
        joint[a * bins + b] += 1
    total = len(symbols) - 1 + 0.5 * bins * bins
    joint = [v / total for v in joint]
    rows = [sum(joint[a * bins:a * bins + bins]) for a in range(bins)]
    columns = [sum(joint[a * bins + b] for a in range(bins)) for b in range(bins)]
    return [v - rows[i // bins] * columns[i % bins] for i, v in enumerate(joint)]


def sequence_features(x: list[int]) -> list[float]:
    n = len(x)
    m = mean(x)
    sd = max(math.sqrt(mean([(v - m) ** 2 for v in x])), 1)
    s = [(v - m) / sd for v in x]
    out: list[float] = []
    for lag in (1, 2, 3, 5, 8, 13, 21, 34):
        count = n - lag
        d = [x[i + lag] - x[i] for i in range(count)]
        out += [mean([s[i + lag] * s[i] for i in range(count)]), mean([abs(v) for v in d]) / 355,
                sum(v > 0 for v in d) / count, sum(v == 0 for v in d) / count,
                sum(abs(v) <= 5 for v in d) / count, sum(abs(v) <= 20 for v in d) / count]
    out += centred_transitions([min(7, math.floor((v - 1) * 8 / 355)) for v in x], 8)
    out += centred_transitions([v % 10 for v in x], 10)
    difference = [x[i + 1] - x[i] for i in range(n - 1)]
    sign = [(d > 0) - (d < 0) for d in difference]
    z_step = [abs(s[i + 1] - s[i]) for i in range(n - 1)]
    out += [sum(sign[i + 1] == sign[i] for i in range(n - 2)) / (n - 2),
            sum(difference[i + 1] == difference[i] for i in range(n - 2)) / (n - 2),
            sum(x[i] % 10 == x[i + 1] % 10 for i in range(n - 1)) / (n - 1),
            sum(v > 1 for v in z_step) / (n - 1), sum(v > 2 for v in z_step) / (n - 1)]
    buckets = [min(15, math.floor((v - 1) * 16 / 355)) for v in x]
    for width in (4, 8, 16):
        total = 0.0
        for i in range(n - width + 1):
            total += len(set(buckets[i:i + width])) / width
        out.append(total / (n - width + 1))
    return out


# ---------------------------------------------------------------- 排名与置信度（移植自 shared/shared-detector.ts）

def transform(numbers: list[int], params: list[dict]) -> list[float]:
    out: list[float] = []
    for block, p, weight in zip((hellinger_feature(numbers), ordered_block_feature(numbers)), params, (0.75, 0.25)):
        out += [v * math.sqrt(weight) for v in unit([(v - m) / s for v, m, s in zip(block, p["mean"], p["scale"])])]
    return out


def nearest(x: list[float], references: list[list[float]]) -> float:
    """与某个模型最近的 7 个参考向量的平均平方距离。"""
    xx = dot(x, x)
    return mean(sorted(max(0, xx + dot(ref, ref) - 2 * dot(x, ref)) for ref in references)[:7])


def subtract(values: list[float], basis: list[list[float]]) -> list[float]:
    coefficients = [dot(values, row) for row in basis]
    return [v - sum(c * row[i] for c, row in zip(coefficients, basis)) for i, v in enumerate(values)]


def baseline(numbers: list[int], bank: dict) -> list[float]:
    h, o = bank["hellinger"], bank["ordered_blocks"]

    def centered(block, p):
        return [(v - m) / s for v, m, s in zip(block, p["feature_mean"], p["feature_scale"])]

    h_unit = unit(subtract(centered(hellinger_feature(numbers), h), h["nuisance_basis"]))
    marginal = [dot(h_unit, row) for row in h["centroids"]]
    o_centered = centered(ordered_block_feature(numbers), o)
    raw, projected = unit(o_centered), unit(subtract(o_centered, o["nuisance_basis"]))
    templates = [max(dot(raw, rows[i]) for rows in o["environment_centroids"]) for i in range(len(o["centroids"]))]
    nuisance = [dot(projected, row) for row in o["centroids"]]
    ordered = z([0.5 * a + 0.5 * b for a, b in zip(z(templates), z(nuisance))])
    return [0.5 * a + 0.5 * b for a, b in zip(z(marginal), ordered)]


def positional(numbers: list[int], head: dict) -> list[float]:
    blocks = (hellinger_feature(numbers), ordered_block_feature(numbers), position_kernel(numbers),
              sequence_features(numbers))
    x: list[float] = []
    for block, p, weight in zip(blocks, head["params"], (0.6, 0.15, 0.125, 0.125)):
        scaled = [(v - m) / s for v, m, s in zip(block, p["mean"], p["scale"])]
        length = max(math.sqrt(sum(v * v for v in scaled)), 1e-12)
        x += [v / length * math.sqrt(weight) for v in scaled]
    return [sum(map(mul, row, x), bias) for row, bias in zip(head["weights"], head["bias"])]


def rank(replies: list[list[int]], r: dict) -> list[float]:
    """计算每个参考模型的排名分数。"""
    lda = []
    for numbers in replies:
        x = transform(numbers[:128], r["head_params"])
        lda.append(z([dot(row, x) + bias for row, bias in zip(r["lda_weights"], r["lda_bias"])]))
    full = [transform(numbers, r["full_params"]) for numbers in replies]
    near = z([median([-nearest(x, references) for x in full]) for references in r["references"]])
    base = z(column_mean([baseline(numbers, r["bank"]) for numbers in replies]))
    frequency = z([0.5 * a + 0.25 * b + 0.25 * c for a, b, c in zip(z(column_mean(lda)), near, base)])
    position = z(column_mean([z(positional(numbers, r["positional"])) for numbers in replies]))
    return [r["blend"]["frequency"] * f + r["blend"]["positional"] * p for f, p in zip(frequency, position)]


def model_like(replies: list[list[int]]) -> bool:
    """语言模型的回答中，相邻数字通常呈负自相关，且相差 20 以内的比例很低；两项都不符合时判定为非模型回答。"""
    autocorrelations, small_steps = [], []
    for n in replies:
        m = mean(n)
        variance = mean([(v - m) ** 2 for v in n])
        autocorrelations.append(mean([(b - m) * (a - m) for a, b in zip(n, n[1:])]) / variance if variance else 1)
        small_steps.append(mean([abs(b - a) <= 20 for a, b in zip(n, n[1:])]))
    return median(autocorrelations) <= -0.23 or median(small_steps) <= 0.05


def analyze(replies: list[list[int]], detector: dict):
    """返回排名分数和置信度。置信度由温度 tau 校准，仅在 3 条回答均有效时计算。"""
    ranking = rank(replies, detector["ranker"])
    if len(replies) < 3:
        return ranking, None
    logits = [detector["calibration"]["tau"] * score for score in ranking]
    weights = [math.exp(v - max(logits)) for v in logits]
    return ranking, [w / sum(weights) for w in weights]


# ---------------------------------------------------------------- Codex

def resolve_codex_executable() -> str:
    """Windows 上无扩展名的 npm shim 无法直接启动，优先使用 codex.cmd。"""
    for name in ("codex.cmd", "codex.exe", "codex") if os.name == "nt" else ("codex",):
        exe = shutil.which(name)
        if exe:
            return exe
    raise RuntimeError("找不到 codex，请确认已安装并加入 PATH")


def run_codex(prompt: str, model: str | None, effort: str) -> str:
    cmd = [
        resolve_codex_executable(), "exec", "--json",
        "--skip-git-repo-check",
        "-s", "read-only",
        "--disable", "memories",  # 关闭跨会话记忆，避免历史内容混入上下文
        "-c", f"model_reasoning_effort={effort}",
        "-c", "notify=[]",
    ]
    if model:
        cmd += ["-m", model]
    # 提示词经 stdin 传入，避免 Windows 下 codex.cmd 改写命令行参数
    proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "codex exec 执行失败")
    text = ""
    for line in proc.stdout.splitlines():  # 取最后一条 agent_message 作为回答
        if line.startswith("{"):
            event = json.loads(line)
            item = event.get("item") or {}
            if event.get("type") == "item.completed" and item.get("type") == "agent_message":
                text = item.get("text", "")
    return text


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-m", "--model", required=True, help="要测试的 Codex 模型名")
    parser.add_argument("-r", "--reasoning-effort", default="low", help="推理强度，默认 low，与参考库中 GPT 样本一致")
    args = parser.parse_args()

    models, detector, templates = load_reference()
    challenges = generate_challenges(templates)

    def run_one(prompt: str):
        started = time.monotonic()
        return parse_numbers(run_codex(prompt, args.model, args.reasoning_effort)), time.monotonic() - started

    row = "{:>1}  {:>4}  {:>4}  {:>7}  {}".format
    print(row("#", "Want", "Got", "Time(s)", "OK"))
    print(row("-", "----", "----", "-------", "--"))
    replies, errors = [], []
    with ThreadPoolExecutor(len(challenges)) as pool:
        futures = {pool.submit(run_one, c["prompt"]): (i, c["expected_count"]) for i, c in enumerate(challenges, 1)}
        for future in as_completed(futures):
            index, want = futures[future]
            try:
                numbers, seconds = future.result()
            except Exception as exc:
                errors.append(f"#{index}: {' '.join(str(exc).split())}")
                print(row(index, want, "-", "-", "✗"), flush=True)
                continue
            # 多出的整数截掉，不足下限的回答视为无效
            minimum = max(80, math.ceil(want * 0.55))
            ok = len(numbers) >= minimum
            if ok:
                replies.append(numbers[:want])
            else:
                errors.append(f"#{index}: 有效整数仅 {len(numbers)} 个，至少需要 {minimum} 个")
            print(row(index, want, len(numbers), f"{seconds:.1f}", "✓" if ok else "✗"), flush=True)
    for error in sorted(errors):
        print(error)

    print()
    if not replies:
        print("测试失败：没有有效回答")
        return
    if not model_like(replies):
        print("测试失败：不像语言模型的回答，相邻数字的变化规律与参考库中的模型回答不符")
        return
    ranking, probabilities = analyze(replies, detector)
    order = sorted(range(len(models)), key=lambda i: -ranking[i])
    # 排名在全部参考模型上计算，只列出前 5 名 GPT 模型
    shown = [i for i in order if models[i]["family"] == "gpt"][:5]
    width = max(len(models[i]["display_name"]) for i in shown)
    print(f"{'#':>2}  {'Model':<{width}}  {'Score':>7}  {'Confidence':>10}")
    for place, i in enumerate(shown, 1):
        confidence = f"{probabilities[i] * 100:.1f}%" if probabilities else "-"
        print(f"{place:>2}  {models[i]['display_name']:<{width}}  {ranking[i]:>7.3f}  {confidence:>10}")

    # 归因模型取全部参考模型中的第一名
    print()
    if MODEL_ALIASES.get(args.model, args.model) == models[order[0]]["id"]:
        print("\033[32m✓ 与测试模型一致\033[0m")
    else:
        print("\033[31m✗ 与测试模型不一致\033[0m")


if __name__ == "__main__":
    main()
