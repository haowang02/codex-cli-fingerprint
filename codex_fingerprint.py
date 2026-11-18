#!/usr/bin/env python3
"""用数字指纹检测本地 Codex CLI 实际调用的模型。"""

from __future__ import annotations

import argparse
import hashlib
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
from typing import List, NamedTuple, Sequence

DATA_FILES = ("bundle.json", "shared_detector.bin", "templates.json")
REMOTE_DATA = "https://raw.githubusercontent.com/haowang02/codex-cli-fingerprint/main/data/"
DETECTOR_SCHEMA = "shared-detector-v2"
SAMPLE_COUNT = 3
MAX_VALUE = 355
PSEUDOCOUNT = 0.5
DISTRIBUTION_WEIGHTS = (0.75, 0.25)
MODEL_ALIASES = {"gpt-6.1-sol": "gpt-6-astra"}

Vector = List[float]


class Challenge(NamedTuple):
    prompt: str
    expected_count: int


class Candidate(NamedTuple):
    model_id: str
    display_name: str
    score: float
    probability: float | None


def read_data() -> dict[str, bytes]:
    local = Path(__file__).resolve().parent / "data"
    if all((local / name).is_file() for name in DATA_FILES):
        return {name: (local / name).read_bytes() for name in DATA_FILES}

    print("正在从 GitHub 下载指纹数据...", flush=True)
    data = {}
    for name in DATA_FILES:
        with urllib.request.urlopen(REMOTE_DATA + name, timeout=120) as response:
            data[name] = response.read()
    return data


def decode_detector(data: bytes) -> dict:
    """读取长度前缀的 JSON 头和按字节平面存放的 float32 参数。"""
    if len(data) < 4:
        raise ValueError("检测器数据不完整")
    (header_length,) = struct.unpack_from("<I", data)
    header_end = 4 + header_length
    start = (header_end + 3) // 4 * 4
    size = len(data) - start
    if size < 0 or size % 4:
        raise ValueError("检测器数据不完整")

    count = size // 4
    raw = bytearray(size)
    for plane in range(4):
        raw[plane::4] = data[start + plane * count:start + (plane + 1) * count]
    values = struct.unpack(f"<{count}f", raw)

    def expand(value):
        if "$f32" not in value:
            return value
        offset = value["$f32"]
        return list(values[offset:offset + value["n"]])

    return json.loads(data[4:header_end], object_hook=expand)


def load_reference() -> tuple[list[dict], dict, list[dict]]:
    data = read_data()
    bundle = json.loads(data["bundle.json"])
    binary = data["shared_detector.bin"]
    if hashlib.sha256(binary).hexdigest() != bundle["files"]["shared_detector.bin"]["sha256"]:
        raise ValueError("检测器文件的 SHA-256 与参考库清单不匹配")

    detector = decode_detector(binary)
    models = bundle["models"]
    if (
        bundle["schema"] != "lmfpd-data-v1"
        or bundle["detector_schema"] != DETECTOR_SCHEMA
        or detector["schema"] != DETECTOR_SCHEMA
        or bundle["built_at"] != detector["bank_built_at"]
        or [model["id"] for model in models] != detector["model_ids"]
        or [model["response_count"] for model in models] != detector["response_counts"]
    ):
        raise ValueError("参考库清单与检测器版本不一致")

    return models, detector, json.loads(data["templates.json"])["languages"]


def generate_challenges(templates: list[dict]) -> list[Challenge]:
    challenges = []
    counts = secrets.SystemRandom().sample(range(292, 333), SAMPLE_COUNT)
    for count in counts:
        language = secrets.choice(templates)
        action = secrets.choice(language["actions"])
        request = language["request"].replace("{action}", action).replace("{count}", str(count))
        prompt = language["space"].join([
            secrets.choice(language["openings"]),
            request,
            language["rules"],
            secrets.choice(language["endings"]),
            secrets.choice(language["separators"]),
            language["start"],
        ])
        challenges.append(Challenge(prompt, count))
    return challenges


def parse_numbers(text: str) -> list[int]:
    """取最长的一段有效整数，数字之间的字母会分隔段落。"""
    best: list[int] = []
    current: list[int] = []
    previous_end = 0
    for match in re.finditer(r"[0-9]+", text):
        if current and any(c.isalpha() for c in text[previous_end:match.start()]):
            if len(current) > len(best):
                best = current
            current = []
        previous_end = match.end()
        digits = match.group().lstrip("0")
        if 1 <= len(digits) <= 3:
            value = int(digits)
            if value <= MAX_VALUE:
                current.append(value)
    return current if len(current) > len(best) else best


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def dot(a: Vector, b: Vector) -> float:
    return sum(map(mul, a, b))


def normalize(values: Vector) -> Vector:
    scale = max(math.sqrt(dot(values, values)), 1e-12)
    return [value / scale for value in values]


def center_and_variance(values: Sequence[float]) -> tuple[Vector, float]:
    center = mean(values)
    deviations = [value - center for value in values]
    return deviations, mean([value ** 2 for value in deviations])


def z_scores(values: Vector) -> Vector:
    deviations, variance = center_and_variance(values)
    scale = max(math.sqrt(variance), 1e-12)
    return [value / scale for value in deviations]


def column_means(rows: list[Vector]) -> Vector:
    return [mean(column) for column in zip(*rows)]


def lag_differences(numbers: list[int], lag: int) -> list[int]:
    return [right - left for left, right in zip(numbers, numbers[lag:])]


def sequence_shape(numbers: list[int]) -> tuple[float, float, float]:
    centered, variance = center_and_variance(numbers)
    autocorrelation = mean([a * b for a, b in zip(centered, centered[1:])]) / variance if variance else 1.0

    steps = [lag_differences(numbers, lag) for lag in (1, 2, 3)]
    close = [sum(abs(step) <= 20 for step in differences) for differences in steps]
    return autocorrelation, close[0] / len(steps[0]), sum(close) / sum(map(len, steps))


def model_like(replies: list[list[int]]) -> bool:
    shapes = [sequence_shape(reply) for reply in replies]
    autocorrelation, small_steps, nearby_pairs = map(median, zip(*shapes))
    return autocorrelation <= -0.23 or small_steps <= 0.05 or nearby_pairs <= 0.07


def hellinger_histogram(indices: Sequence[int], bins: int) -> Vector:
    counts = [PSEUDOCOUNT] * bins
    for index in indices:
        counts[index] += 1
    total = len(indices) + PSEUDOCOUNT * bins
    return [math.sqrt(count / total) for count in counts]


def distribution_features(numbers: list[int]) -> tuple[Vector, Vector]:
    marginal = hellinger_histogram([value - 1 for value in numbers], MAX_VALUE)
    buckets = [(value - 1) * 16 // MAX_VALUE for value in numbers]
    base, remainder = divmod(len(numbers), 4)
    ordered = []
    start = 0
    for index in range(4):
        end = start + base + (index < remainder)
        ordered.extend(hellinger_histogram(buckets[start:end], 16))
        start = end
    ordered.extend(hellinger_histogram([value % 10 for value in numbers], 10))
    return marginal, ordered


def position_kernel(numbers: list[int]) -> Vector:
    count = len(numbers)
    centres = [1 + index * (MAX_VALUE - 1) / 23 for index in range(24)]
    features = [0.0] * ((len(centres) + 10) * 3)
    for position, value in enumerate(numbers):
        t = (position + 0.5) * 2 / count - 1
        basis = (t, 0.5 * (3 * t * t - 1), 0.5 * (5 * t ** 3 - 3 * t))
        weights = [math.exp(-0.5 * ((value - centre) / 20) ** 2) for centre in centres]
        weights.extend(digit == value % 10 for digit in range(10))
        for index, weight in enumerate(weights):
            for degree, coefficient in enumerate(basis):
                features[index * 3 + degree] += weight * coefficient / count
    return features


def centred_transitions(symbols: list[int], bins: int) -> Vector:
    joint = [PSEUDOCOUNT] * (bins * bins)
    for left, right in zip(symbols, symbols[1:]):
        joint[left * bins + right] += 1
    total = len(symbols) - 1 + PSEUDOCOUNT * len(joint)
    joint = [value / total for value in joint]
    rows = [sum(joint[start:start + bins]) for start in range(0, len(joint), bins)]
    columns = [sum(joint[column::bins]) for column in range(bins)]
    return [value - rows[index // bins] * columns[index % bins] for index, value in enumerate(joint)]


def sequence_features(numbers: list[int]) -> Vector:
    centered, variance = center_and_variance(numbers)
    scale = max(math.sqrt(variance), 1)
    standardized = [value / scale for value in centered]
    features = []
    for lag in (1, 2, 3, 5, 8, 13, 21, 34):
        steps = lag_differences(numbers, lag)
        magnitude = [abs(step) for step in steps]
        features.extend((
            mean([a * b for a, b in zip(standardized, standardized[lag:])]),
            mean(magnitude) / MAX_VALUE,
            mean([step > 0 for step in steps]),
            mean([step == 0 for step in steps]),
            mean([step <= 5 for step in magnitude]),
            mean([step <= 20 for step in magnitude]),
        ))

    digits = [value % 10 for value in numbers]
    features.extend(centred_transitions([(value - 1) * 8 // MAX_VALUE for value in numbers], 8))
    features.extend(centred_transitions(digits, 10))

    adjacent = lag_differences(numbers, 1)
    signs = [(step > 0) - (step < 0) for step in adjacent]
    standardized_steps = [abs(b - a) for a, b in zip(standardized, standardized[1:])]
    features.extend((
        mean([a == b for a, b in zip(signs, signs[1:])]),
        mean([a == b for a, b in zip(adjacent, adjacent[1:])]),
        mean([a == b for a, b in zip(digits, digits[1:])]),
        mean([step > 1 for step in standardized_steps]),
        mean([step > 2 for step in standardized_steps]),
    ))

    buckets = [(value - 1) * 16 // MAX_VALUE for value in numbers]
    for width in (4, 8, 16):
        features.append(mean([
            len(set(buckets[start:start + width])) / width
            for start in range(len(buckets) - width + 1)
        ]))
    return features


def rescale(values: Vector, centers: Vector, scales: Vector) -> Vector:
    return [(value - center) / scale for value, center, scale in zip(values, centers, scales)]


def transform(blocks: Sequence[Vector], params: list[dict], weights: Sequence[float]) -> Vector:
    features = []
    for block, param, weight in zip(blocks, params, weights):
        scaled = normalize(rescale(block, param["mean"], param["scale"]))
        scale = math.sqrt(weight)
        features.extend(value * scale for value in scaled)
    return features


def reference_similarity(queries: list[Vector], references: list[list[Vector]]) -> Vector:
    query_norms = [dot(query, query) for query in queries]
    scores = []
    for model_references in references:
        norms = [dot(ref, ref) for ref in model_references]
        nearest_distances = []
        for query, query_norm in zip(queries, query_norms):
            distances = sorted(
                max(0.0, query_norm + norm - 2 * dot(query, ref))
                for norm, ref in zip(norms, model_references)
            )
            nearest_distances.append(mean(distances[:7]))
        scores.append(-median(nearest_distances))
    return scores


def subtract_projection(values: Vector, basis: list[Vector]) -> Vector:
    if not basis:
        return values
    coefficients = [dot(values, row) for row in basis]
    return [value - dot(coefficients, column) for value, column in zip(values, zip(*basis))]


def baseline(blocks: tuple[Vector, Vector], bank: dict) -> Vector:
    marginal_bank = bank["hellinger"]
    ordered_bank = bank["ordered_blocks"]
    marginal = rescale(blocks[0], marginal_bank["feature_mean"], marginal_bank["feature_scale"])
    marginal = normalize(subtract_projection(marginal, marginal_bank["nuisance_basis"]))
    marginal_scores = z_scores([dot(marginal, row) for row in marginal_bank["centroids"]])

    ordered = rescale(blocks[1], ordered_bank["feature_mean"], ordered_bank["feature_scale"])
    raw = normalize(ordered)
    projected = normalize(subtract_projection(ordered, ordered_bank["nuisance_basis"]))
    environments = zip(*ordered_bank["environment_centroids"])
    templates = z_scores([max(dot(raw, row) for row in rows) for rows in environments])
    nuisance = z_scores([dot(projected, row) for row in ordered_bank["centroids"]])
    ordered_scores = z_scores([(a + b) * 0.5 for a, b in zip(templates, nuisance)])
    return [(a + b) * 0.5 for a, b in zip(marginal_scores, ordered_scores)]


def rank(replies: list[list[int]], ranker: dict) -> Vector:
    lda, queries, baselines, positional = [], [], [], []
    head = ranker["positional"]
    for numbers in replies:
        blocks = distribution_features(numbers)
        head_features = distribution_features(numbers[:128])
        head_query = transform(head_features, ranker["head_params"], DISTRIBUTION_WEIGHTS)
        lda.append(z_scores([
            dot(row, head_query) + bias for row, bias in zip(ranker["lda_weights"], ranker["lda_bias"])
        ]))
        queries.append(transform(blocks, ranker["full_params"], DISTRIBUTION_WEIGHTS))
        baselines.append(baseline(blocks, ranker["bank"]))

        positional_blocks = (*blocks, position_kernel(numbers), sequence_features(numbers))
        features = transform(positional_blocks, head["params"], (0.6, 0.15, 0.125, 0.125))
        positional.append(z_scores([
            sum(map(mul, row, features), bias) for row, bias in zip(head["weights"], head["bias"])
        ]))

    discriminant = z_scores(column_means(lda))
    nearest = z_scores(reference_similarity(queries, ranker["references"]))
    baseline_scores = z_scores(column_means(baselines))
    frequency = z_scores([
        0.5 * a + 0.25 * b + 0.25 * c for a, b, c in zip(discriminant, nearest, baseline_scores)
    ])
    position = z_scores(column_means(positional))
    blend = ranker["blend"]
    return [blend["frequency"] * f + blend["positional"] * p for f, p in zip(frequency, position)]


def calibrated_probabilities(scores: Vector, detector: dict) -> Vector | None:
    calibration = detector.get("calibration")
    if not isinstance(calibration, dict):
        return None
    binding = calibration.get("binding")
    tau = calibration.get("tau")
    if (
        calibration.get("schema") != "shared-confidence-v2"
        or calibration.get("method") != "ranking-temperature"
        or not isinstance(binding, dict)
        or not isinstance(tau, (int, float))
        or isinstance(tau, bool)
        or not 0.001 <= tau <= 1000
        or binding.get("base_sha256") != detector["base_sha256"]
        or binding.get("reference_sha256") != detector["source_reference_sha256"]
        or binding.get("model_ids") != detector["model_ids"]
    ):
        return None

    maximum = max(scores)
    weights = [math.exp(tau * (score - maximum)) for score in scores]
    total = sum(weights)
    return [weight / total for weight in weights]


def score_replies(replies: list[list[int]], models: list[dict], detector: dict) -> list[Candidate]:
    scores = rank(replies, detector["ranker"])
    if len(scores) != len(models) or not all(map(math.isfinite, scores)):
        raise ValueError("排名计算产生无效数值")
    probabilities = calibrated_probabilities(scores, detector) if len(replies) == SAMPLE_COUNT else None
    candidates = [
        Candidate(model["id"], model["display_name"], score, probability)
        for model, score, probability in zip(models, scores, probabilities or [None] * len(scores))
    ]
    return sorted(candidates, key=lambda candidate: -candidate.score)


def resolve_codex_executable() -> str:
    # Windows 的 npm 安装依赖带扩展名的 shim。
    for name in ("codex.cmd", "codex.exe", "codex") if os.name == "nt" else ("codex",):
        executable = shutil.which(name)
        if executable:
            return executable
    raise RuntimeError("找不到 codex，请确认已安装并加入 PATH")


def run_codex(executable: str, prompt: str, model: str, effort: str) -> str:
    command = [
        executable, "exec", "--json", "--skip-git-repo-check", "-s", "read-only",
        "--disable", "memories", "-c", f"model_reasoning_effort={effort}", "-c", "notify=[]", "-m", model,
    ]
    process = subprocess.run(
        command, input=prompt, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip() or process.stdout.strip() or "codex exec 执行失败")
    answer = ""
    for line in process.stdout.splitlines():
        if line.startswith("{"):
            event = json.loads(line)
            item = event.get("item") or {}
            if event.get("type") == "item.completed" and item.get("type") == "agent_message":
                answer = item.get("text", "")
    return answer


def collect_replies(executable: str, challenges: list[Challenge], model: str, effort: str) -> list[list[int]]:
    def sample(prompt: str) -> tuple[list[int], float]:
        started = time.monotonic()
        numbers = parse_numbers(run_codex(executable, prompt, model, effort))
        return numbers, time.monotonic() - started

    row = "{:>1}  {:>4}  {:>4}  {:>7}  {}".format
    print(row("#", "Want", "Got", "Time(s)", "OK"))
    print(row("-", "----", "----", "-------", "--"), flush=True)
    replies, errors = {}, {}
    with ThreadPoolExecutor(len(challenges)) as pool:
        futures = {
            pool.submit(sample, challenge.prompt): (index, challenge.expected_count)
            for index, challenge in enumerate(challenges, 1)
        }
        for future in as_completed(futures):
            index, want = futures[future]
            try:
                numbers, seconds = future.result()
            except (OSError, RuntimeError, ValueError) as exc:
                errors[index] = " ".join(str(exc).split())
                print(row(index, want, "-", "-", "✗"), flush=True)
                continue

            minimum = max(80, math.ceil(want * 0.55))
            ok = len(numbers) >= minimum
            if ok:
                replies[index] = numbers[:want]
            else:
                errors[index] = f"有效整数仅 {len(numbers)} 个，至少需要 {minimum} 个"
            print(row(index, want, len(numbers), f"{seconds:.1f}", "✓" if ok else "✗"), flush=True)

    for index, message in sorted(errors.items()):
        print(f"#{index}: {message}")
    return [replies[index] for index in sorted(replies)]


def print_ranking(candidates: list[Candidate], model: str) -> None:
    shown = candidates[:5]
    width = max(len(candidate.display_name) for candidate in shown)
    print(f"#  {'Model':<{width}}  {'Score':>7}  {'Confidence':>10}")
    for place, candidate in enumerate(shown, 1):
        probability = candidate.probability
        confidence = f"{probability:.1%}" if probability is not None else "-"
        print(f"{place}  {candidate.display_name:<{width}}  {candidate.score:>7.3f}  {confidence:>10}")

    print()
    if MODEL_ALIASES.get(model, model) == candidates[0].model_id:
        print("\033[32m✓ 与测试模型一致\033[0m")
    else:
        print("\033[31m✗ 与测试模型不一致\033[0m")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model", required=True, help="要测试的 Codex 模型名")
    parser.add_argument(
        "-r", "--reasoning-effort", default="low", help="推理强度，默认 low，与参考库中 GPT 样本一致",
    )
    args = parser.parse_args()

    try:
        executable = resolve_codex_executable()
        models, detector, templates = load_reference()
        replies = collect_replies(executable, generate_challenges(templates), args.model, args.reasoning_effort)
        print()
        if not replies:
            print("测试失败：没有有效回答")
        elif not model_like(replies):
            print("测试失败：不像语言模型的回答，近邻数字的变化与参考库不符")
        else:
            print_ranking(score_replies(replies, models, detector), args.model)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"测试失败：{exc}\n")


if __name__ == "__main__":
    main()
