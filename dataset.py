"""data.json 로드 + 스키마 검증 + 케이스별 판정. 오류가 있어도 프로그램을 멈추지 않고 케이스 단위로 FAIL 처리한다."""

import json
import re
from pathlib import Path

from npu_core import CROSS, X, Matrix, judge, mac, normalize_label

FILTER_KEY = re.compile(r"^size_(\d+)$")           # 예: size_13
PATTERN_KEY = re.compile(r"^size_(\d+)_(\d+)$")    # 예: size_13_2  → N=13, idx=2

class CaseResult:
    """패턴 1개의 판정 결과."""

    def __init__(self, key, size=None):
        self.key = key
        self.size = size
        self.cross_score = None
        self.x_score = None
        self.decision = None   # Cross / X / UNDECIDED
        self.expected = None   # 정규화된 expected (Cross / X)
        self.passed = False
        self.reason = ""       # FAIL 사유 요약

    def fail(self, reason):
        self.passed = False
        self.reason = reason
        return self

def load_json(path):
    """(데이터, 오류 메시지) 를 돌려준다. 파일 없음/JSON 오류도 예외 대신 메시지로."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        return None, f"{path.name} 파일을 찾을 수 없습니다 (위치: main.py 와 같은 폴더)"
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        return None, f"{path.name} 이(가) 올바른 JSON 이 아닙니다: {error}"
    except OSError as error:
        return None, f"{path.name} 을(를) 읽을 수 없습니다: {error}"
    if not isinstance(data, dict):
        return None, "최상위 구조가 객체({ })가 아닙니다"
    return data, None

def load_filters(data):
    """filters 를 읽어 {N: {"Cross": Matrix, "X": Matrix}} 와 (N, 메시지) 로그 목록을 만든다."""
    filters, logs = {}, []
    raw_filters = data.get("filters")
    if not isinstance(raw_filters, dict):
        logs.append((None, "✗ filters 항목이 없거나 객체가 아닙니다"))
        return filters, logs

    for key, entry in raw_filters.items():
        match = FILTER_KEY.match(key)
        if not match:
            logs.append((None, f"✗ {key}: 키 형식 오류 (size_N 이어야 함)"))
            continue
        size = int(match.group(1))
        if not isinstance(entry, dict):
            logs.append((size, f"✗ {key}: 필터 묶음이 객체가 아닙니다"))
            continue
        pair = {}
        problems = []
        for label_key, rows in entry.items():
            try:
                label = normalize_label(label_key)        # 'cross' → Cross, 'x' → X
                matrix = Matrix.from_rows(rows)
                if matrix.size != size:
                    raise ValueError(f"키는 {size}×{size} 인데 실제 {matrix.size}×{matrix.size}")
                pair[label] = matrix
            except ValueError as error:
                problems.append(f"{label_key}: {error}")
        missing = [label for label in (CROSS, X) if label not in pair]
        if missing:
            problems.append(f"{', '.join(missing)} 필터 없음")
        if problems:
            logs.append((size, f"✗ {key:<8} 필터 로드 실패 - " + "; ".join(problems)))
            continue
        filters[size] = pair
        logs.append((size, f"✓ {key:<8} 필터 로드 완료 (Cross, X)"))
    logs.sort(key=lambda item: (item[0] is None, item[0] or 0))
    return filters, logs

def _pattern_sort_key(key):
    match = PATTERN_KEY.match(key)
    return (int(match.group(1)), int(match.group(2))) if match else (10**9, 0)

def evaluate_patterns(data, filters):
    """patterns 의 각 케이스를 검증하고 판정한다. 어떤 오류가 나도 해당 케이스만 FAIL."""
    results = []
    raw_patterns = data.get("patterns")
    if not isinstance(raw_patterns, dict):
        results.append(CaseResult("patterns").fail("patterns 항목이 없거나 객체가 아닙니다"))
        return results

    for key in sorted(raw_patterns, key=_pattern_sort_key):
        entry = raw_patterns[key]
        match = PATTERN_KEY.match(key)
        if not match:
            results.append(CaseResult(key).fail("키 형식 오류 (size_N_idx 이어야 함)"))
            continue
        size = int(match.group(1))
        result = CaseResult(key, size)
        if not isinstance(entry, dict) or "input" not in entry or "expected" not in entry:
            results.append(result.fail("input / expected 항목 누락"))
            continue
        try:
            result.expected = normalize_label(entry["expected"])   # '+' → Cross, 'x' → X
        except ValueError as error:
            results.append(result.fail(f"expected 라벨 오류 - {error}"))
            continue
        if size not in filters:
            results.append(result.fail(f"size_{size} 필터가 없거나 로드 실패"))
            continue
        try:
            pattern = Matrix.from_rows(entry["input"])
        except ValueError as error:
            results.append(result.fail(f"input 형식 오류 - {error}"))
            continue
        if pattern.size != size:
            results.append(result.fail(f"크기 불일치 - 키는 {size}×{size}, input 은 {pattern.size}×{pattern.size}"))
            continue

        result.cross_score = mac(pattern, filters[size][CROSS])
        result.x_score = mac(pattern, filters[size][X])
        result.decision = judge(result.cross_score, result.x_score, CROSS, X)
        result.passed = result.decision == result.expected
        if not result.passed:
            if result.decision == "UNDECIDED":
                gap = abs(result.cross_score - result.x_score)
                result.reason = f"동점(UNDECIDED) 처리 규칙에 따라 FAIL (|Cross-X| = {gap:.3g} < 1e-9)"
            else:
                result.reason = f"판정 {result.decision} ≠ expected {result.expected}"
        results.append(result)
    return results
