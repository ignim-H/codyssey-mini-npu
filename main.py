"""Mini NPU Simulator - MAC 연산으로 십자가(Cross) / X 패턴을 판별한다.

실행: python main.py
"""

import json
import math
import os
import sys
from pathlib import Path

from benchmark import average_ms, format_table, measure_sizes
from dataset import evaluate_patterns, load_filters, load_json
from npu_core import CROSS, X
from npu_core import EPSILON_TEXT, UNDECIDED, Matrix, judge, mac
from patterns import make_pattern

DATA_PATH = Path(__file__).resolve().parent / "data.json"  # main.py 와 같은 폴더의 data.json
LINE = "#" + "-" * 40
ECHO_INPUT = os.environ.get("NPU_ECHO_INPUT") == "1"  # 파일로 입력할 때 입력값도 화면에 표시
SIZES = (3, 5, 13, 25)  # 성능 분석 대상 크기

class QuitProgram(Exception):
    """Ctrl+C / 입력 종료(EOF) 시 안전하게 끝내기 위한 신호."""

# ---------------------------------------------------------------- 공통 입출력
def use_utf8_console():
    """Windows 터미널에서도 한글/기호가 깨지지 않도록 입출력을 UTF-8 로 맞춘다."""
    for stream in (sys.stdin, sys.stdout):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

def section(title):
    print()
    print(LINE)
    print(f"# {title}")
    print(LINE)

def read_line(prompt=""):
    """한 줄 입력 (앞뒤 공백 제거). Ctrl+C / EOF 는 QuitProgram 으로 바꾼다."""
    try:
        value = input(prompt)
    except (KeyboardInterrupt, EOFError) as error:
        raise QuitProgram from error
    if ECHO_INPUT:
        print(value)
    return value.strip()

def read_matrix(name, size=3):
    """size 줄을 공백 구분으로 입력받아 Matrix 로 만든다. 형식이 틀린 줄은 다시 입력받는다."""
    print(f"{name} ({size}줄 입력, 공백 구분)")
    rows = []
    while len(rows) < size:
        parts = read_line().split()
        if len(parts) != size:
            print(f"입력 형식 오류: 각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요. "
                  f"(입력한 값 {len(parts)}개, {len(rows) + 1}번째 줄 다시 입력)")
            continue
        try:
            row = [float(part) for part in parts]
        except ValueError:
            print(f"입력 형식 오류: 숫자로 바꿀 수 없는 값이 있습니다. "
                  f"각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요. ({len(rows) + 1}번째 줄 다시 입력)")
            continue
        if not all(math.isfinite(value) for value in row):
            print(f"입력 형식 오류: inf / nan 은 사용할 수 없습니다. ({len(rows) + 1}번째 줄 다시 입력)")
            continue
        rows.append(row)
    return Matrix.from_rows(rows)

def print_matrix(name, matrix):
    print(f"{name}:")
    for line in matrix.to_text().splitlines():
        print(f"  {line}")

def select_mode():
    print()
    print("[모드 선택]")
    print("1. 사용자 입력 (3x3)")
    print("2. data.json 분석")
    print("3. 패턴 생성기 · 최적화 비교 (보너스)")
    print("0. 종료")
    valid = {0, 1, 2, 3}
    while True:
        text = read_line("선택: ")
        if text.isdigit() and int(text) in valid:
            return int(text)
        print(f"⚠️ 잘못된 선택입니다. {', '.join(str(v) for v in sorted(valid))} 중 하나를 입력하세요.")

# ---------------------------------------------------------------- 모드 1: 사용자 입력 (3×3)
def run_user_mode():
    section("[1] 필터 입력")
    filter_a = read_matrix("필터 A")
    print()
    filter_b = read_matrix("필터 B")
    print()
    print("✓ 필터 저장 완료")
    print_matrix("필터 A", filter_a)
    print_matrix("필터 B", filter_b)

    section("[2] 패턴 입력")
    pattern = read_matrix("패턴")

    section("[3] MAC 결과")
    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)
    print(f"A 점수: {score_a!r}")
    print(f"B 점수: {score_b!r}")
    elapsed = average_ms(mac, pattern, filter_a)  # 입출력 제외, MAC 함수 호출 구간만 10회 평균
    print(f"연산 시간(평균/10회): {elapsed:.4f} ms")
    decision = judge(score_a, score_b, "A", "B")
    if decision == UNDECIDED:
        print(f"판정: 판정 불가 (|A-B| < {EPSILON_TEXT})")
    else:
        print(f"판정: {decision}")

    section("[4] 성능 분석 (3×3, 평균/10회)")
    print(format_table(measure_sizes([(3, pattern, filter_a)])))

# ---------------------------------------------------------------- 모드 2: data.json 분석
def run_data_mode():
    section("[1] 필터 로드")
    data, error = load_json(DATA_PATH)
    if error:
        print(f"✗ {error}")
        print("  → 모드 선택으로 돌아갑니다.")
        return
    filters, logs = load_filters(data)
    for _, message in logs:
        print(message)

    section("[2] 패턴 분석 (라벨 정규화 적용)")
    results = evaluate_patterns(data, filters)
    for result in results:
        print(f"--- {result.key} ---")
        if result.cross_score is None:
            print(f"FAIL: {result.reason}")
            continue
        print(f"Cross 점수: {result.cross_score!r}")
        print(f"X 점수: {result.x_score!r}")
        status = "PASS" if result.passed else "FAIL" + (" (동점 규칙)" if result.decision == UNDECIDED else "")
        print(f"판정: {result.decision} | expected: {result.expected} | {status}")

    section("[3] 성능 분석 (평균/10회)")
    bench_cases = build_bench_cases(filters)
    perf_rows = measure_sizes(bench_cases)
    print(format_table(perf_rows))

    section("[4] 결과 요약")
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]
    print(f"총 테스트: {len(results)}개")
    print(f"통과: {len(passed)}개")
    print(f"실패: {len(failed)}개")
    if failed:
        print()
        print("실패 케이스:")
        for result in failed:
            print(f"- {result.key}: {result.reason}")
        print("(상세 원인 분석 및 복잡도 설명은 README.md 의 '결과 리포트' 섹션 참고)")
    save_result_json(results, perf_rows, [])

def build_bench_cases(filters=None):
    """성능 측정용 입력: 생성기로 만든 십자가 패턴 × (data.json 필터, 없으면 생성 필터)."""
    cases = []
    for size in SIZES:
        pattern = make_pattern(CROSS, size)
        if filters and size in filters:
            filt = filters[size][CROSS]
        else:
            filt = make_pattern(CROSS, size)
        cases.append((size, pattern, filt))
    return cases

def save_result_json(results, perf_rows, compare_rows):
    """NPU_RESULT_JSON 환경 변수가 있으면 결과를 JSON 으로 저장한다 (README 결과 리포트 작성용)."""
    path = os.environ.get("NPU_RESULT_JSON")
    if not path:
        return
    data = {
        "cases": [
            {
                "key": r.key, "cross": r.cross_score, "x": r.x_score, "decision": r.decision,
                "expected": r.expected, "passed": r.passed, "reason": r.reason,
            }
            for r in results
        ],
        "perf": [{"size": s, "ms": ms, "ops": ops} for s, ms, ops in perf_rows],
        "compare": [{"size": s, "ms_2d": a, "ms_1d": b, "same": same} for s, a, b, same in compare_rows],
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------- 모드 3: 패턴 생성기 (보너스)
def read_size():
    while True:
        text = read_line("크기 N 입력 (3 이상 99 이하의 홀수): ")
        if not text.isdigit():
            print("⚠️ 숫자를 입력하세요.")
            continue
        size = int(text)
        if size < 3 or size > 99 or size % 2 == 0:
            print("⚠️ 3 이상 99 이하의 홀수만 가능합니다 (가운데 칸이 있어야 십자가/X 가 대칭).")
            continue
        return size

def run_generator_mode():
    section("[1] 패턴 생성")
    size = read_size()
    cross, x_pattern = make_pattern(CROSS, size), make_pattern(X, size)
    if size <= 13:
        print_matrix(f"Cross {size}×{size}", cross)
        print_matrix(f"X {size}×{size}", x_pattern)
    else:
        print(f"(크기 {size}×{size} 는 화면 출력을 생략합니다)")

    section("[2] 생성 패턴 판정 (필터도 생성 패턴 사용)")
    for label, pattern in ((CROSS, cross), (X, x_pattern)):
        cross_score, x_score = mac(pattern, cross), mac(pattern, x_pattern)
        decision = judge(cross_score, x_score, CROSS, X)
        print(f"{label:<5} 패턴 → Cross 점수 {cross_score!r}, X 점수 {x_score!r} → 판정 {decision}")

# ---------------------------------------------------------------- 시작점
def main():
    use_utf8_console()
    print("=== Mini NPU Simulator ===")
    try:
        while True:
            mode = select_mode()
            if mode == 0:
                break
            if mode == 1:
                run_user_mode()
            elif mode == 2:
                run_data_mode()
            elif mode == 3:
                run_generator_mode()
    except QuitProgram:
        print()
        print("⚠️ 입력이 중단되어 프로그램을 종료합니다.")
        return
    print("👋 Mini NPU Simulator 를 종료합니다.")

if __name__ == "__main__":
    main()
