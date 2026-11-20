"""Mini NPU Simulator - MAC 연산으로 십자가(Cross) / X 패턴을 판별한다.

실행: python main.py
"""

import math
import os
import sys

from npu_core import EPSILON_TEXT, UNDECIDED, Matrix, judge, mac

LINE = "#" + "-" * 40
ECHO_INPUT = os.environ.get("NPU_ECHO_INPUT") == "1"  # 파일로 입력할 때 입력값도 화면에 표시

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
    print("0. 종료")
    valid = {0, 1}
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
    decision = judge(score_a, score_b, "A", "B")
    if decision == UNDECIDED:
        print(f"판정: 판정 불가 (|A-B| < {EPSILON_TEXT})")
    else:
        print(f"판정: {decision}")

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
    except QuitProgram:
        print()
        print("⚠️ 입력이 중단되어 프로그램을 종료합니다.")
        return
    print("👋 Mini NPU Simulator 를 종료합니다.")

if __name__ == "__main__":
    main()
