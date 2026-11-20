"""성능 분석: 크기별 MAC 연산 시간을 반복 측정해 평균(ms)을 구한다. 입출력 시간은 제외."""

import time

from npu_core import mac
from npu_core import mac_flat

REPEAT = 10   # 크기별 반복 측정 횟수 (최소 기준 10회)
INNER = 100   # 1회 측정에서 연속 호출하는 횟수 (마이크로초 단위 측정 잡음을 줄이기 위함)

def average_ms(function, *args, repeat=REPEAT, inner=INNER):
    """function(*args) 호출 구간만 repeat 번 측정해 '호출 1번당' 평균 시간(ms)을 돌려준다.

    3×3 MAC 은 1마이크로초 안팎이라 한 번만 재면 타이머 해상도·OS 잡음이 결과를 좌우한다.
    그래서 1회 측정마다 inner 번 연속 호출한 시간을 재고, repeat(10)회 측정의 평균을 호출 수로 나눈다.
    """
    function(*args)  # 첫 호출은 준비 비용이 섞일 수 있어 측정에서 제외(워밍업)
    total_ns = 0
    for _ in range(repeat):
        start = time.perf_counter_ns()
        for _ in range(inner):
            function(*args)
        total_ns += time.perf_counter_ns() - start
    return total_ns / (repeat * inner) / 1_000_000

def measure_sizes(cases):
    """cases: [(N, 패턴 Matrix, 필터 Matrix), ...] → [(N, 평균 ms, N²), ...]"""
    rows = []
    for size, pattern, filt in cases:
        rows.append((size, average_ms(mac, pattern, filt), size * size))
    return rows

def format_table(rows):
    """크기 / 평균 시간(ms) / 연산 횟수(N²) 표 문자열."""
    lines = [f"{'크기':<10}{'평균 시간(ms)':>16}{'연산 횟수(N²)':>16}", "-" * 44]
    for size, ms, ops in rows:
        lines.append(f"{f'{size}×{size}':<12}{ms:>14.4f}{ops:>16}")
    return "\n".join(lines)

def compare_flat(cases):
    """보너스: 같은 입력·같은 반복 횟수로 2차원 MAC 과 1차원 MAC 을 비교한다."""
    rows = []
    for size, pattern, filt in cases:
        two_d = average_ms(mac, pattern, filt)
        pattern_flat, filter_flat = pattern.flatten(), filt.flatten()  # 변환은 측정 밖에서 1회
        one_d = average_ms(mac_flat, pattern_flat, filter_flat)
        same = abs(mac(pattern, filt) - mac_flat(pattern_flat, filter_flat)) < 1e-9
        rows.append((size, two_d, one_d, same))
    return rows

def format_compare_table(rows):
    lines = [f"{'크기':<10}{'2차원(ms)':>12}{'1차원(ms)':>12}{'변화':>10}{'결과 동일':>10}", "-" * 56]
    for size, two_d, one_d, same in rows:
        change = (one_d - two_d) / two_d * 100 if two_d else 0.0
        lines.append(
            f"{f'{size}×{size}':<12}{two_d:>10.4f}{one_d:>12.4f}{change:>+9.1f}%{'예' if same else '아니오':>9}"
        )
    return "\n".join(lines)
