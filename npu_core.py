"""Mini NPU 핵심: n×n 행렬(Matrix), MAC 연산, 점수 비교(epsilon), 라벨 정규화."""

EPSILON = 1e-9          # |점수A - 점수B| < EPSILON 이면 동점으로 본다
EPSILON_TEXT = "1e-9"    # 화면 표시용 (1e-09 대신)
UNDECIDED = "UNDECIDED"  # 동점일 때의 판정 결과
CROSS = "Cross"          # 표준 라벨
X = "X"                  # 표준 라벨

# 데이터 파일/입력에서 올 수 있는 표기 → 표준 라벨
LABEL_ALIASES = {"+": CROSS, "cross": CROSS, "x": X}

class Matrix:
    """n×n 크기의 2차원 숫자 배열. 위치(행, 열)로 값을 저장하고 읽는다."""

    def __init__(self, size):
        if not isinstance(size, int) or size < 1:
            raise ValueError("크기는 1 이상의 정수여야 합니다")
        self.size = size
        self.rows = [[0.0] * size for _ in range(size)]

    @classmethod
    def from_rows(cls, rows):
        """2차원 리스트로 Matrix 를 만든다. 정사각형이 아니거나 숫자가 아니면 ValueError."""
        if not isinstance(rows, list) or not rows:
            raise ValueError("2차원 배열(리스트의 리스트)이 아닙니다")
        size = len(rows)
        matrix = cls(size)
        for r, row in enumerate(rows):
            if not isinstance(row, list) or len(row) != size:
                length = len(row) if isinstance(row, list) else "?"
                raise ValueError(f"{r + 1}행의 열 수가 {length}개로 {size}×{size} 가 아닙니다")
            for c, value in enumerate(row):
                # bool 은 int 의 하위 타입이라 따로 막는다
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"({r + 1}, {c + 1}) 위치 값 {value!r} 이(가) 숫자가 아닙니다")
                matrix.set(r, c, value)
        return matrix

    def get(self, row, col):
        """(row, col) 위치의 값을 읽는다. 0부터 시작."""
        return self.rows[row][col]

    def set(self, row, col, value):
        """(row, col) 위치에 값을 저장한다. 0부터 시작."""
        self.rows[row][col] = float(value)

    def flatten(self):
        """2차원 → 길이 N² 의 1차원 리스트 (행 우선 순서). 보너스: 메모리 접근 단순화."""
        return [value for row in self.rows for value in row]

    def to_text(self):
        """화면 출력용 문자열."""
        return "\n".join(" ".join(f"{value:g}" for value in row) for row in self.rows)

def mac(pattern, filt):
    """MAC(Multiply-Accumulate): 같은 위치끼리 곱한 값을 모두 더한다. 크기가 다르면 ValueError.

    N×N 이면 곱셈 N² 번 + 덧셈 N² 번 → 시간 복잡도 O(N²).
    """
    if pattern.size != filt.size:
        raise ValueError(f"크기 불일치: 패턴 {pattern.size}×{pattern.size}, 필터 {filt.size}×{filt.size}")
    total = 0.0
    for i in range(pattern.size):
        pattern_row = pattern.rows[i]
        filter_row = filt.rows[i]
        for j in range(pattern.size):
            total += pattern_row[j] * filter_row[j]
    return total

def mac_flat(pattern_flat, filter_flat):
    """보너스: 1차원(길이 N²) 배열끼리의 MAC. 이중 반복 대신 단일 반복으로 같은 결과를 낸다."""
    if len(pattern_flat) != len(filter_flat):
        raise ValueError("크기 불일치")
    total = 0.0
    for k in range(len(pattern_flat)):
        total += pattern_flat[k] * filter_flat[k]
    return total

def judge(score_a, score_b, label_a="A", label_b="B", epsilon=EPSILON):
    """두 점수를 허용오차 기반으로 비교해 더 큰 쪽 라벨을, 동점이면 UNDECIDED 를 돌려준다.

    부동소수점은 0.1 같은 값을 정확히 표현하지 못해 수학적으로 같은 점수도
    0.9 와 0.8999999999999999 처럼 미세하게 달라질 수 있다. 그래서 == 나 > 로 바로 비교하지 않는다.
    """
    if abs(score_a - score_b) < epsilon:
        return UNDECIDED
    return label_a if score_a > score_b else label_b

def normalize_label(value):
    """'+', 'cross', 'x', 'Cross', 'X' 등 여러 표기를 표준 라벨(Cross / X)로 바꾼다. 모르는 값은 ValueError."""
    if not isinstance(value, str):
        raise ValueError(f"라벨이 문자열이 아닙니다: {value!r}")
    key = value.strip().lower()
    if key not in LABEL_ALIASES:
        raise ValueError(f"알 수 없는 라벨: {value!r} (허용: +, cross, x)")
    return LABEL_ALIASES[key]
