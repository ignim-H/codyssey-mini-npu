"""보너스: N×N 십자가(Cross) / X 패턴 자동 생성기."""

from npu_core import CROSS, X, Matrix

def make_cross(size, value=1.0):
    """가운데 행과 가운데 열이 value 인 십자가 패턴 (size 는 3 이상의 홀수)."""
    _check(size)
    matrix = Matrix(size)
    middle = size // 2
    for i in range(size):
        matrix.set(middle, i, value)
        matrix.set(i, middle, value)
    return matrix

def make_x(size, value=1.0):
    """두 대각선이 value 인 X 패턴 (size 는 3 이상의 홀수)."""
    _check(size)
    matrix = Matrix(size)
    for i in range(size):
        matrix.set(i, i, value)
        matrix.set(i, size - 1 - i, value)
    return matrix

def make_pattern(label, size):
    """표준 라벨(Cross / X)에 맞는 패턴을 만든다."""
    return make_cross(size) if label == CROSS else make_x(size)

def _check(size):
    # 짝수면 정가운데 칸이 없어 십자가/X 가 대칭이 되지 않는다
    if not isinstance(size, int) or size < 3 or size % 2 == 0:
        raise ValueError("크기는 3 이상의 홀수여야 합니다 (예: 3, 5, 13, 25)")
