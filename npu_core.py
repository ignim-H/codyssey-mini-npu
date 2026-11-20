"""Mini NPU 핵심: n×n 행렬(Matrix), MAC 연산, 점수 비교(epsilon), 라벨 정규화."""

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

    def to_text(self):
        """화면 출력용 문자열."""
        return "\n".join(" ".join(f"{value:g}" for value in row) for row in self.rows)
