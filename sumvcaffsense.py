import time
from typing import Optional, Dict, Generator, Tuple
import random

# 1) ---- 센서 어댑터 인터페이스 ----
# 하드웨어 도착 후 아래 3개 함수만 실제 드라이버로 교체하면 됩니다.

def read_hr_spo2() -> Optional[Tuple[float, float]]:
    """
    MAX30102용 자리. (hr_bpm, spo2_percent) 반환.
    접촉 상실/에러면 None 반환.
    지금은 더미값으로 시뮬레이션.
    """
    # --- 더미 시뮬레이션 ---
    if random.random() < 0.05:  # 5% 확률로 접촉 상실
        return None
    hr = 70 + random.uniform(-2, 2)          # 68~72 bpm
    spo2 = 98.0                              # 우리 프로젝트에선 상수 사용
    return (hr, spo2)

def read_gsr() -> Optional[int]:
    """
    GSR(MCP3008) 자리. 0~1023 범위 정수 권장.
    접촉 상실/에러면 None 반환.
    지금은 더미값으로 시뮬레이션.
    """
    if random.random() < 0.05:
        return None
    return int(600 + random.uniform(-10, 10))  # 590~610

# 2) ---- 전처리 훅 (원하면 여기서 이동평균/버터워스 추가) ----
def preprocess_hr(hr: float) -> float:
    return hr  # 나중에 필터/클램프 추가

def preprocess_gsr(gsr: int) -> int:
    return gsr  # 나중에 이동평균/클램프 추가

# 3) ---- 합친 샘플을 만들어주는 제너레이터 ----
def stream_samples(dt: float = 1.0) -> Generator[Dict, None, None]:
    """
    매 dt초마다 통합 샘플 딕셔너리를 yield.
    출력 형식:
      {
        "ts": epoch_sec,
        "hr": float|None,
        "spo2": float|None,
        "gsr": int|None,
        "valid": 0|1   # 모든 값이 유효하면 1, 아니면 0
      }
    """
    while True:
        ts = time.time()
        hr_spo2 = read_hr_spo2()
        gsr = read_gsr()

        if hr_spo2 is None:
            hr, spo2 = None, None
        else:
            hr, spo2 = hr_spo2

        # 전처리 훅
        if hr is not None:
            hr = preprocess_hr(hr)
        if gsr is not None:
            gsr = preprocess_gsr(gsr)

        valid = int((hr is not None) and (gsr is not None))
        yield {"ts": ts, "hr": hr, "spo2": spo2, "gsr": gsr, "valid": valid}

        time.sleep(dt)

# 4) ---- 사용 예시: while 루프로 소비 ----
if __name__ == "__main__":
    for i, sample in enumerate(stream_samples(dt=1.0)):
        print(sample)
        if i >= 10:
            break
