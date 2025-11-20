import time
import numpy as np

def measure_hr(duration=10, fs=30):
    """
    예전에 여민이가 쓰던 '잘 되던' 버전 복구본
    - 3,2,1 카운트다운
    - duration 초 동안 MAX30102에서 IR값 읽고
    - 피크 개수로 bpm 추정
    """

    try:
        from max30102_driver import MAX30102
    except Exception as e:
        print("[HW_605] max30102 import 실패:", e)
        return 0.0

    try:
        sensor = MAX30102()
    except Exception as e:
        print("[HW_605] 센서 초기화 실패:", e)
        return 0.0

    print("심박 측정 준비하세요!")
    for i in range(3, 0, -1):
        print(i)
        time.sleep(1)
    print("측정 시작!\n")

    ir_values = []

    start = time.time()

    while time.time() - start < duration:
        try:
            ir, red = sensor.read_fifo()
        except Exception as e:
            print("[HW_605] 센서 읽기 오류:", e)
            continue

        # 손가락 안 올라온 상태면 스킵
        if ir < 30000:
            # print("손가락 없음, IR =", ir)
            time.sleep(0.05)
            continue

        ir_values.append(ir)

        time.sleep(1.0 / fs)

    # ---------- 여기서부터 분석 ----------
    if len(ir_values) < fs * duration * 0.3:
        # 샘플이 너무 적으면 실패 처리
        print("[HW_605] 유효한 데이터가 너무 적습니다.")
        return 0.0

    data = np.array(ir_values, dtype=float)

    # 간단 smoothing (노이즈 줄이기)
    smooth = np.convolve(data, np.ones(3)/3, mode='same')

    # peak 찾기 (local maxima + 최소 간격)
    MIN_PEAK_DIST_SEC = 0.4  # 최소 박동 간격(초) → 최대 150bpm 정도
    MIN_PEAK_DIST = int(MIN_PEAK_DIST_SEC * fs)

    peaks = []
    last = -1000

    for i in range(1, len(smooth)-1):
        if smooth[i] > smooth[i-1] and smooth[i] > smooth[i+1]:
            if i - last > MIN_PEAK_DIST:
                peaks.append(i)
                last = i

    num_beats = len(peaks)

    # 실제 측정 시간 (정확히 duration이 아닐 수도 있으니 계산해서 사용)
    measured_time = len(ir_values) / fs

    if measured_time == 0:
        return 0.0

    bpm = num_beats * 60.0 / measured_time

    print(f"[HW_605] beats={num_beats}, time={measured_time:.2f}s, bpm={bpm:.1f}")

    return float(bpm)


if __name__ == "__main__":
    bpm = measure_hr(10)
    print("테스트 BPM:", bpm)
