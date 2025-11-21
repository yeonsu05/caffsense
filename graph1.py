import matplotlib.pyplot as plt
import numpy as np

# ================================================================
# 📌 메인코드에서 계산된 전체 로그를 그대로 받아 그래프 출력
# ================================================================
def visualize_from_main(time_log, S_log, C_log, R_log):
    """
    메인코드에서 누적한:
    - time_log : 분 단위 시간 리스트
    - S_log    : S-value 리스트
    - C_log    : C(t) 리스트
    - R_log    : R(t) 리스트
    를 이용해 그래프를 출력한다.
    """

    plt.figure(figsize=(12, 7))

    plt.plot(time_log, S_log, label="S-value")
    plt.plot(time_log, C_log, label="Caffeine C(t)")
    plt.plot(time_log, R_log, label="R(t)")

    plt.title("CaffSense Visualization (Live Data)")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Values")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
