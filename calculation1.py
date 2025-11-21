import time
import math
import matplotlib.pyplot as plt

from HW_605 import measure_hr
from gsr_sensor import measure_gsr


# =============================
# 설정값
# =============================
MEASURE_INTERVAL_MIN = 1           # ★ 1분 대기 (테스트용)

CAFFEINE_DB = { "아메리카노": 100 }


# ===== 그래프 저장용 리스트 =====
time_log = []
HR_log = []
GSR_log = []
S_log = []
alpha_log = []
beta_log = []
C_log = []
R_log = []


def save_and_plot(t_min, HR, GSR, S, alpha, beta, C, R):
    # 값 누적 저장-측정 종료 후에 선택적으로 출력
    time_log.append(t_min)
    HR_log.append(HR)
    GSR_log.append(GSR)
    S_log.append(S)
    alpha_log.append(alpha)
    beta_log.append(beta)
    C_log.append(C)
    R_log.append(R)

def plot_final_graph():
    # 측정 종료 후 그래프 출력
    plt.figure(figsize=(12, 7))
    plt.plot(time_log, S_log, label="S-value")
    plt.plot(time_log, C_log, label="C(t) - Caffeine")
    plt.plot(time_log, R_log, label="R(t) - Arousal")

    plt.xlabel("Time (min)")
    plt.ylabel("Value")
    plt.title("Real-time CaffSense Graph")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =============================
# 1) S-value
# =============================
def calculate_S(H_now, H_prev, G_now, G_prev, dt_seconds):

    if dt_seconds <= 0 or H_prev is None or G_prev is None:
        return 0.0
    
    dH_dt = (H_now - H_prev) / dt_seconds
    dG_dt = (G_now - G_prev) / dt_seconds

    # 감도 스케일은 기존대로 10배 유지
    return (0.6 * dH_dt + 0.4 * dG_dt) * 10.0


# =============================
# 2) α 형성률
# =============================
def update_alpha(alpha_prev, S):
    gamma_rise = 0.2
    gamma_decay = 0.05

    if S > 0:
        return (1 - gamma_rise) * alpha_prev + gamma_rise * S
    else:
        return (1 - gamma_decay) * alpha_prev


# =============================
# 3) β 회복률
# =============================
def calculate_beta(H_peak, H_now, G_peak, G_now, t_peak, t_now):

    if H_peak is None or G_peak is None:
        return 0.0

    dt = t_now - t_peak
    if dt <= 0:
        return 0.0

    try:
        beta_hr = (math.log(H_peak) - math.log(H_now)) / dt
        beta_gs = (math.log(G_peak) - math.log(G_now)) / dt
    except:
        return 0.0

    return 10.0 * (0.6 * beta_hr + 0.4 * beta_gs)


# =============================
# 4) C(t) — 카페인 농도 (반감기 기반, S와 독립)
# =============================
def update_caffeine(C_prev, new_doses, dt, S_prev, S_now):
    """
    C_prev  : 직전 카페인 잔류량 (상대량)
    new_doses : 이번 구간에 새로 마신 카페인 [mg, ...]
    dt      : 지난 시간 (초 단위)  ← 너 코드에서 dt = t_now - prev_time
    S_prev, S_now : 더 이상 사용하지 않지만, 함수 시그니처 유지를 위해 받기만 함
    """

    if dt <= 0:
        # 시간 안 지났으면 그냥 새로 마신 것만 더함
        return C_prev + sum(new_doses)

    # 1) 카페인 반감기 기반 소실 상수 (고정값)
    HALF_LIFE_HOURS = 5.0           # 필요하면 나중에 사용자별 튜닝
    k = math.log(2) / HALF_LIFE_HOURS  # 단위: 1/시간

    # 2) dt(초) -> 시간 단위로 변환
    dt_hours = dt / 3600.0

    # 3) 이전 잔류량을 반감기 기반으로 감소
    C_decay = C_prev * math.exp(-k * dt_hours)

    # 4) 이번 구간에 새로 마신 카페인 양 더하기
    C_add = sum(new_doses)

    return C_decay + C_add


# =============================
# 5) R(t)
# =============================
def calculate_R(alpha, beta, C, t_min):
    t_hr = t_min / 60.0
    return alpha * C - beta * t_hr


# =============================
# 6) 메인
# =============================
def main():
    print("\n=== CaffSense (1분 간격 테스트 버전) ===\n")

    C = 0.0
    t_min = 0

    prev_HR = None
    prev_GSR = None
    prev_time = None

    alpha = 0.0
    prev_S = 0.0

    H_peak = None
    G_peak = None
    t_peak = None

    while True:
        print(f"\n--- {t_min}분 경과 측정 ---\n")

     
        # ==========================
        # 카페인 입력
        # ==========================
        drink = input("섭취 음료? (아메리카노 / 엔터): ").strip()
        new_doses = []
        if drink in CAFFEINE_DB:
            new_doses.append(CAFFEINE_DB[drink])
            print(f"[INFO] +{CAFFEINE_DB[drink]} mg 추가")

        # ==========================
        # 생체신호 측정
        # ==========================
        HR_now = measure_hr(10)
        G_now = measure_gsr()
        t_now = time.time()

        print(f"HR = {HR_now}, GSR = {G_now}")

        # HR 실패 시 처리
        if HR_now == 0.0:
            if prev_HR is not None:
                print("[WARN] HR=0 → 이전 HR 사용")
                HR_now = prev_HR
            else:
                print("[ERROR] HR baseline 없음 → skip")
                C = C + sum(new_doses)
                prev_time = t_now
                t_min += MEASURE_INTERVAL_MIN
                time.sleep(60)
                continue

        # ==========================
        # 첫 측정 처리
        # ==========================
        if prev_time is None:
            dt = 60  # ★ 처음 dt를 60초로 고정 (테스트 안정화)
            H_peak = HR_now
            G_peak = G_now
            t_peak = t_now
            S_now = 0
            beta = 0

        else:
            dt = t_now - prev_time
            S_now = calculate_S(HR_now, prev_HR, G_now, prev_GSR, dt)
            alpha = update_alpha(alpha, S_now)

            # 피크 갱신
            if H_peak is None or HR_now > H_peak:
                H_peak = HR_now
                G_peak = G_now
                t_peak = t_now

            beta = calculate_beta(H_peak, HR_now, G_peak, G_now, t_peak, t_now)

        print(f"S-value = {S_now}")
        print(f"α = {alpha}")
        print(f"β = {beta}")

        # ==========================
        # C(t)
        # ==========================
        C = update_caffeine(C, new_doses, dt, prev_S, S_now)
        print(f"C(t) = {C}")

        # ==========================
        # R(t)
        # ==========================
        R = calculate_R(alpha, beta, C, t_min)
        print(f"R(t) = {R}")


        # R 계산 이후 그래프 로그에 저장
        save_and_plot(t_min, HR_now, G_now, S_now, alpha, beta, C, R)


        # === 상태 업데이트 ===
        prev_HR = HR_now
        prev_GSR = G_now
        prev_time = t_now
        prev_S = S_now

        # ==========================
        # 센서 측정/계산 끝난 뒤 종료 여부 묻기
        # ==========================
        if t_min > 0:
            stop_check = input("측정을 종료하려면 'q' 입력 (계속하려면 엔터): ").strip()
            if stop_check.lower() == "q":
                print("\n측정 종료!")
                break

        # === 1분 대기 ===
        print("\n1분 대기...\n")
        time.sleep(60)

        t_min += MEASURE_INTERVAL_MIN


if __name__ == "__main__":
    main()

    # ==========================================
    # 종료 후 그래프 출력 여부 확인
    # ==========================================
    show_graph = input("\n그래프 출력할까요? (y/n): ").strip().lower()
    if show_graph == 'y':
        plot_final_graph()
    else:
        print("그래프 출력 생략됨.")
