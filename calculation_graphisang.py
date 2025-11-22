import time
import math
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from HW_605 import measure_hr
from gsr_sensor import measure_gsr


# =============================
# 설정값
# =============================
MEASURE_INTERVAL_MIN = 1
CAFFEINE_DB = {"아메리카노": 100}

# ===== 그래프 저장용 리스트 =====
time_log = []
HR_log = []
GSR_log = []
S_log = []
alpha_log = []
beta_log = []
C_log = []
R_log = []

# 예측 그래프
future_time_log = []
future_C_log = []
future_R_log = []

# =============================
# HR 노이즈 제거용 버퍼
# =============================
HR_buffer = []
HR_BUFFER_SIZE = 3
HR_JUMP_LIMIT = 25

# =============================
# 개인화 가중치
# =============================
W_ALPHA = 0.3
W_BETA  = 0.5

S_REF = 0.5
BETA_REF = 0.05


# =============================
# HR 안정화 함수
# =============================
def smooth_hr(raw_hr, prev_hr):
    if prev_hr is None:
        HR_buffer.clear()
        HR_buffer.append(raw_hr)
        return raw_hr

    if raw_hr < 40 or raw_hr > 180:
        return prev_hr

    if abs(raw_hr - prev_hr) > HR_JUMP_LIMIT:
        return prev_hr

    HR_buffer.append(raw_hr)
    if len(HR_buffer) > HR_BUFFER_SIZE:
        HR_buffer.pop(0)

    return sum(HR_buffer) / len(HR_buffer)


def save_and_plot(t_min, HR, GSR, S, alpha, beta, C, R):
    time_log.append(t_min)
    HR_log.append(HR)
    GSR_log.append(GSR)
    S_log.append(S)
    alpha_log.append(alpha)
    beta_log.append(beta)
    C_log.append(C)
    R_log.append(R)


# =============================
# 1) S-value
# =============================
def calculate_S(H_now, H_prev, G_now, G_prev, dt_seconds):
    if dt_seconds <= 0 or H_prev is None or G_prev is None:
        return 0.0
    
    dH_dt = (H_now - H_prev) / dt_seconds
    dG_dt = (G_now - G_prev) / dt_seconds

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
# 4) C(t)
# =============================
def update_caffeine(C_prev, new_doses, dt, S_prev, S_now):
    if dt <= 0:
        return C_prev + sum(new_doses)

    HALF_LIFE_HOURS = 5.0
    k = math.log(2) / HALF_LIFE_HOURS
    dt_hours = dt / 3600.0

    C_decay = C_prev * math.exp(-k * dt_hours)
    C_add = sum(new_doses)

    return C_decay + C_add


# =============================
# 5) R(t)
# =============================
def calculate_R(alpha, beta, C, t_min):
    t_hr = t_min / 60.0
    return alpha * C - beta * t_hr


# =============================
# 미래 예측 그래프 생성
# =============================
def simulate_future(C_now, R_now, alpha, beta, t_min):
    future_time_log.clear()
    future_C_log.clear()
    future_R_log.clear()

    dt = 300
    k = math.log(2) / (5.0 * 3600)

    C_threshold = 80
    R_threshold = 1.0

    C = C_now
    R = R_now
    t = t_min

    for step in range(1, 5000):
        # 미래 카페인 감소
        C = C * math.exp(-k * dt)

        # 미래 R 업데이트
        R = R + (alpha * C - beta) * (dt / 3600)

        t += dt / 60.0

        future_time_log.append(t)
        future_C_log.append(C)
        future_R_log.append(R)

        if C < C_threshold and abs(R) < R_threshold:
            return t

    return None


# =============================
# 그래프 출력
# =============================
def plot_final_graph(sleep_time):
    plt.figure(figsize=(12, 7))

    # 현재 측정된 값
    plt.plot(time_log, C_log, label="C(t)", linewidth=2)
    plt.plot(time_log, R_log, label="R(t)", linewidth=2)
    plt.plot(time_log, S_log, label="S(t)", linewidth=2)

    # 미래 예측
    if len(future_time_log) > 0:
        plt.plot(future_time_log, future_C_log, 'r--', label="C future")
        plt.plot(future_time_log, future_R_log, 'g--', label="R future")

    if sleep_time is not None:
        plt.axvline(sleep_time, color='purple', linestyle=':', linewidth=2)
        plt.text(sleep_time, max(C_log + future_C_log), "Sleep Possible", color='purple')

    plt.xlabel("Time (min)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =============================
# 메인
# =============================
def main():
    print("\n=== CaffSense (10회 이후 예측 + 미래 그래프) ===\n")

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

    drink_log = []

    while True:
        print(f"\n--- {t_min}분 경과 측정 ---\n")

        drink = input("섭취 음료? (아메리카노 / 엔터): ").strip()
        new_doses = []
        if drink in CAFFEINE_DB:
            dose = CAFFEINE_DB[drink]
            new_doses.append(dose)
            drink_log.append((time.time(), dose))
            print(f"[INFO] +{dose} mg 추가")

        raw_HR = measure_hr(10)
        G_now = measure_gsr()
        t_now = time.time()

        HR_now = smooth_hr(raw_HR, prev_HR)
        print(f"Raw HR = {raw_HR}, Smoothed HR = {HR_now}")

        if prev_time is None:
            dt = 60
            H_peak = HR_now
            G_peak = G_now
            t_peak = t_now
            S_now = 0
            beta = 0
        else:
            dt = t_now - prev_time
            S_now = calculate_S(HR_now, prev_HR, G_now, prev_GSR, dt)
            alpha = update_alpha(alpha, S_now)

            if HR_now > H_peak:
                H_peak = HR_now
                G_peak = G_now
                t_peak = t_now

            beta = calculate_beta(H_peak, HR_now, G_peak, G_now, t_peak, t_now)

        C = update_caffeine(C, new_doses, dt, prev_S, S_now)
        R = calculate_R(alpha, beta, C, t_min)

        save_and_plot(t_min, HR_now, G_now, S_now, alpha, beta, C, R)

        print(f"S = {S_now}, α = {alpha}, β = {beta}, C = {C}, R = {R}")

        # ---------------------------
        # 10회 이전 예측 없음
        # ---------------------------
        if len(S_log) < 10:
            print("[수면 예측] → 최소 10회 측정 후 예측 가능합니다.\n")
        else:
            sleep_time = simulate_future(C, R, alpha, beta, t_min)
            if sleep_time is None:
                print("[수면 예측] 매우 장시간 동안 불가\n")
            else:
                print(f"[수면 예측] {sleep_time:.1f}분 시점에 수면 가능\n")

        prev_HR = HR_now
        prev_GSR = G_now
        prev_time = t_now
        prev_S = S_now

        stop_check = input("종료하려면 'q' 입력: ").strip()
        if stop_check == "q":
            break

        print("1분 대기...\n")
        time.sleep(60)
        t_min += MEASURE_INTERVAL_MIN

    # 종료 후 그래프
    if len(S_log) >= 5:
        sleep_time = simulate_future(C_log[-1], R_log[-1], alpha_log[-1], beta_log[-1], time_log[-1])
        plot_final_graph(sleep_time)
    else:
        plot_final_graph(None)



if __name__ == "__main__":
    main()
