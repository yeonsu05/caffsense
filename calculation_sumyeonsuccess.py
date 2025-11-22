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

# =============================
# HR 노이즈 제거용 버퍼
# =============================
HR_buffer = []
HR_BUFFER_SIZE = 3
HR_JUMP_LIMIT = 25

# =============================
# 개인화 가중치
# =============================
W_ALPHA = 0.3   # 형성률 개인화 반영 비율 (0~1)
W_BETA  = 0.5   # 회복률 개인화 반영 비율 (0~1)

# 기준값 (대략적인 레퍼런스 스케일)
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


def plot_final_graph():
    plt.figure(figsize=(12, 7))
    plt.plot(time_log, S_log, label="S-value")
    plt.plot(time_log, C_log, label="C(t)")
    plt.plot(time_log, R_log, label="R(t)")
    plt.plot(time_log, HR_log, label="HR")
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
# 6) 개인 반감기
# =============================
def calculate_personal_half_life(S_history):
    if len(S_history) == 0:
        return 5.0

    S_avg = sum(S_history) / len(S_history)
    personal_half_life = 5.0 / (1 + 0.1 * abs(S_avg))
    personal_half_life = max(2.0, min(7.0, personal_half_life))
    return personal_half_life


# =============================
# 7) 수면 예측 (개인화 + 가중치)
# =============================
def predict_sleep_time(drink_log, HALF_LIFE_HOURS, R_now, alpha, beta, gamma):

    if not drink_log:
        return "이미 취침 가능"

    total_dose = sum(D for (t, D) in drink_log)
    k = math.log(2) / (HALF_LIFE_HOURS * 3600)

    C_sleep_threshold = max(total_dose * 0.35, 10.0)
    R_sleep_threshold = 3.0

    dt = 300
    future_hours = 24
    R = R_now
    t_now = time.time()

    lambda_decay = 0.15

    global S_log, beta_log

    # ----- α 개인화 -----
    if len(S_log) > 0:
        S_peak = max(abs(s) for s in S_log)
        if S_REF > 0:
            k_alpha_raw = S_peak / S_REF
        else:
            k_alpha_raw = 1.0
        k_alpha_raw = max(0.5, min(1.5, k_alpha_raw))
    else:
        k_alpha_raw = 1.0

    alpha_personal = alpha * k_alpha_raw
    alpha_eff = alpha * (1 - W_ALPHA) + alpha_personal * W_ALPHA

    # ----- β 개인화 -----
    if len(beta_log) > 0:
        beta_avg = sum(abs(b) for b in beta_log) / len(beta_log)
        if BETA_REF > 0:
            k_beta_raw = beta_avg / BETA_REF
        else:
            k_beta_raw = 1.0
        k_beta_raw = max(0.5, min(1.5, k_beta_raw))
    else:
        k_beta_raw = 1.0

    beta_personal = beta * k_beta_raw
    beta_eff = beta * (1 - W_BETA) + beta_personal * W_BETA

    # ----- 미래 R(t) 예측 -----
    for step in range(int((future_hours * 3600) / dt)):
        t_future = t_now + step * dt

        C = 0.0
        for (ti, Di) in drink_log:
            if t_future >= ti:
                C += Di * math.exp(-k * (t_future - ti))

        dt_hr = dt / 3600.0

        R = R * math.exp(-lambda_decay * dt_hr) + alpha_eff * C * dt_hr - beta_eff * dt_hr
        if R < 0:
            R = 0

        if C <= C_sleep_threshold and R <= R_sleep_threshold:
            remaining_sec = step * dt
            hours = remaining_sec // 3600
            minutes = (remaining_sec % 3600) // 60
            return f"{hours}시간 {minutes}분 후 취침 가능"

    return "취침 불가능 (각성도·카페인 유지됨)"


# =============================
# 8) 메인
# =============================
def main():
    print("\n=== CaffSense (개선된 HR 안정화 + 개인화 가중 버전) ===\n")

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

        if HR_now == 0.0:
            if prev_HR is not None:
                HR_now = prev_HR
            else:
                C = C + sum(new_doses)
                prev_time = t_now
                t_min += MEASURE_INTERVAL_MIN
                time.sleep(60)
                continue

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

        print(f"S-value = {S_now}")
        print(f"α = {alpha}")
        print(f"β = {beta}")

        C = update_caffeine(C, new_doses, dt, prev_S, S_now)
        print(f"C(t) = {C}")

        R = calculate_R(alpha, beta, C, t_min)
        print(f"R(t) = {R}")

        save_and_plot(t_min, HR_now, G_now, S_now, alpha, beta, C, R)

        personal_half_life = calculate_personal_half_life(S_log)

        sleep_time = predict_sleep_time(
            drink_log,
            personal_half_life,
            R,
            alpha,
            beta,
            0
        )
        print(f"[수면 예측] → {sleep_time}")

        prev_HR = HR_now
        prev_GSR = G_now
        prev_time = t_now
        prev_S = S_now

        if t_min > 0:
            stop_check = input("종료하려면 'q' 입력: ").strip()
            if stop_check.lower() == "q":
                break

        print("\n1분 대기...\n")
        time.sleep(60)
        t_min += MEASURE_INTERVAL_MIN



if __name__ == "__main__":
    main()

    show_graph = input("\n그래프 출력할까요? (y/n): ").strip().lower()
    if show_graph == "y":
        plot_final_graph()
    else:
        print("그래프 출력 생략됨.")
