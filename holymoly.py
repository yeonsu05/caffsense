import time 
import math
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from HW_605 import measure_hr
from gsr_sensor import measure_gsr
import random


# =============================
# 설정값
# =============================
MEASURE_INTERVAL_MIN = 1
CAFFEINE_DB = {"아메리카노": 100}
CAFFEINE_JUMP_RELAX_SEC = 600
last_caffeine_time = None

# =============================
# 그래프 저장용 리스트
# =============================
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
HR_BUFFER_SIZE = 3                                         # 몇 개의 HR 값들을 리스트에 저장할지
HR_JUMP_LIMIT = 25                                         # HR값의 어느 정도 차이를 무시할지

# =============================
# 개인화 가중치 (예측용에서는 거의 안씀)
# =============================
W_ALPHA = 0.3
W_BETA  = 0.5

# 기준값
S_REF = 0.5
BETA_REF = 0.05


# =============================
# HR 안정화 함수
# =============================
def smooth_hr(raw_hr, prev_raw_hr):
    """
    raw_hr(방금 측정한 HR)만 받아서 버퍼 기반으로 스무딩.
    >>> 이 함수 결과(HR_s)는 오직 S-value 계산에만 사용. <<<
    """
    global last_caffeine_time

    if prev_raw_hr is None:                              # prev_raw_hr 값이 없을 때
        HR_buffer.clear()
        HR_buffer.append(raw_hr)
        return raw_hr

    if raw_hr < 50 or raw_hr > 130:                      # raw_hr 값 최소(50) 최대(130) 설정
        return prev_raw_hr
    
    # 카페인 섭취 후 일정 시간(10분) 동안은 HR 점프 필터 비활성화
    use_jump_filter = True
    if last_caffeine_time is not None:
        elapsed_since_caffeine = time.time() - last_caffeine_time
        if elapsed_since_caffeine <= CAFFEINE_JUMP_RELAX_SEC:
            use_jump_filter = False

    if use_jump_filter and abs(raw_hr - prev_raw_hr) > HR_JUMP_LIMIT:        # 튀는 값 무시 과정
        return prev_raw_hr

    HR_buffer.append(raw_hr)                             # 정상 HR값 buffer에 저장
    if len(HR_buffer) > HR_BUFFER_SIZE:                  # HR_BUFFER_SIZE 유지
        HR_buffer.pop(0)                                 # 제일 마지막으로 저장한 raw_hr 없애기

    return sum(HR_buffer) / len(HR_buffer)               # 버퍼에 저장된 정상 HR값 평균내기


# =============================
# 그래프 저장용 리스트 함수
# =============================
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
# 그래프 만드는 함수
# =============================
def plot_final_graph():
    plt.figure(figsize=(12, 7))
    plt.plot(time_log, S_log, label="S-value")
    plt.plot(time_log, C_log, label="C(t)")
    plt.plot(time_log, R_log, label="R(t)")
    plt.plot(time_log, HR_log, label="HR (raw)")
    plt.xlabel("Time (min)")
    plt.ylabel("Value")
    plt.title("Real-time CaffSense Graph")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


########## 식 계산 시작 ##########


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
# 7) 수면 예측 (A버전: 안정형)
# =============================
def predict_sleep_time(drink_log, HALF_LIFE_HOURS, R_now, alpha, beta, gamma):

    if not drink_log:
        return "이미 취침 가능"

    # 카페인 반감기 상수
    k = math.log(2) / (HALF_LIFE_HOURS * 3600)

    # 수면 기준
    C_SLEEP_THRESHOLD = 70.0   # mg
    R_SLEEP_THRESHOLD = 5.0

    dt = 300                   # 5분 단위
    future_hours = 24
    t_now = time.time()

    # R는 여기서는 "단순 지수 감소" 모델로 예측
    # R(t) = R_now * exp(-k_R * t_hr)
    k_R = 0.7                  # R의 시간당 감소율(대충 1시간 반감 느낌)

    # 24시간 동안 5분 간격으로 훑으면서, 처음으로
    #  C <= 70 이고 R_simple <= 5 가 되는 시점을 찾음
    for step in range(int((future_hours * 3600) / dt)):
        t_future = t_now + step * dt
        t_hr = step * dt / 3600.0

        # 1) 카페인 농도 C(t) 계산 (기존 방식 그대로)
        C = 0.0
        for (ti, Di) in drink_log:
            if t_future >= ti:
                C += Di * math.exp(-k * (t_future - ti))

        # 2) R(t) 단순 예측 (현재 R_now 기준 지수 감소)
        R_simple = R_now * math.exp(-k_R * t_hr)

        # 3) 수면 가능 조건 체크
        if C <= C_SLEEP_THRESHOLD and R_simple <= R_SLEEP_THRESHOLD:
            remaining_sec = step * dt
            hours = remaining_sec // 3600
            minutes = (remaining_sec % 3600) // 60
            return f"{hours}시간 {minutes}분 후 취침 가능"

    return "취침 불가능 (각성도·카페인 유지됨)"


# =============================
# 8) 메인
# =============================
def main():
    global last_caffeine_time
    print("\n=== CaffSense (A버전: 안정형 수면 예측) ===\n")

    C = 0.0
    t_min = 0

    prev_raw_HR = None
    prev_HR_s = None
    prev_GSR = None
    prev_GSR_norm = None
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
            now_drink_time = time.time()
            drink_log.append((now_drink_time, dose))
            last_caffeine_time = now_drink_time
            print(f"[INFO] +{dose} mg 추가")

        # HR/GSR 측정
        raw_HR = measure_hr(10)
        _ = measure_gsr()  # 실제 센서값은 안쓰고 아래에서 G_raw 시뮬레이션
        t_now = time.time()

        # HR 스무딩
        HR_s = smooth_hr(raw_HR, prev_raw_HR)
        print(f"Raw HR = {raw_HR}, Smoothed HR (for S) = {HR_s}")

        # HR 측정 실패 처리
        if raw_HR == 0.0:
            if prev_raw_HR is not None:
                raw_HR = prev_raw_HR
                HR_s = HR_s if HR_s != 0.0 else prev_HR_s if prev_HR_s is not None else prev_raw_HR
            else:
                C = C + sum(new_doses)
                prev_time = t_now
                t_min += MEASURE_INTERVAL_MIN
                print("\nHR 측정 실패, 1분 대기...\n")
                time.sleep(60)
                continue

        # GSR 시뮬레이션 + 정규화
        if prev_GSR is None:
            G_raw = random.uniform(5.0, 10.0)
        else:
            drift = random.uniform(-0.5, 0.8)
            G_raw = prev_GSR + drift

        G_raw = max(1.0, min(20.0, G_raw))
        G_now = (G_raw - 1.0) / 19.0

        print(f"GSR_raw = {G_raw:.2f} µS  |  GSR_norm(S용) = {G_now:.3f}")        

        # 시간, S, α, β 계산
        if prev_time is None:
            dt = 60
            H_peak = raw_HR
            G_peak = G_now
            t_peak = t_now
            S_now = 0.0
            beta = 0.0
        else:
            dt = t_now - prev_time
            S_now = calculate_S(HR_s, prev_HR_s, G_now, prev_GSR_norm, dt)
            alpha = update_alpha(alpha, S_now)

            if H_peak is None or raw_HR > H_peak:
                H_peak = raw_HR
                G_peak = G_now
                t_peak = t_now

            beta = calculate_beta(H_peak, raw_HR, G_peak, G_now, t_peak, t_now)

        print(f"S-value = {S_now}")
        print(f"α = {alpha}")
        print(f"β = {beta}")

        # C(t), R(t)
        C = update_caffeine(C, new_doses, dt, prev_S, S_now)
        print(f"C(t) = {C}")

        R = calculate_R(alpha, beta, C, t_min)
        print(f"R(t) = {R}")

        save_and_plot(t_min, raw_HR, G_now, S_now, alpha, beta, C, R)

        # 수면 예측용 half-life
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

        # 이전 값 업데이트
        prev_raw_HR = raw_HR
        prev_HR_s = HR_s
        prev_GSR = G_raw
        prev_GSR_norm = G_now
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
