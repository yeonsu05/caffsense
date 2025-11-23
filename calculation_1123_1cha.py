import time
import math
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from HW_605 import measure_hr
from gsr_sensor import measure_gsr

# =============================
# 설정값
# =============================
MEASURE_INTERVAL_MIN = 1                                     # 센서 측정 대기 시간
CAFFEINE_DB = {"아메리카노": 100}                             # 입력받을 음료
CAFFEINE_JUMP_RELAX_SEC = 600                                # 카페인 섭취 후 HR 점프 필터를 끄는 시간(초) (= 10분)
last_caffeine_time = None                                    # 마지막 카페인 섭취 시각(전역)

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
HR_buffer = []                                             # 최근 HR값들 저장하는 리스트
HR_BUFFER_SIZE = 3                                         # 몇 개의 HR 값들을 리스트에 저장할지
HR_JUMP_LIMIT = 25                                         # HR값의 어느 정도 차이를 무시할지

# =============================
# 개인화 가중치
# =============================
W_ALPHA = 0.3   # 형성률 개인화 반영 비율 (0~1)
W_BETA  = 0.5   # 회복률 개인화 반영 비율 (0~1)

# 기준값 (대략적인 레퍼런스 스케일)                           # S값 과 베타 값이 얼마나 큰지 작은지 비교하는 기준값, 수면 예측 단계에서 비교한 값 사용됨
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
    time_log.append(t_min)                               # 측정시간 분 단위, 그래프에서 x축
    HR_log.append(HR)                                    # 노이즈 안잡은 HR 값
    GSR_log.append(GSR)
    S_log.append(S)                                      # 계산된 S_value 값
    alpha_log.append(alpha)                              # 계산된 알파 값
    beta_log.append(beta)                                # 계산된 베타 값         
    C_log.append(C)                                      # 계산된 카페인 농도 값
    R_log.append(R)                                      # 계산된 각성도 값


# =============================
# 그래프 만드는 함수
# =============================
def plot_final_graph():
    plt.figure(figsize=(12, 7))                         # 가로 12 세로 7
    plt.plot(time_log, S_log, label="S-value")
    plt.plot(time_log, C_log, label="C(t)")
    plt.plot(time_log, R_log, label="R(t)")
    plt.plot(time_log, HR_log, label="HR (raw)")
    plt.xlabel("Time (min)")
    plt.ylabel("Value")
    plt.title("Real-time CaffSense Graph")
    plt.grid(True)                                     # 그래프 격자선
    plt.legend()                                       # 그래프 선 이름 창
    plt.tight_layout()                                 # 글자 안겹치고 안잘리게 자동으로  배치 조절
    plt.show()



########## 식 계산 시작 ##########


# =============================
# 1) S-value                                           # S-value: 현재 카페인 반응 민감도 지표
# =============================
def calculate_S(H_now, H_prev, G_now, G_prev, dt_seconds):
    if dt_seconds <= 0 or H_prev is None or G_prev is None:
        return 0.0
    
    dH_dt = (H_now - H_prev) / dt_seconds
    dG_dt = (G_now - G_prev) / dt_seconds

    return (0.6 * dH_dt + 0.4 * dG_dt) * 10.0         # 원래 S_value 값이 너무 작아서 보기 편하게 10을 곱해 스케일링     


# =============================
# 2) α 형성률                                          # α: 카페인 효과(각성도)를 얼마나 빨리 느끼는지
# =============================                       # 감마를 쓰는게 지수평활(EWMA) 개념임
def update_alpha(alpha_prev, S):                      # 감마를 쓰는 이유: 카페인 효과가 빠르게 올라가고 천천히 내려가는 생리학 특성을 반영하기 위해
    gamma_rise = 0.2                                  # 
    gamma_decay = 0.05                                # 각성 감소를 천천히 하기 위해
                                                      # 식 구조 이유: 누적 반응(알파)에 순간적인 변화(S_value)를 반영해 더해가는 구조(사실상 알파는 S의 누적 반응임)
    if S > 0:                                         # 각성 효과가 상승 중(S > 0)일 때 이전 알파 값을 베이스(80% 반영)로 하고 S 상승률 반영
        return (1 - gamma_rise) * alpha_prev + gamma_rise * S
    else:                                             # 각성 효과가 하강 중(S < 0)일 때 이전 알파 값을 베이스(80% 반영)로 하여 일부 반영
        return (1 - gamma_decay) * alpha_prev

"""
사실 HR센서와 GSR센서는 엄청 예민한 센서이어서 그 값을 그대로 받은 S_value는 매우 민감하고 튀기 쉬움(ex HR-손가락 위치와 압력, GSR-땀과 습도)
그래서 알파에 S를 그대로 반영한다면 알파 또한 값이 터져버리고 알파가 사용되는 R식도 같이 터지면서 수면 예측이 불가능해짐

gamma_rise = 0.2인 이유는 이게 지수평활(EWMA) 개념 때문
지수평활이란 최근 값을 많이 반영하고 이전 값을 조금 반영하는 가중 평균 방식

자세히 말하자면 새로운 값들을 조금씩 계속 넣어 과거의 값들은 점점 더 비중이 작아지게 하고 최신의 값들의 비중이 커지게 하는 방식
(과거로 갈수록 반영 비율이 지수적으로 줄어들어서 지수평활이라고 부름)
"""

# =============================
# 3) β 회복률                                          # β: 카페인 효과가 얼마나 빨리 떨어지는지, 즉 얼마나 빨리 회복하는지
# =============================
def calculate_beta(H_peak, H_now, G_peak, G_now, t_peak, t_now):

    if H_peak is None or G_peak is None:
        return 0.0

    dt = t_now - t_peak
    if dt <= 0:
        return 0.0

    try:                                             # HR, GSR이 peak 값에서 어느 속도로 내려오는지
        beta_hr = (math.log(H_peak) - math.log(H_now)) / dt
        beta_gs = (math.log(G_peak) - math.log(G_now)) / dt
    except:
        return 0.0

    return 10.0 * (0.6 * beta_hr + 0.4 * beta_gs)   # HR이  GSR보다 더 안정적이어서 가중치를 6:4로 함


# =============================
# 4) C(t)                                           # C(t): 카페인 농도(반감기 모델)
# =============================
def update_caffeine(C_prev, new_doses, dt, S_prev, S_now):
    if dt <= 0:
        return C_prev + sum(new_doses)

    HALF_LIFE_HOURS = 5.0                          # 일반적인 성인의 기준 반감기는 5시간임
    k = math.log(2) / HALF_LIFE_HOURS              # 개인화는 못했지만 공식 자체는 약동학 모델에서 많이 쓰는 지수감소 반감기 공식임
    dt_hours = dt / 3600.0

    C_decay = C_prev * math.exp(-k * dt_hours)     # 카페인 농도 식(시간 단위), 지수적으로 감소
    C_add = sum(new_doses)

    return C_decay + C_add


# =============================
# 5) R(t)                                          # R(t): 현재 몸의 각성도를 구하는 최종 식, 회복률인 베타가 사용되는데 베타가 시간 단위라 R도 시간 단위로 해야함
# =============================
def calculate_R(alpha, beta, C, t_min):
    t_hr = t_min / 60.0                            # 코드가 작동 시작한 시점부터의 시간
    return alpha * C - beta * t_hr                 # alpha * C(각성 상승, 즉 현재 카페인 효과가 얼마나 크게 작용하고 있는지를 수치화한 값) - beta * t_hr(각성 감소)

                                                   # 센서 -> S -> 알파, 베타 -> R (R은 엄청 중요한 식임 ㅇㅇ), R을 구하기까지의 식들에서 노이즈를 계속해서 잡았기 때문에 노이즈에 강함
"""
# R값 해석:
R이 큼 -> 각성 효과가 강하고 수면 불가능, 각성 유지
R이 낮아짐 -> 각성 효과가 점점 사라짐, 수면 가능성이 생김
R <= 임계값(예: 3.0) -> 수면 가능 시간 도달
"""


# =============================
# 6) 개인 반감기                                    # 사용자의 S_value 패턴을 보고 카페인 반감기(half_life)를 개인화하는 함수
# =============================
def calculate_personal_half_life(S_history):
    if len(S_history) == 0:                       # S값이 없으면(첫 번째 측정) 기본 반감기인 5시간으로 사용
        return 5.0

    S_avg = sum(S_history) / len(S_history)                          # S_value 평균
    personal_half_life = 5.0 / (1 + 0.1 * abs(S_avg))                # S_avg의 값이 0.5 이하냐 이상이냐에 따라 반감기가 5시간(평균)에 가까운지 5시간보다 빠른지(분해가 빠른 사람) 개인화된 카페인 반감기 시간
    personal_half_life = max(2.0, min(7.0, personal_half_life))      # 최소(2) 최대(7) 설정
    return personal_half_life                     # 카페인 농도 C 식에 안쓰는 이유: S_history를 뽑아내기에는 측정시간이 길어야 하고 노이즈가 입력된다면 R도 바로 의미 없는 값이 되어버리고 수면 에측 또한 의미 없어짐


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
    global last_caffeine_time
    print("\n=== CaffSense (개선된 HR 안정화 + 개인화 가중 버전) ===\n")

    C = 0.0
    t_min = 0

    # HR 관련 이전 값들 분리 관리
    prev_raw_HR = None      # 원본 HR
    prev_HR_s = None        # 스무딩된 HR (S 전용)

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
            now_drink_time = time.time()
            drink_log.append((now_drink_time, dose))
            # 카페인 섭취 시각을 전역 변수에 기록 → 이후 10분 동안 HR 점프 필터 비활성화
            last_caffeine_time = now_drink_time
            print(f"[INFO] +{dose} mg 추가")

        # ---- 원본 HR 측정 ----
        raw_HR = measure_hr(10)
        G_now = measure_gsr()
        t_now = time.time()

        # ---- 스무딩 HR (S 계산용) ----
        HR_s = smooth_hr(raw_HR, prev_raw_HR)
        print(f"Raw HR = {raw_HR}, Smoothed HR (for S) = {HR_s}")

        # HR 측정 실패 처리 (원본 HR 기준)
        if raw_HR == 0.0:
            if prev_raw_HR is not None:
                raw_HR = prev_raw_HR
                HR_s = HR_s if HR_s != 0.0 else prev_HR_s if prev_HR_s is not None else prev_raw_HR
            else:
                # 첫 회차부터 HR이 0이면 그냥 카페인만 누적하고 스킵
                C = C + sum(new_doses)
                prev_time = t_now
                t_min += MEASURE_INTERVAL_MIN
                print("\nHR 측정 실패, 1분 대기...\n")
                time.sleep(60)
                continue

        # ---- 시간 간격 및 S, α, β 계산 ----
        if prev_time is None:
            # 첫 측정 회차: 기준 설정만
            dt = 60
            H_peak = raw_HR
            G_peak = G_now
            t_peak = t_now
            S_now = 0.0
            beta = 0.0
        else:
            dt = t_now - prev_time
            # S 계산에만 smoothed HR 사용
            S_now = calculate_S(HR_s, prev_HR_s, G_now, prev_GSR, dt)
            alpha = update_alpha(alpha, S_now)

            # 피크 및 β 계산은 raw HR 기반
            if H_peak is None or raw_HR > H_peak:
                H_peak = raw_HR
                G_peak = G_now
                t_peak = t_now

            beta = calculate_beta(H_peak, raw_HR, G_peak, G_now, t_peak, t_now)

        print(f"S-value = {S_now}")
        print(f"α = {alpha}")
        print(f"β = {beta}")

        C = update_caffeine(C, new_doses, dt, prev_S, S_now)
        print(f"C(t) = {C}")

        R = calculate_R(alpha, beta, C, t_min)
        print(f"R(t) = {R}")

        # 그래프용 HR은 raw HR 사용
        save_and_plot(t_min, raw_HR, G_now, S_now, alpha, beta, C, R)

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

        # ----- 이전 값 업데이트 -----
        prev_raw_HR = raw_HR
        prev_HR_s = HR_s
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
