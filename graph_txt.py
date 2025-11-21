import re
import matplotlib.pyplot as plt

# ===== txt 파일에서 로그 읽기 =====
with open("log.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

def parse_blocks(text):
    time_log = [int(t) for t in re.findall(r"--- (\d+)분 경과 측정 ---", text)]
    S_log    = [float(s) for s in re.findall(r"S-value = ([\-\d\.]+)", text)]
    C_log    = [float(c) for c in re.findall(r"C\(t\) = ([\-\d\.]+)", text)]
    R_log    = [float(r) for r in re.findall(r"R\(t\) = ([\-\d\.]+)", text)]

    L = min(len(time_log), len(S_log), len(C_log), len(R_log))
    return time_log[:L], S_log[:L], C_log[:L], R_log[:L]

def visualize(time_log, S_log, C_log, R_log):
    plt.figure(figsize=(14,7))
    plt.plot(time_log, S_log, label="S-value", linewidth=2)
    plt.plot(time_log, C_log, label="C(t)", linewidth=2)
    plt.plot(time_log, R_log, label="R(t)", linewidth=2)
    plt.xlabel("Time (min)")
    plt.ylabel("Values")
    plt.title("S / C(t) / R(t) Combined Graph")
    plt.legend()
    plt.grid(True)
    plt.show()

time_log, S_log, C_log, R_log = parse_blocks(raw_text)
visualize(time_log, S_log, C_log, R_log)
