import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

# -----------------------------
# 1. 데이터 불러오기 및 기본 확인
# -----------------------------
base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
results_dir = os.path.join(base_dir, "results")
figures_dir = os.path.join(base_dir, "figures") # figures 폴더 경로 설정

# 결과 및 피규어 저장용 폴더 생성
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

data_path = os.path.join(base_dir, "data", "ai4i2020.csv")
df = pd.read_csv(data_path, encoding="utf-8-sig")

print("데이터 크기:", df.shape)
print(df.head())

print("\n컬럼 정보")
print(df.columns.tolist())

print("\n결측치")
print(df.isnull().sum())

print("\nMachine failure 분포")
print(df["Machine failure"].value_counts())
print(df["Machine failure"].value_counts(normalize=True))


# -----------------------------
# 2. 데이터 기본 전처리 확인
# -----------------------------
failure_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]

print("\n전체 중복 행:", df.duplicated().sum())

print("\nType 분포")
print(df["Type"].value_counts())

print("\n고장 유형별 발생 건수")
print(failure_cols)

df["Failure_count"] = df[failure_cols].sum(axis=1)

print("\n동시 고장 유형 개수 분포")
print(df["Failure_count"].value_counts().sort_index())

print("\nMachine failure=1 중 고장 유형 표시 개수")
print(df.loc[df["Machine failure"] == 1, "Failure_count"].value_counts().sort_index())

data = df.drop(columns=["UDI", "Product ID"]).copy()

print("\n분석용 데이터 크기:", data.shape)
print(data.head())


# -----------------------------
# 3. Machine failure과 고장 유형 라벨 관계 확인
# -----------------------------
case_a = df[
    (df["Machine failure"] == 0) &
    (df["Failure_count"] > 0)
]

case_b = df[
    (df["Machine failure"] == 1) &
    (df["Failure_count"] == 0)
]

print("\nMachine failure=0인데 고장유형 존재:", len(case_a))
print(case_a[failure_cols].sum())

print("\nMachine failure=1인데 고장유형 없음:", len(case_b))


# -----------------------------
# 4. Feature Engineering 및 핵심 그래프 figures 저장
# -----------------------------
data = df.copy()

# UDI를 다시 활용하기 위해 컬럼이 있는지 확인 후 포함하거나 복사
if "UDI" not in data.columns and "UDI" in df.columns:
    data["UDI"] = df["UDI"]

data = data.drop(columns=["Product ID"], errors="ignore")
data = pd.get_dummies(data, columns=["Type"], drop_first=True)

data["Temp_Diff"] = data["Process temperature [K]"] - data["Air temperature [K]"]
data["Mechanical_Power"] = data["Torque [Nm]"] * (2 * np.pi * data["Rotational speed [rpm]"] / 60)
data["Wear_Torque"] = data["Tool wear [min]"] * data["Torque [Nm]"]

window = 10
data["Torque_MA"] = data["Torque [Nm]"].rolling(window=window).mean()
data["Torque_STD"] = data["Torque [Nm]"].rolling(window=window).std()
data["RPM_Diff"] = data["Rotational speed [rpm]"].diff()

data = data.dropna().reset_index(drop=True)

new_features = [
    "Temp_Diff",
    "Mechanical_Power",
    "Wear_Torque",
    "Torque_MA",
    "Torque_STD",
    "RPM_Diff"
]

print("\nFeature Engineering 완료")
print("데이터 크기:", data.shape)

# 공통으로 사용할 고장 발생 지점 인덱스 필터링
failure_points = data[data["Machine failure"] == 1]

# ---------------------------------------------
# [그림 1] Torque Trend and Machine Failure
# ---------------------------------------------
plt.figure(figsize=(12, 4.5))
plt.plot(data["UDI"], data["Torque [Nm]"], label="Torque", color="#1f77b4", linewidth=0.8, alpha=0.6)
plt.plot(data["UDI"], data["Torque_MA"], label="Torque Rolling Average", color="#ff7f0e", linewidth=1.5)
plt.scatter(failure_points["UDI"], failure_points["Torque [Nm]"], 
            color="#1f77b4", marker="x", s=40, label="Machine Failure", zorder=5)
plt.title("Torque Trend and Machine Failure", fontsize=11, fontweight="bold")
plt.xlabel("UDI (Progress Order)", fontsize=9)
plt.ylabel("Torque [Nm]", fontsize=9)
plt.legend(loc="upper right", fontsize=8)
plt.grid(True, linestyle="--", alpha=0.5)

fig_path_1 = os.path.join(figures_dir, "torque_trend_and_machine_failure.png")
plt.savefig(fig_path_1, dpi=300, bbox_inches="tight")
plt.close()
print(f"그림 1 저장 완료: {fig_path_1}")

# ---------------------------------------------
# [그림 2] Torque Variability and Machine Failure
# ---------------------------------------------
plt.figure(figsize=(12, 4.5))
plt.plot(data["UDI"], data["Torque_STD"], label="Torque Rolling STD", color="#1f77b4", linewidth=1.2)
plt.scatter(failure_points["UDI"], failure_points["Torque_STD"], 
            color="#1f77b4", marker="x", s=40, label="Machine Failure", zorder=5)
plt.title("Torque Variability and Machine Failure", fontsize=11, fontweight="bold")
plt.xlabel("UDI (Progress Order)", fontsize=9)
plt.ylabel("Torque Rolling STD", fontsize=9)
plt.legend(loc="upper right", fontsize=8)
plt.grid(True, linestyle="--", alpha=0.5)

fig_path_2 = os.path.join(figures_dir, "torque_variability_and_machine_failure.png")
plt.savefig(fig_path_2, dpi=300, bbox_inches="tight")
plt.close()
print(f"그림 2 저장 완료: {fig_path_2}")

# ---------------------------------------------
# [그림 3] RPM Change and Machine Failure
# ---------------------------------------------
plt.figure(figsize=(12, 4.5))
plt.plot(data["UDI"], data["RPM_Diff"], label="RPM Diff (1-step)", color="#1f77b4", linewidth=1.0, alpha=0.7)
plt.axhline(0, color="#aec7e8", linestyle="--", linewidth=1)
plt.scatter(failure_points["UDI"], failure_points["RPM_Diff"], 
            color="#1f77b4", marker="x", s=40, label="Machine Failure", zorder=5)
plt.title("RPM Change and Machine Failure", fontsize=11, fontweight="bold")
plt.xlabel("UDI (Progress Order)", fontsize=9)
plt.ylabel("RPM Difference [rpm]", fontsize=9)
plt.legend(loc="upper right", fontsize=8)
plt.grid(True, linestyle="--", alpha=0.5)

fig_path_3 = os.path.join(figures_dir, "rpm_change_and_machine_failure.png")
plt.savefig(fig_path_3, dpi=300, bbox_inches="tight")
plt.close()
print(f"그림 3 저장 완료: {fig_path_3}")

# ---------------------------------------------
# [추가] False Negative Case around UDI 8027 그래프
# ---------------------------------------------
# UDI 8005 ~ 8047 구간 데이터 필터링 (이미지 참고)
fn_subset = data[(data["UDI"] >= 8005) & (data["UDI"] <= 8047)]

plt.figure(figsize=(12, 5))
plt.plot(fn_subset["UDI"], fn_subset["Torque [Nm]"], marker='o', color="#1f77b4", linewidth=1.2, label="Torque")
plt.axvline(8027, color="#1f77b4", linestyle="--", linewidth=1.5, label="False Negative")

plt.title("False Negative Case around UDI 8027", fontsize=12, fontweight="bold")
plt.xlabel("UDI (Progress Order)", fontsize=10)
plt.ylabel("Torque [Nm]", fontsize=10)
plt.legend(loc="upper right")
plt.grid(True, linestyle="--", alpha=0.5)

fig_path_fn = os.path.join(figures_dir, "false_negative_case_udi_8027.png")
plt.savefig(fig_path_fn, dpi=300, bbox_inches="tight")
plt.close()
print(f"False Negative Case 그래프 저장 완료: {fig_path_fn}")


# -----------------------------
# 5. Train / Test 분할
# -----------------------------
target = "Machine failure"

drop_cols = [
    "Machine failure",
    "TWF", "HDF", "PWF", "OSF", "RNF",
    "Failure_count",
    "UDI"
]

X = data.drop(columns=[col for col in drop_cols if col in data.columns])
y = data[target]

split_idx = int(len(data) * 0.8)

X_train = X.iloc[:split_idx].copy()
X_test  = X.iloc[split_idx:].copy()

y_train = y.iloc[:split_idx].copy()
y_test  = y.iloc[split_idx:].copy()

print("\n전체 데이터:", len(data))
print("Train:", len(X_train))
print("Test :", len(X_test))

print("\nTrain 고장 분포")
print(y_train.value_counts())

print("\nTest 고장 분포")
print(y_test.value_counts())

print("\n최종 Feature 수:", X_train.shape[1])
print(X_train.columns.tolist())