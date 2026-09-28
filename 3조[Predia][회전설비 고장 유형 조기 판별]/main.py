# ============================================================
# 회전설비 고장 유형 조기 판별 및 모델링 통합 분석 코드
# Dataset: AI4I 2020 Predictive Maintenance Dataset
# ============================================================

# 0. 라이브러리 임포트
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_auc_score
)

# 한글 폰트 설정 (Windows 환경)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 경로 설정 (어느 터미널 경로에서 실행하든 항상 이 파일의 위치를 기준)
base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
figures_dir = os.path.join(base_dir, "figures")
results_dir = os.path.join(base_dir, "results")

# 결과 및 피규어 저장용 폴더 생성
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# 1. 데이터 불러오기 및 기본 확인
data_path = os.path.join(base_dir, "data", "ai4i2020.csv")
df = pd.read_csv(data_path, encoding="utf-8-sig")

print("===== 데이터 크기 =====")
print(df.shape)

print("\n===== 컬럼 =====")
print(df.columns.tolist())

print("\n===== 상위 데이터 =====")
print(df.head())

# 2. 데이터 품질 점검 및 기본 전처리 확인
print("\n===== 결측치 =====")
print(df.isnull().sum())

print("\n===== 중복 데이터 =====")
print(df.duplicated().sum())

print("\n===== 데이터 타입 =====")
print(df.dtypes)

print("\n===== Machine failure 분포 =====")
print(df["Machine failure"].value_counts())
print(df["Machine failure"].value_counts(normalize=True))

print("\n===== Type 분포 =====")
print(df["Type"].value_counts())


# 3. 정상 / 이상 정의 및 고장 유형 확인
df["상태"] = np.where(
    df["Machine failure"] == 0,
    "정상",
    "이상"
)

print("\n===== 정상 / 이상 건수 =====")
print(df["상태"].value_counts())

print("\n===== 정상 / 이상 비율 =====")
print(
    df["상태"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

failure_types = ["TWF", "HDF", "PWF", "OSF", "RNF"]

print("\n===== 고장 유형별 발생 건수 =====")
failure_count = df[failure_types].sum().sort_values(ascending=False)
print(failure_count)

print("\n===== 고장 유형별 발생률(%) =====")
print(
    df[failure_types]
    .mean()
    .mul(100)
    .round(2)
)

df["고장유형_개수"] = df[failure_types].sum(axis=1)
# 2번 코드와의 호환성을 위해 대소문자 별칭 추가 부여
df["Failure_count"] = df["고장유형_개수"]

print("\n===== 한 데이터에 포함된 고장 유형 개수 =====")
print(df["고장유형_개수"].value_counts().sort_index())

print("\n===== Machine failure=1인데 유형 라벨이 없는 데이터 =====")
print(
    (
        (df["Machine failure"] == 1) &
        (df[failure_types].sum(axis=1) == 0)
    ).sum()
)

case_a = df[
    (df["Machine failure"] == 0) &
    (df["Failure_count"] > 0)
]
print("\nMachine failure=0인데 고장유형 존재:", len(case_a))
print(case_a[failure_types].sum())


# 4. 파생변수 생성 및 Feature Engineering (1번 및 2번 통합)
# 온도 차이
df["Temp_diff"] = (
    df["Process temperature [K]"]
    - df["Air temperature [K]"]
)
df["Temp_Diff"] = df["Temp_diff"]  # 대소문자 양립 호환

# 기계적 출력(kW) 및 2번 코드의 기계적 출력/마모 토크 계산
df["Power [kW]"] = (
    df["Rotational speed [rpm]"]
    * df["Torque [Nm]"]
    * 2 * np.pi / 60
    / 1000
)
df["Mechanical_Power"] = df["Torque [Nm]"] * (2 * np.pi * df["Rotational speed [rpm]"] / 60)
df["Wear_Torque"] = df["Tool wear [min]"] * df["Torque [Nm]"]

# 롤링 및 시계열 성격 파생변수 (2번 코드 기능)
window = 10
df["Torque_MA"] = df["Torque [Nm]"].rolling(window=window).mean()
df["Torque_STD"] = df["Torque [Nm]"].rolling(window=window).std()
df["RPM_Diff"] = df["Rotational speed [rpm]"].diff()

print("\n===== 파생변수 및 Feature Engineering 완료 =====")


# 5. 주요 분석 변수 정의 및 기본 통계 출력
feature_cols = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Temp_diff",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Power [kW]",
    "Tool wear [min]"
]

print("\n===== 정상 데이터 평균 =====")
print(
    df.loc[df["Machine failure"] == 0, feature_cols]
    .mean()
    .round(3)
)

print("\n===== 이상 데이터 평균 =====")
print(
    df.loc[df["Machine failure"] == 1, feature_cols]
    .mean()
    .round(3)
)


# 6. 정상 vs 이상 분포 비교 시각화 (1번 코드)
normal = df[df["Machine failure"] == 0]
failure = df[df["Machine failure"] == 1]

for col in feature_cols:
    plt.figure(figsize=(8, 4))
    plt.hist(normal[col], bins=30, alpha=0.6, label="정상")
    plt.hist(failure[col], bins=30, alpha=0.6, label="이상")

    plt.title(f"{col} - 정상 vs 이상")
    plt.xlabel(col)
    plt.ylabel("데이터 수")
    plt.legend()
    plt.tight_layout()

    safe_name = col.replace("/", "_").replace("[", "").replace("]", "").replace(" ", "_").replace(":", "_")
    save_path = os.path.join(figures_dir, f"분포_{safe_name}.png")
    
    plt.savefig(save_path, dpi=150)
    plt.close()


# 7. 상관관계 분석 (1번 코드)
corr_cols = feature_cols + ["Machine failure"]
corr = df[corr_cols].corr()

print("\n===== 상관관계 =====")
print(corr["Machine failure"].sort_values(ascending=False).round(3))

plt.figure(figsize=(10, 7))
plt.imshow(corr, cmap="coolwarm", aspect="auto")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=70)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("주요 변수 상관관계")
plt.tight_layout()

plt.savefig(os.path.join(figures_dir, "상관관계.png"), dpi=150)
plt.show()
plt.close()


# 8. 공구 마모시간을 이용한 진행 단계 분석 (1번 코드)
df["Tool_wear_bin"] = pd.cut(
    df["Tool wear [min]"],
    bins=[-1, 50, 100, 150, 200, 250, np.inf],
    labels=["0~50", "51~100", "101~150", "151~200", "201~250", "251+"]
)

wear_summary = (
    df.groupby("Tool_wear_bin", observed=False)
    .agg(
        데이터수=("Machine failure", "size"),
        고장수=("Machine failure", "sum"),
        고장률=("Machine failure", "mean")
    )
)
wear_summary["고장률"] = (wear_summary["고장률"] * 100).round(2)

print("\n===== 공구 마모 구간별 고장률 =====")
print(wear_summary)
wear_summary.to_csv(os.path.join(figures_dir, "공구마모_구간별_고장률.csv"), encoding="utf-8-sig")

plt.figure(figsize=(8, 4))
plt.plot(wear_summary.index.astype(str), wear_summary["고장률"], marker="o")
plt.title("공구 마모 진행 단계별 고장률")
plt.xlabel("Tool wear [min] 구간")
plt.ylabel("고장률 (%)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "공구마모_구간별_고장률.png"), dpi=150)
plt.show()
plt.close()


# 9. 고장 유형별 주요 변수 비교 (1번 코드)
type_mean_result = []
for ft in failure_types:
    normal_ft = df[df[ft] == 0][feature_cols].mean()
    abnormal_ft = df[df[ft] == 1][feature_cols].mean()

    for col in feature_cols:
        type_mean_result.append({
            "고장유형": ft,
            "변수": col,
            "정상평균": normal_ft[col],
            "고장평균": abnormal_ft[col],
            "차이": abnormal_ft[col] - normal_ft[col]
        })

type_mean_df = pd.DataFrame(type_mean_result)
type_mean_df.to_csv(os.path.join(figures_dir, "고장유형별_변수비교.csv"), index=False, encoding="utf-8-sig")


# 10. Z-score 기반 단변량 이상 탐지 (1번 코드)
z_result = []
for col in feature_cols:
    mean_value = normal[col].mean()
    std_value = normal[col].std()

    if std_value == 0:
        continue

    df[f"Z_{col}"] = (df[col] - mean_value) / std_value
    df[f"Z이상_{col}"] = (df[f"Z_{col}"].abs() >= 3).astype(int)

    for ft in ["Machine failure"] + failure_types:
        target = df["Machine failure"] if ft == "Machine failure" else df[ft]
        detected = ((df[f"Z이상_{col}"] == 1) & (target == 1)).sum()
        total = target.sum()
        recall = detected / total if total > 0 else 0

        z_result.append({
            "대상": ft,
            "변수": col,
            "Z>=3 이상후보수": int(df[f"Z이상_{col}"].sum()),
            "고장검출수": int(detected),
            "검출률(Recall)": round(recall, 3)
        })

z_result_df = pd.DataFrame(z_result)
z_result_df.to_csv(os.path.join(figures_dir, "Zscore_이상탐지결과.csv"), index=False, encoding="utf-8-sig")


# 11. 2번 코드 전용 고급 시각화 플롯 (토크 트렌드 및 고장 지점 그래프)
data_clean = df.dropna().reset_index(drop=True)
failure_points = data_clean[data_clean["Machine failure"] == 1]

# [그림 1] Torque Trend and Machine Failure
plt.figure(figsize=(12, 4.5))
plt.plot(data_clean["UDI"], data_clean["Torque [Nm]"], label="Torque", color="#1f77b4", linewidth=0.8, alpha=0.6)
plt.plot(data_clean["UDI"], data_clean["Torque_MA"], label="Torque Rolling Average", color="#ff7f0e", linewidth=1.5)
plt.scatter(failure_points["UDI"], failure_points["Torque [Nm]"], 
            color="#1f77b4", marker="x", s=40, label="Machine Failure", zorder=5)
plt.title("Torque Trend and Machine Failure", fontsize=11, fontweight="bold")
plt.xlabel("UDI (Progress Order)", fontsize=9)
plt.ylabel("Torque [Nm]", fontsize=9)
plt.legend(loc="upper right", fontsize=8)
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig(os.path.join(figures_dir, "torque_trend_and_machine_failure.png"), dpi=300, bbox_inches="tight")
plt.close()

# [그림 2] Torque Variability and Machine Failure
plt.figure(figsize=(12, 4.5))
plt.plot(data_clean["UDI"], data_clean["Torque_STD"], label="Torque Rolling STD", color="#1f77b4", linewidth=1.2)
plt.scatter(failure_points["UDI"], failure_points["Torque_STD"], 
            color="#1f77b4", marker="x", s=40, label="Machine Failure", zorder=5)
plt.title("Torque Variability and Machine Failure", fontsize=11, fontweight="bold")
plt.xlabel("UDI (Progress Order)", fontsize=9)
plt.ylabel("Torque Rolling STD", fontsize=9)
plt.legend(loc="upper right", fontsize=8)
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig(os.path.join(figures_dir, "torque_variability_and_machine_failure.png"), dpi=300, bbox_inches="tight")
plt.close()

# [그림 3] RPM Change and Machine Failure
plt.figure(figsize=(12, 4.5))
plt.plot(data_clean["UDI"], data_clean["RPM_Diff"], label="RPM Diff (1-step)", color="#1f77b4", linewidth=1.0, alpha=0.7)
plt.axhline(0, color="#aec7e8", linestyle="--", linewidth=1)
plt.scatter(failure_points["UDI"], failure_points["RPM_Diff"], 
            color="#1f77b4", marker="x", s=40, label="Machine Failure", zorder=5)
plt.title("RPM Change and Machine Failure", fontsize=11, fontweight="bold")
plt.xlabel("UDI (Progress Order)", fontsize=9)
plt.ylabel("RPM Difference [rpm]", fontsize=9)
plt.legend(loc="upper right", fontsize=8)
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig(os.path.join(figures_dir, "rpm_change_and_machine_failure.png"), dpi=300, bbox_inches="tight")
plt.close()

# [추가] False Negative Case around UDI 8027 그래프
fn_subset = data_clean[(data_clean["UDI"] >= 8005) & (data_clean["UDI"] <= 8047)]
if not fn_subset.empty:
    plt.figure(figsize=(12, 5))
    plt.plot(fn_subset["UDI"], fn_subset["Torque [Nm]"], marker='o', color="#1f77b4", linewidth=1.2, label="Torque")
    plt.axvline(8027, color="#1f77b4", linestyle="--", linewidth=1.5, label="False Negative")
    plt.title("False Negative Case around UDI 8027", fontsize=12, fontweight="bold")
    plt.xlabel("UDI (Progress Order)", fontsize=10)
    plt.ylabel("Torque [Nm]", fontsize=10)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(figures_dir, "false_negative_case_udi_8027.png"), dpi=300, bbox_inches="tight")
    plt.close()


# 12. 머신러닝 모델용 데이터 구성 및 학습 (1번 코드 기반 모델링 프로세스 유지)
model_df = df.copy()
type_dummies = pd.get_dummies(model_df["Type"], prefix="Type", drop_first=False, dtype=int)

X = pd.concat([
    model_df[feature_cols].reset_index(drop=True),
    type_dummies.reset_index(drop=True)
], axis=1)

print("\n===== 모델 입력 변수 =====")
print(X.columns.tolist())

# 13. 정상 / 이상 분류 모델
y = model_df["Machine failure"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

failure_model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1)
failure_model.fit(X_train, y_train)

y_pred = failure_model.predict(X_test)
y_prob = failure_model.predict_proba(X_test)[:, 1]

print("\n========================================")
print(" 정상 / 이상 분류 결과")
print("========================================")
print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))
print("Recall   :", round(recall_score(y_test, y_pred, zero_division=0), 4))
print("F1-score :", round(f1_score(y_test, y_pred, zero_division=0), 4))
print("ROC-AUC  :", round(roc_auc_score(y_test, y_prob), 4))

print("\n===== Classification Report =====")
print(classification_report(y_test, y_pred, target_names=["정상", "이상"], zero_division=0))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["정상", "이상"])
plt.title("정상 / 이상 분류 혼동행렬")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "정상_이상_혼동행렬.png"), dpi=150)
plt.show()
plt.close()


# 14. 정상 / 이상 분류 변수 중요도
importance_df = pd.DataFrame({
    "변수": X.columns,
    "중요도": failure_model.feature_importances_
}).sort_values("중요도", ascending=False)

importance_df.to_csv(os.path.join(figures_dir, "정상_이상_변수중요도.csv"), index=False, encoding="utf-8-sig")

plt.figure(figsize=(8, 5))
top_imp = importance_df.head(10).sort_values("중요도")
plt.barh(top_imp["변수"], top_imp["중요도"])
plt.title("정상 / 이상 분류 변수 중요도 TOP 10")
plt.xlabel("중요도")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "정상_이상_변수중요도.png"), dpi=150)
plt.show()
plt.close()


# 15. 고장 유형별 분류 모델 및 성능 비교 (1번 코드)
type_model_results = []
type_importance_results = []

for ft in failure_types:
    y_type = model_df[ft]
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X, y_type, test_size=0.2, random_state=42, stratify=y_type)

    model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1)
    model.fit(X_train_t, y_train_t)

    pred_t = model.predict(X_test_t)
    prob_t = model.predict_proba(X_test_t)[:, 1]

    acc = accuracy_score(y_test_t, pred_t)
    pre = precision_score(y_test_t, pred_t, zero_division=0)
    rec = recall_score(y_test_t, pred_t, zero_division=0)
    f1 = f1_score(y_test_t, pred_t, zero_division=0)

    try:
        auc = roc_auc_score(y_test_t, prob_t)
    except ValueError:
        auc = np.nan

    type_model_results.append({
        "고장유형": ft, "Accuracy": acc, "Precision": pre, "Recall": rec, "F1": f1, "ROC-AUC": auc
    })

    ft_imp = pd.DataFrame({
        "고장유형": ft, "변수": X.columns, "중요도": model.feature_importances_
    }).sort_values("중요도", ascending=False)
    type_importance_results.append(ft_imp)

    ConfusionMatrixDisplay.from_predictions(y_test_t, pred_t, display_labels=["정상", "이상"])
    plt.title(f"{ft} 고장 분류 혼동행렬")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"{ft}_혼동행렬.png"), dpi=150)
    plt.show()
    plt.close()

type_result_df = pd.DataFrame(type_model_results)
type_result_df.to_csv(os.path.join(figures_dir, "고장유형별_모델성능.csv"), index=False, encoding="utf-8-sig")

all_type_importance = pd.concat(type_importance_results, ignore_index=True)
all_type_importance.to_csv(os.path.join(figures_dir, "고장유형별_변수중요도.csv"), index=False, encoding="utf-8-sig")

for ft in failure_types:
    temp = all_type_importance[all_type_importance["고장유형"] == ft].sort_values("중요도", ascending=False).head(7).sort_values("중요도")

    plt.figure(figsize=(8, 5))
    plt.barh(temp["변수"], temp["중요도"])
    plt.title(f"{ft} 주요 변수 중요도 TOP 7")
    plt.xlabel("중요도")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"{ft}_변수중요도.png"), dpi=150)
    plt.show()
    plt.close()


# 16. 최종 요약 및 저장
summary = [{
    "분석대상": "Machine failure", "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0), "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1": f1_score(y_test, y_pred, zero_division=0), "ROC-AUC": roc_auc_score(y_test, y_prob)
}]

for _, row in type_result_df.iterrows():
    summary.append({
        "분석대상": row["고장유형"], "Accuracy": row["Accuracy"], "Precision": row["Precision"],
        "Recall": row["Recall"], "F1": row["F1"], "ROC-AUC": row["ROC-AUC"]
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(figures_dir, "최종_모델성능_요약.csv"), index=False, encoding="utf-8-sig")

df.to_csv(os.path.join(figures_dir, "분석완료_데이터.csv"), index=False, encoding="utf-8-sig")

print("\n========================================")
print(" 모든 통합 분석 및 모델링 완료")
print("========================================")