# ============================================================
# 회전설비 고장 유형 조기 판별 분석 코드
# Dataset: AI4I 2020 Predictive Maintenance Dataset
# ============================================================

# 0. 라이브러리
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
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

# 한글 폰트 (Windows 환경)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# [수정됨] 절대 경로 설정 (어느 터미널 경로에서 실행하든 항상 이 파일의 위치를 기준으로 함)
base_dir = os.path.dirname(os.path.abspath(__file__))
# 결과 저장 폴더명을 figures로 변경
figures_dir = os.path.join(base_dir, "figures")

# 결과 저장용 폴더 생성
os.makedirs(figures_dir, exist_ok=True)

# 1. 데이터 불러오기
# [수정됨] 절대 경로로 data 폴더 내 파일 참조
data_path = os.path.join(base_dir, "data", "ai4i2020.csv")
df = pd.read_csv(data_path, encoding="utf-8-sig")

print("===== 데이터 크기 =====")
print(df.shape)

print("\n===== 컬럼 =====")
print(df.columns.tolist())

print("\n===== 상위 데이터 =====")
print(df.head())

# 2. 데이터 품질 점검
print("\n===== 결측치 =====")
print(df.isnull().sum())

print("\n===== 중복 데이터 =====")
print(df.duplicated().sum())

print("\n===== 데이터 타입 =====")
print(df.dtypes)


# 3. 정상 / 이상 정의
# 정상: Machine failure = 0
# 이상: Machine failure = 1
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


# 4. 고장 유형 확인
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

# 여러 고장 유형이 동시에 발생할 수 있으므로
# 유형별 분석은 각각의 이진 분류 문제로 진행한다.
df["고장유형_개수"] = df[failure_types].sum(axis=1)

print("\n===== 한 데이터에 포함된 고장 유형 개수 =====")
print(df["고장유형_개수"].value_counts().sort_index())

print("\n===== Machine failure=1인데 유형 라벨이 없는 데이터 =====")
print(
    (
        (df["Machine failure"] == 1) &
        (df[failure_types].sum(axis=1) == 0)
    ).sum()
)


# 5. 파생변수 생성
# 온도 차이
df["Temp_diff"] = (
    df["Process temperature [K]"]
    - df["Air temperature [K]"]
)

# 회전속도 + 토크로 계산한 기계적 출력(kW)
df["Power [kW]"] = (
    df["Rotational speed [rpm]"]
    * df["Torque [Nm]"]
    * 2 * np.pi / 60
    / 1000
)

print("\n===== 파생변수 생성 완료 =====")
print(df[[
    "Temp_diff",
    "Power [kW]"
]].describe().round(3))


# 6. 주요 분석 변수
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


# 7. 정상 vs 이상 분포 비교
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

    # [수정됨] 파일명에 사용할 수 없는 특수문자와 띄어쓰기를 모두 언더바로 치환
    safe_name = col.replace("/", "_").replace("[", "").replace("]", "").replace(" ", "_").replace(":", "_")
    save_path = os.path.join(figures_dir, f"분포_{safe_name}.png")
    
    plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close() # 메모리 관리를 위해 닫기 추가


# 8. 상관관계 분석
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

# [수정됨] 절대 경로로 저장
plt.savefig(os.path.join(figures_dir, "상관관계.png"), dpi=150)
plt.show()
plt.close()


# 9. 공구 마모시간을 이용한 진행 단계 분석
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

# [수정됨] 절대 경로로 CSV 저장
wear_summary.to_csv(os.path.join(figures_dir, "공구마모_구간별_고장률.csv"), encoding="utf-8-sig")

plt.figure(figsize=(8, 4))
plt.plot(wear_summary.index.astype(str), wear_summary["고장률"], marker="o")
plt.title("공구 마모 진행 단계별 고장률")
plt.xlabel("Tool wear [min] 구간")
plt.ylabel("고장률 (%)")
plt.xticks(rotation=30)
plt.tight_layout()

# [수정됨] 절대 경로로 저장
plt.savefig(os.path.join(figures_dir, "공구마모_구간별_고장률.png"), dpi=150)
plt.show()
plt.close()


# 10. 고장 유형별 주요 변수 비교
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

print("\n===== 고장 유형별 평균 차이 =====")
print(type_mean_df.round(3))

# [수정됨] 절대 경로로 CSV 저장
type_mean_df.to_csv(os.path.join(figures_dir, "고장유형별_변수비교.csv"), index=False, encoding="utf-8-sig")


# 11. Z-score 기반 단변량 이상 탐지
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

print("\n===== Z-score 이상 탐지 결과 =====")
print(z_result_df.sort_values(["대상", "검출률(Recall)"], ascending=[True, False]).head(30))

# [수정됨] 절대 경로로 CSV 저장
z_result_df.to_csv(os.path.join(figures_dir, "Zscore_이상탐지결과.csv"), index=False, encoding="utf-8-sig")


# 12. 모델용 데이터 구성
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

print("\n===== Confusion Matrix =====")
print(confusion_matrix(y_test, y_pred))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["정상", "이상"])
plt.title("정상 / 이상 분류 혼동행렬")
plt.tight_layout()

# [수정됨] 절대 경로로 저장
plt.savefig(os.path.join(figures_dir, "정상_이상_혼동행렬.png"), dpi=150)
plt.show()
plt.close()


# 14. 정상 / 이상 분류 변수 중요도
importance_df = pd.DataFrame({
    "변수": X.columns,
    "중요도": failure_model.feature_importances_
}).sort_values("중요도", ascending=False)

print("\n===== 정상 / 이상 분류 변수 중요도 =====")
print(importance_df.round(4))

# [수정됨] 절대 경로로 저장
importance_df.to_csv(os.path.join(figures_dir, "정상_이상_변수중요도.csv"), index=False, encoding="utf-8-sig")

plt.figure(figsize=(8, 5))
top_imp = importance_df.head(10).sort_values("중요도")

plt.barh(top_imp["변수"], top_imp["중요도"])
plt.title("정상 / 이상 분류 변수 중요도 TOP 10")
plt.xlabel("중요도")
plt.tight_layout()

# [수정됨] 절대 경로로 저장
plt.savefig(os.path.join(figures_dir, "정상_이상_변수중요도.png"), dpi=150)
plt.show()
plt.close()


# 15. 고장 유형별 분류 모델
type_model_results = []
type_importance_results = []

for ft in failure_types:
    print("\n========================================")
    print(f" {ft} 고장 분류")
    print("========================================")

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

    print("Accuracy :", round(acc, 4))
    print("Precision:", round(pre, 4))
    print("Recall   :", round(rec, 4))
    print("F1-score :", round(f1, 4))
    print("ROC-AUC  :", round(auc, 4) if not np.isnan(auc) else "N/A")

    print("\n혼동행렬")
    print(confusion_matrix(y_test_t, pred_t))

    ft_imp = pd.DataFrame({
        "고장유형": ft, "변수": X.columns, "중요도": model.feature_importances_
    }).sort_values("중요도", ascending=False)
    type_importance_results.append(ft_imp)

    ConfusionMatrixDisplay.from_predictions(y_test_t, pred_t, display_labels=["정상", "이상"])
    plt.title(f"{ft} 고장 분류 혼동행렬")
    plt.tight_layout()

    # [수정됨] 절대 경로로 저장
    plt.savefig(os.path.join(figures_dir, f"{ft}_혼동행렬.png"), dpi=150)
    plt.show()
    plt.close()


# 16. 고장 유형별 모델 성능 비교
type_result_df = pd.DataFrame(type_model_results)

print("\n========================================")
print(" 고장 유형별 모델 성능")
print("========================================")
print(type_result_df.round(4))

# [수정됨] 절대 경로로 저장
type_result_df.to_csv(os.path.join(figures_dir, "고장유형별_모델성능.csv"), index=False, encoding="utf-8-sig")


# 17. 고장 유형별 변수 중요도
all_type_importance = pd.concat(type_importance_results, ignore_index=True)
print("\n===== 고장 유형별 변수 중요도 TOP 5 =====")

for ft in failure_types:
    print(f"\n[{ft}]")
    print(all_type_importance[all_type_importance["고장유형"] == ft].sort_values("중요도", ascending=False).head(5).round(4))

# [수정됨] 절대 경로로 저장
all_type_importance.to_csv(os.path.join(figures_dir, "고장유형별_변수중요도.csv"), index=False, encoding="utf-8-sig")


# 18. 고장 유형별 변수 중요도 시각화
for ft in failure_types:
    temp = all_type_importance[all_type_importance["고장유형"] == ft].sort_values("중요도", ascending=False).head(7).sort_values("중요도")

    plt.figure(figsize=(8, 5))
    plt.barh(temp["변수"], temp["중요도"])
    plt.title(f"{ft} 주요 변수 중요도 TOP 7")
    plt.xlabel("중요도")
    plt.tight_layout()

    # [수정됨] 절대 경로로 저장
    plt.savefig(os.path.join(figures_dir, f"{ft}_변수중요도.png"), dpi=150)
    plt.show()
    plt.close()


# 19. 최종 분석 결과 요약표
summary = []

summary.append({
    "분석대상": "Machine failure", "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0), "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1": f1_score(y_test, y_pred, zero_division=0), "ROC-AUC": roc_auc_score(y_test, y_prob)
})

for _, row in type_result_df.iterrows():
    summary.append({
        "분석대상": row["고장유형"], "Accuracy": row["Accuracy"], "Precision": row["Precision"],
        "Recall": row["Recall"], "F1": row["F1"], "ROC-AUC": row["ROC-AUC"]
    })

summary_df = pd.DataFrame(summary)

print("\n========================================")
print(" 최종 모델 성능 요약")
print("========================================")
print(summary_df.round(4))

# [수정됨] 절대 경로로 저장
summary_df.to_csv(os.path.join(figures_dir, "최종_모델성능_요약.csv"), index=False, encoding="utf-8-sig")


# 20. 최종 결론에 사용할 주요 결과 자동 출력
top_failure_features = importance_df.head(5)[["변수", "중요도"]]

print("\n========================================")
print(" 최종 결론용 핵심 결과")
print("========================================")
print("\n[1] 정상/이상 분류 주요 변수")
print(top_failure_features.round(4))
print("\n[2] 고장 유형별 모델 성능")
print(type_result_df.round(4))
print("\n[3] 고장 유형별 주요 변수")

for ft in failure_types:
    top5 = all_type_importance[all_type_importance["고장유형"] == ft].sort_values("중요도", ascending=False).head(5)
    print(f"\n{ft}")
    print(top5[["변수", "중요도"]].round(4))


# 21. 분석 결과 저장
# [수정됨] 절대 경로로 저장
df.to_csv(os.path.join(figures_dir, "분석완료_데이터.csv"), index=False, encoding="utf-8-sig")

print("\n========================================")
print(" 모든 분석 완료")
print("========================================")
print("분석 결과는 figures 폴더에 정상적으로 저장되었습니다.")