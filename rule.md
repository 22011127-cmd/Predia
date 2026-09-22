# 2. 규칙

| 브랜치 | 용도 | 예시 | 상세 |
| --- | --- | --- | --- |
| `main` | 최종 배포/제출용 | `main` | 수정 완료된 코드 병합 |
| `feature/` | 새로운 기능 개발 | `feature/time-series` | 변수별 분석 방법 |
| `experiment/` | model 생성 | `experiment/randforest` | 모델 생성 |
| `fix/` | 오류 수정 | `fix/missing-value` | feature/experiment의 코드 수정 작업 |
| `docs/` | 문서 작성/수정 | `docs/data-dictionary` | 결과 보고서 작성 |

폴더 구조 및 규칙

1. 데이터셋은 data폴더 안에 넣을 것 (read_csv 경로 : “data/파일명.csv”)
2. 변수 · 함수명 축약 없이 직관적인 단어로 선정할 것
3. 코드에 주석 꼼꼼히 작성할 것
4. LLM 사용 시 기존 코드와 충돌하지 않게 디버깅 할 것 (AI답변 복붙 지양)