# Idea Tries

## 코드 구성

- SparseMatrix:
  - CSR 형식으로 희소 행렬 표현 및 행렬-벡터 곱 지원.
  - .mtx 파일 읽기/쓰기 기능 포함.
- JacobiRelaxation:
  - Jacobi 반복법으로 Relaxation 수행.
- AMGLevel:
  - 연결 강도 기반 coarsening (ComputeStrongConnections).
  - 효율적인 aggregation (AggregateNodes).
  - 다중 smoothed interpolation (BuildInterpolationMatrix).
- AMGSolver:
  - 계층 생성 (BuildHierarchy) 및 V-Cycle (Solve, VCycle).
  - 잔차 출력으로 디버깅 지원.

### Test Problem:
- 열전달 문제
  - 16x16 열전달 문제 생성 및 해결.
  - 실행 시간과 잔차 출력.
