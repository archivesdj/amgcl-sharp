# amgcl-sharp

## AMGCL?

[AMGCL](https://github.com/ddemidov/amgcl)은 대규모 희소 선형 시스템(sparse linear systems)을 효율적으로 해결하기 위한 C++ 헤더 전용 라이브러리입니다. 이 라이브러리는 대수적 다중 격자 방법(Algebraic Multigrid, AMG)을 구현하며, 특히 구조화되지 않은 격자(unstructured grids)에서 편미분 방정식(PDE)을 이산화할 때 발생하는 방정식 시스템을 풀기에 적합합니다. AMG는 기하학적 정보 없이도 "블랙박스 솔버"로 활용될 수 있어 다양한 계산 문제에 유용하며, 종종 독립적인 솔버로 사용되기보다는 반복 솔버(예: Conjugate Gradients, BiCGStab, GMRES)의 전제조건자(preconditioner)로 사용됩니다.

### 주요 특징

- 헤더 전용 설계: 외부 종속성을 최소화하며, 설치가 간편합니다. Boost 라이브러리 외에 추가적인 의존성이 거의 없습니다.
- 다양한 백엔드 지원: AMGCL은 CPU에서 AMG 계층을 구축한 후 이를 다양한 백엔드로 전송해 솔루션 단계를 가속화합니다. 지원되는 백엔드는 다음과 같습니다:
  - OpenMP: 멀티스레딩 CPU 병렬 처리를 지원.
  - OpenCL: GPU 및 기타 병렬 프로세서를 활용.
  - CUDA: NVIDIA GPU에서 고성능 연산 가능.
  - 사용자가 직접 백엔드를 정의할 수도 있어 사용자 코드와의 긴밀한 통합이 가능합니다.
- 공유 및 분산 메모리: 단일 노드(공유 메모리)와 다중 노드(MPI를 통한 분산 메모리) 환경 모두에서 동작합니다.
- 확장성: 컴파일 타임 정책 기반 설계를 통해 사용자가 AMG 구성 요소를 커스터마이징하거나 새 기능을 추가할 수 있습니다. 런타임 인터페이스도 제공되어 유연성을 더합니다.
- 성능: 다른 AMG 구현과 비교해도 경쟁력 있는 성능을 보여줍니다.

### 사용 예시

간단한 예로, Poisson 방정식을 풀기 위해 BiCGStab 솔버와 smoothed aggregation AMG를 조합한 코드는 다음과 같습니다:

```cpp
#include <amgcl/make_solver.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/adapter/crs_tuple.hpp>

typedef amgcl::backend::builtin<double> Backend;
typedef amgcl::make_solver<
    amgcl::amg<Backend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::spai0>,
    amgcl::solver::bicgstab<Backend>
> Solver;
```

AMGCL은 C++로 작성된 헤더 전용 라이브러리이므로, C#에서 사용하기 위해서는 네이티브 C++코드를 호출하여야 합니다. 관련하여 간단한 구현 코드는 [Amgcl.Net](https://github.com/archivesdj/Amgcl.Net)를 참고하시기 바랍니다.

## C#으로 유사 기능 구현

이 프로젝트는 C#으로 유사 기능을 구현해 보고자 하여 시작하였습니다.

AMG(Algebraic Multigrid) 알고리즘을 C#으로 직접 구현하는 것은 상당히 복잡하지만, 단계별로 진행하면 가능합니다. 여기서는 단순화된 AMG 솔버를 C#으로 구현해 보겠습니다. 이 구현은 개념 설명과 실용성을 위해 다음과 같은 요소를 포함합니다:

- 희소 행렬 표현: CSR(Compressed Sparse Row) 형식 사용.
- AMG 구성 요소: 
  - Smoothed Aggregation 기반 coarsening(굵은 격자 생성).
  - Relaxation(SOR 또는 Jacobi).
  - 다중 격자 계층 구축 및 V-cycle.
  - `ILGPU`를 이용하여 병렬처리 알고리즘 추가
