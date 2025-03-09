using ILGPU;
using ILGPU.Runtime;

namespace AmgSharp.Logics;

public class JacobiRelaxation : IDisposable
{
    private readonly SparseMatrix A;
    private readonly Accelerator accelerator;
    private readonly MemoryBuffer1D<double, Stride1D.Dense> d_x;
    private readonly MemoryBuffer1D<double, Stride1D.Dense> d_b;
    private readonly MemoryBuffer1D<double, Stride1D.Dense> d_result;
    private readonly MemoryBuffer1D<double, Stride1D.Dense> d_diag;
    private readonly MemoryBuffer1D<double, Stride1D.Dense> d_values;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> d_colIndices;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> d_rowPointers;

    public JacobiRelaxation(SparseMatrix matrix, Accelerator accel)
    {
        A = matrix;
        accelerator = accel;

        // GPU 메모리에 상수 데이터 할당 (한 번만 초기화)
        d_values = accelerator.Allocate1D<double>(A.Values);
        d_colIndices = accelerator.Allocate1D<int>(A.ColIndices);
        d_rowPointers = accelerator.Allocate1D<int>(A.RowPointers);

        double[] diag = new double[A.Rows];
        for (int i = 0; i < A.Rows; i++)
        {
            for (int j = A.RowPointers[i]; j < A.RowPointers[i + 1]; j++)
            {
                if (A.ColIndices[j] == i)
                {
                    diag[i] = A.Values[j];
                    break;
                }
            }
        }
        d_diag = accelerator.Allocate1D<double>(diag);

        // 동적 버퍼 초기화 (크기만 지정, 데이터는 Relax에서 설정)
        d_x = accelerator.Allocate1D<double>(A.Rows);
        d_b = accelerator.Allocate1D<double>(A.Rows);
        d_result = accelerator.Allocate1D<double>(A.Rows);
    }

    public void Relax(double[] x, double[] b, int iterations)
    {
        // GPU 메모리에 입력 데이터 업로드
        d_x.CopyFromCPU(x);
        d_b.CopyFromCPU(b);

        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<double, Stride1D.Dense>,
            ArrayView1D<double, Stride1D.Dense>,
            ArrayView1D<double, Stride1D.Dense>,
            ArrayView1D<double, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<double, Stride1D.Dense>>(JacobiKernel);

        for (int iter = 0; iter < iterations; iter++)
        {
            kernel(A.Rows, d_x.View, d_b.View, d_result.View, d_values.View, d_colIndices.View, d_rowPointers.View, d_diag.View);
            accelerator.Synchronize();
            d_x.CopyFrom(d_result); // GPU 버퍼 간 복사
        }

        // 최종 결과 CPU로 복사
        d_x.CopyToCPU(x);
    }

    private static void JacobiKernel(
        Index1D i,
        ArrayView1D<double, Stride1D.Dense> x,
        ArrayView1D<double, Stride1D.Dense> b,
        ArrayView1D<double, Stride1D.Dense> result,
        ArrayView1D<double, Stride1D.Dense> values,
        ArrayView1D<int, Stride1D.Dense> colIndices,
        ArrayView1D<int, Stride1D.Dense> rowPointers,
        ArrayView1D<double, Stride1D.Dense> diag)
    {
        double ax = 0.0;
        for (int j = rowPointers[i]; j < rowPointers[i + 1]; j++)
        {
            ax += values[j] * x[colIndices[j]];
        }
        result[i] = x[i] + (b[i] - ax) / diag[i];
    }

    public void Dispose()
    {
        d_x?.Dispose();
        d_b?.Dispose();
        d_result?.Dispose();
        d_diag?.Dispose();
        d_values?.Dispose();
        d_colIndices?.Dispose();
        d_rowPointers?.Dispose();
    }
}