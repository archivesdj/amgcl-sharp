using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using Amgcl.Matrix;

namespace Amgcl.Solver;

public class ConjugateGradientSolverGPU : ISolver
{
    private readonly Accelerator accelerator;
    private readonly SparseMatrixCSR matrix;

    // Constructor: Initialize with matrix and an existing Accelerator
    // Comment: Matches LUDirectSolverGPU's structure for consistency.
    // Assumption: Matrix A is symmetric positive definite, required for CG convergence.
    public ConjugateGradientSolverGPU(SparseMatrixCSR matrix, Accelerator? accelerator)
    {
        this.matrix = matrix ?? throw new ArgumentNullException(nameof(matrix));
        this.accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
    }

    // Relax: Perform conjugate gradient iterations on GPU
    // Comment: Iterative solver using GPU for parallel matrix-vector multiplications and vector operations.
    // Observation: Unlike LUDirectSolverGPU, CG minimizes the quadratic form iteratively.
    public void Relax(double[] b, double[] x, int maxIterations, double tolerance)
    {
        if (b.Length != matrix.Rows || x.Length != matrix.Rows)
            throw new ArgumentException("Vector length does not match matrix dimensions.");

        int n = matrix.Rows;

        // Allocate GPU memory buffers
        // Comment: Manages resources for CG vectors (residual, direction, Ap) alongside standard buffers.
        using var bBuffer = accelerator.Allocate1D<double>(b);
        using var xBuffer = accelerator.Allocate1D<double>(x);
        using var rBuffer = accelerator.Allocate1D<double>(n); // Residual r = b - Ax
        using var pBuffer = accelerator.Allocate1D<double>(n); // Search direction p
        using var apBuffer = accelerator.Allocate1D<double>(n); // A * p
        using var valuesBuffer = accelerator.Allocate1D<double>(matrix.Values);
        using var colIndicesBuffer = accelerator.Allocate1D<int>(matrix.ColIndices);
        using var rowPointersBuffer = accelerator.Allocate1D<int>(matrix.RowPointers);

        // Load GPU kernels
        // Comment: Kernels for matrix-vector multiplication and vector operations, optimized for parallelism.
        var matVecKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<double, Stride1D.Dense>,
            ArrayView1D<double, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<double, Stride1D.Dense>>(MatrixVectorMultiplyKernel);

        var updateKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<double, Stride1D.Dense>,
            ArrayView1D<double, Stride1D.Dense>,
            ArrayView1D<double, Stride1D.Dense>,
            double>(UpdateVectorKernel);

        // Initialize: r = b - Ax, p = r
        matVecKernel(n, xBuffer.View, valuesBuffer.View, colIndicesBuffer.View, rowPointersBuffer.View, apBuffer.View);
        accelerator.Synchronize();
        updateKernel(n, bBuffer.View, apBuffer.View, rBuffer.View, -1.0); // r = b - Ax
        accelerator.Synchronize();
        pBuffer.CopyFrom(rBuffer); // p = r initially

        // Initial residual norm
        double[] r = new double[n];
        rBuffer.CopyToCPU(r);
        double rTr = DotProduct(r, r); // r^T r
        double rTrOld = rTr;

        // CG iteration
        // Comment: Iterates until convergence, leveraging GPU for key operations.
        for (int iter = 0; iter < maxIterations; iter++)
        {
            if (Math.Sqrt(rTr) < tolerance) break;

            // Compute Ap
            matVecKernel(n, pBuffer.View, valuesBuffer.View, colIndicesBuffer.View, rowPointersBuffer.View, apBuffer.View);
            accelerator.Synchronize();

            // Compute alpha = r^T r / (p^T Ap)
            double[] ap = new double[n];
            apBuffer.CopyToCPU(ap);
            double pTAp = DotProduct(pBuffer, ap);
            double alpha = rTr / pTAp;

            // Update x = x + alpha * p
            updateKernel(n, xBuffer.View, pBuffer.View, xBuffer.View, alpha);
            accelerator.Synchronize();

            // Update r = r - alpha * Ap
            updateKernel(n, rBuffer.View, apBuffer.View, rBuffer.View, -alpha);
            accelerator.Synchronize();

            // Compute new r^T r and beta
            rBuffer.CopyToCPU(r);
            rTr = DotProduct(r, r);
            double beta = rTr / rTrOld;

            // Update p = r + beta * p
            updateKernel(n, rBuffer.View, pBuffer.View, pBuffer.View, beta);
            accelerator.Synchronize();

            rTrOld = rTr;
        }

        // Copy final result back to CPU
        xBuffer.CopyToCPU(x);
    }

    /// <summary>
    /// GPU kernel for sparse matrix-vector multiplication: computes Ax for a given row.
    /// </summary>
    // Comment: Parallelizes Ax computation across rows, consistent with LUDirectSolverGPU style.
    private static void MatrixVectorMultiplyKernel(
        Index1D index,
        ArrayView1D<double, Stride1D.Dense> x,
        ArrayView1D<double, Stride1D.Dense> values,
        ArrayView1D<int, Stride1D.Dense> colIndices,
        ArrayView1D<int, Stride1D.Dense> rowPointers,
        ArrayView1D<double, Stride1D.Dense> ax)
    {
        int i = index;
        if (i >= rowPointers.Length - 1) return;

        int start = rowPointers[i];
        int end = rowPointers[i + 1];
        double sum = 0.0;

        for (int j = start; j < end; j++)
        {
            int col = colIndices[j];
            sum += values[j] * x[col];
        }

        ax[i] = sum;
    }

    /// <summary>
    /// GPU kernel for vector update: computes result = a + scalar * b.
    /// </summary>
    // Comment: General-purpose vector operation for CG updates, parallelized across elements.
    private static void UpdateVectorKernel(
        Index1D index,
        ArrayView1D<double, Stride1D.Dense> a,
        ArrayView1D<double, Stride1D.Dense> b,
        ArrayView1D<double, Stride1D.Dense> result,
        double scalar)
    {
        int i = index;
        if (i >= a.Length) return;

        result[i] = a[i] + scalar * b[i];
    }

    // Helper method: Compute dot product on CPU (for simplicity)
    // Comment: Temporary CPU-based implementation; consider GPU reduction for performance.
    private static double DotProduct(double[] a, double[] b)
    {
        double sum = 0.0;
        for (int i = 0; i < a.Length; i++)
            sum += a[i] * b[i];
        return sum;
    }

    private static double DotProduct(ArrayView1D<double, Stride1D.Dense> a, double[] b)
    {
        double[] aArray = new double[a.Length];
        a.CopyToCPU(aArray);
        return DotProduct(aArray, b);
    }

    // Solve: Initialize x and call Relax to compute solution
    // Comment: Consistent with LUDirectSolverGPU's structure.
    public double[] Solve(double[] b, int maxIterations, double tolerance)
    {
        if (b.Length != matrix.Rows)
            throw new ArgumentException("Length of b does not match matrix rows.");

        double[] x = new double[matrix.Rows]; // Zero-initialized
        Relax(b, x, maxIterations, tolerance);
        return x;
    }
}