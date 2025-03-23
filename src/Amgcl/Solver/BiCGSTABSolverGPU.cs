using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using Amgcl.Matrix;

namespace Amgcl.Solver;

public class BiCGSTABSolverGPU : ISolver
{
    private readonly Accelerator accelerator;
    private readonly SparseMatrixCSR matrix;

    // Constructor: Initialize with matrix and an existing Accelerator
    // Comment: Matches LUDirectSolverGPU's structure for consistency.
    // Assumption: Works for non-symmetric matrices, unlike CG which requires symmetry.
    public BiCGSTABSolverGPU(SparseMatrixCSR matrix, Accelerator? accelerator)
    {
        this.matrix = matrix ?? throw new ArgumentNullException(nameof(matrix));
        this.accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
    }

    // Relax: Perform BiCGSTAB iterations on GPU
    // Comment: Iterative solver using GPU for parallel matrix-vector multiplications and vector operations.
    // Observation: More complex than LUDirectSolverGPU due to BiCGSTAB's stabilization steps.
    public void Relax(double[] b, double[] x, int maxIterations, double tolerance)
    {
        if (b.Length != matrix.Rows || x.Length != matrix.Rows)
            throw new ArgumentException("Vector length does not match matrix dimensions.");

        int n = matrix.Rows;

        // Allocate GPU memory buffers
        // Comment: Manages multiple vectors required for BiCGSTAB algorithm.
        using var bBuffer = accelerator.Allocate1D<double>(b);
        using var xBuffer = accelerator.Allocate1D<double>(x);
        using var rBuffer = accelerator.Allocate1D<double>(n); // Residual r
        using var r0Buffer = accelerator.Allocate1D<double>(n); // Initial residual (shadow residual)
        using var pBuffer = accelerator.Allocate1D<double>(n); // Search direction p
        using var vBuffer = accelerator.Allocate1D<double>(n); // v = A * p
        using var sBuffer = accelerator.Allocate1D<double>(n); // s = r - alpha * v
        using var tBuffer = accelerator.Allocate1D<double>(n); // t = A * s
        using var valuesBuffer = accelerator.Allocate1D<double>(matrix.Values);
        using var colIndicesBuffer = accelerator.Allocate1D<int>(matrix.ColIndices);
        using var rowPointersBuffer = accelerator.Allocate1D<int>(matrix.RowPointers);

        // Load GPU kernels
        // Comment: Kernels for matrix-vector multiplication and vector updates, optimized for GPU parallelism.
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

        // Initialize: r = b - Ax, r0 = r, p = r
        matVecKernel(n, xBuffer.View, valuesBuffer.View, colIndicesBuffer.View, rowPointersBuffer.View, vBuffer.View);
        accelerator.Synchronize();
        updateKernel(n, bBuffer.View, vBuffer.View, rBuffer.View, -1.0); // r = b - Ax
        accelerator.Synchronize();
        r0Buffer.CopyFrom(rBuffer); // r0 = r (shadow residual)
        pBuffer.CopyFrom(rBuffer); // p = r initially

        // Initial residual norm
        double[] r = new double[n];
        rBuffer.CopyToCPU(r);
        double rho = DotProduct(r, r); // r^T r
        double rhoOld = rho;

        // BiCGSTAB iteration
        // Comment: Implements stabilized biconjugate gradient with GPU acceleration.
        for (int iter = 0; iter < maxIterations; iter++)
        {
            if (Math.Sqrt(rho) < tolerance) break;

            // Compute v = A * p
            matVecKernel(n, pBuffer.View, valuesBuffer.View, colIndicesBuffer.View, rowPointersBuffer.View, vBuffer.View);
            accelerator.Synchronize();

            // Compute alpha = rho / (r0^T v)
            double[] v = new double[n];
            vBuffer.CopyToCPU(v);
            double r0Tv = DotProduct(r0Buffer, v);
            if (r0Tv == 0) break; // Avoid division by zero
            double alpha = rho / r0Tv;

            // Compute s = r - alpha * v
            updateKernel(n, rBuffer.View, vBuffer.View, sBuffer.View, -alpha);
            accelerator.Synchronize();

            // Compute t = A * s
            matVecKernel(n, sBuffer.View, valuesBuffer.View, colIndicesBuffer.View, rowPointersBuffer.View, tBuffer.View);
            accelerator.Synchronize();

            // Compute omega = (t^T s) / (t^T t)
            double[] s = new double[n];
            double[] t = new double[n];
            sBuffer.CopyToCPU(s);
            tBuffer.CopyToCPU(t);
            double tTs = DotProduct(t, s);
            double tTt = DotProduct(t, t);
            if (tTt == 0) break; // Avoid division by zero
            double omegaLocal = tTs / tTt;

            // Update x = x + alpha * p + omega * s
            updateKernel(n, xBuffer.View, pBuffer.View, xBuffer.View, alpha);
            accelerator.Synchronize();
            updateKernel(n, xBuffer.View, sBuffer.View, xBuffer.View, omegaLocal);
            accelerator.Synchronize();

            // Update r = s - omega * t
            updateKernel(n, sBuffer.View, tBuffer.View, rBuffer.View, -omegaLocal);
            accelerator.Synchronize();

            // Compute new rho and beta
            rBuffer.CopyToCPU(r);
            double rhoNew = DotProduct(r0Buffer, r);
            if (rhoNew == 0) break; // Avoid breakdown
            double beta = (rhoNew / rhoOld) * (alpha / omegaLocal);

            // Update p = r + beta * (p - omega * v)
            updateKernel(n, pBuffer.View, vBuffer.View, pBuffer.View, -omegaLocal * beta);
            accelerator.Synchronize();
            updateKernel(n, rBuffer.View, pBuffer.View, pBuffer.View, beta);
            accelerator.Synchronize();

            rhoOld = rhoNew;
            rho = rhoNew;
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
    // Comment: General-purpose vector operation for BiCGSTAB updates, parallelized across elements.
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
    // Comment: CPU-based for now; GPU reduction could enhance performance.
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
    // Comment: Matches LUDirectSolverGPU's structure.
    public double[] Solve(double[] b, int maxIterations, double tolerance)
    {
        if (b.Length != matrix.Rows)
            throw new ArgumentException("Length of b does not match matrix rows.");

        double[] x = new double[matrix.Rows]; // Zero-initialized
        Relax(b, x, maxIterations, tolerance);
        return x;
    }
}