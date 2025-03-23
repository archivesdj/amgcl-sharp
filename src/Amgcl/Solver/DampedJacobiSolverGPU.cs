using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using Amgcl.Matrix;

namespace Amgcl.Solver;

public class DampedJacobiSolverGPU : ISolver
{
    private readonly Accelerator accelerator;
    private readonly SparseMatrixCSR matrix;
    private readonly double omega; // Damping factor

    // Constructor: Initialize with matrix, accelerator, and damping factor
    // Comment: Follows LUDirectSolverGPU's structure, adding omega for damped Jacobi iteration.
    public DampedJacobiSolverGPU(SparseMatrixCSR matrix, Accelerator? accelerator, double omega = 1.0)
    {
        this.matrix = matrix ?? throw new ArgumentNullException(nameof(matrix));
        this.accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
        this.omega = omega; // Default omega = 1.0 (standard Jacobi)
    }

    // Relax: Perform damped Jacobi iterations on GPU
    // Comment: Iterative solver using GPU for parallel matrix-vector multiplication and Jacobi updates.
    public void Relax(double[] b, double[] x, int maxIterations, double tolerance)
    {
        if (b.Length != matrix.Rows || x.Length != matrix.Rows)
            throw new ArgumentException("Vector length does not match matrix dimensions.");

        int n = matrix.Rows;

        // Allocate GPU memory buffers
        using var bBuffer = accelerator.Allocate1D<double>(b);
        using var xBuffer = accelerator.Allocate1D<double>(x);
        using var axBuffer = accelerator.Allocate1D<double>(n); // Buffer for Ax
        using var valuesBuffer = accelerator.Allocate1D<double>(matrix.Values);
        using var colIndicesBuffer = accelerator.Allocate1D<int>(matrix.ColIndices);
        using var rowPointersBuffer = accelerator.Allocate1D<int>(matrix.RowPointers);

        // Load GPU kernels
        // Comment: Corrected signature matches the kernel definitions below.
        var matVecKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<double, Stride1D.Dense>,
            ArrayView1D<double, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<double, Stride1D.Dense>>(MatrixVectorMultiplyKernel);

        var jacobiKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<double, Stride1D.Dense>,
            ArrayView1D<double, Stride1D.Dense>,
            ArrayView1D<double, Stride1D.Dense>,
            ArrayView1D<double, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            double>(DampedJacobiKernel);

        // Perform iterations
        for (int iter = 0; iter < maxIterations; iter++)
        {
            double[] prevX = new double[n];
            xBuffer.CopyToCPU(prevX);

            // Compute Ax
            matVecKernel(n, xBuffer.View, valuesBuffer.View, colIndicesBuffer.View, rowPointersBuffer.View, axBuffer.View);
            accelerator.Synchronize();

            // Apply damped Jacobi update
            jacobiKernel(n, bBuffer.View, axBuffer.View, xBuffer.View, valuesBuffer.View, colIndicesBuffer.View, rowPointersBuffer.View, omega);
            accelerator.Synchronize();

            // Convergence check on CPU
            xBuffer.CopyToCPU(x);
            double maxDiff = 0;
            for (int i = 0; i < n; i++)
            {
                double diff = Math.Abs(x[i] - prevX[i]);
                if (diff > maxDiff) maxDiff = diff;
            }
            if (maxDiff < tolerance) break;
        }

        // Copy final result back to CPU
        xBuffer.CopyToCPU(x);
    }

    /// <summary>
    /// GPU kernel for sparse matrix-vector multiplication: computes Ax for a given row.
    /// </summary>
    // Comment: Corrected to match the LoadAutoGroupedStreamKernel signature with 5 arguments (+ Index1D).
    // Observation: Computes Ax[i] in parallel across rows, consistent with CSR format.
    private static void MatrixVectorMultiplyKernel(
        Index1D index,
        ArrayView1D<double, Stride1D.Dense> x,         // Input vector x
        ArrayView1D<double, Stride1D.Dense> values,    // Matrix values
        ArrayView1D<int, Stride1D.Dense> colIndices,   // Column indices
        ArrayView1D<int, Stride1D.Dense> rowPointers,  // Row pointers
        ArrayView1D<double, Stride1D.Dense> ax)        // Output vector Ax
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
    /// GPU kernel for damped Jacobi update: applies x[i] = (1 - omega) * x[i] + omega * (b[i] - sum(A[i,j] * x[j], j != i)) / A[i,i].
    /// </summary>
    // Comment: Matches the LoadAutoGroupedStreamKernel signature with 6 arguments (+ Index1D).
    private static void DampedJacobiKernel(
        Index1D index,
        ArrayView1D<double, Stride1D.Dense> b,         // Right-hand side b
        ArrayView1D<double, Stride1D.Dense> ax,        // Computed Ax
        ArrayView1D<double, Stride1D.Dense> x,         // Solution vector x
        ArrayView1D<double, Stride1D.Dense> values,    // Matrix values
        ArrayView1D<int, Stride1D.Dense> colIndices,   // Column indices
        ArrayView1D<int, Stride1D.Dense> rowPointers,  // Row pointers
        double omega)                                  // Damping factor
    {
        int i = index;
        if (i >= rowPointers.Length - 1) return;

        int start = rowPointers[i];
        int end = rowPointers[i + 1];
        double sum = 0.0; // Off-diagonal sum
        double diag = 0.0;

        // Compute off-diagonal sum and find diagonal
        for (int j = start; j < end; j++)
        {
            int col = colIndices[j];
            if (col != i)
                sum += values[j] * x[col];
            else
                diag = values[j];
        }

        // Apply damped Jacobi update
        if (diag != 0.0)
        {
            double newX = (b[i] - sum) / diag; // Standard Jacobi update
            x[i] = (1.0 - omega) * x[i] + omega * newX; // Damped update
        }
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