using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using Amgcl.Matrix;

namespace Amgcl.Solver;

public class GaussSeidelSolverGPU : ISolver
{
    private readonly Accelerator accelerator;
    private readonly SparseMatrixCSR matrix;

    // Constructor: Initialize with matrix and an existing Accelerator
    // Comment: Mirrors LUDirectSolverGPU's constructor, ensuring consistency in dependency injection.
    // Suggestion: Add a comment if the nullable Accelerator is intended for testing or fallback scenarios.
    public GaussSeidelSolverGPU(SparseMatrixCSR matrix, Accelerator? accelerator)
    {
        this.matrix = matrix ?? throw new ArgumentNullException(nameof(matrix));
        this.accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
    }

    // Relax: Perform Gauss-Seidel iterations on GPU
    // Comment: Implements an iterative solver using GPU acceleration, updating all rows sequentially.
    // Observation: Unlike LUDirectSolverGPU, this method uses maxIterations and tolerance for convergence.
    public void Relax(double[] b, double[] x, int maxIterations, double tolerance)
    {
        if (b.Length != matrix.Rows || x.Length != matrix.Rows)
            throw new ArgumentException("Vector length does not match matrix dimensions.");

        int n = matrix.Rows;

        // Allocate GPU memory buffers
        // Comment: Proper resource management with 'using' ensures GPU memory is freed after use.
        using var bBuffer = accelerator.Allocate1D<double>(b);
        using var xBuffer = accelerator.Allocate1D<double>(x);
        using var valuesBuffer = accelerator.Allocate1D<double>(matrix.Values);
        using var colIndicesBuffer = accelerator.Allocate1D<int>(matrix.ColIndices);
        using var rowPointersBuffer = accelerator.Allocate1D<int>(matrix.RowPointers);

        // Load Gauss-Seidel kernel
        // Comment: Uses ILGPU's current API with ArrayView1D and Stride1D.Dense, consistent with LUDirectSolverGPU.
        var gsKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<double, Stride1D.Dense>,
            ArrayView1D<double, Stride1D.Dense>,
            ArrayView1D<double, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(GaussSeidelKernel);

        // Perform iterations
        // Comment: Iterates until convergence or maxIterations, with each sweep updating all rows sequentially.
        for (int iter = 0; iter < maxIterations; iter++)
        {
            double[] prevX = new double[n];
            xBuffer.CopyToCPU(prevX);

            // Single sweep across all rows
            // Comment: Unlike Red-Black, processes all rows in one kernel call, respecting dependencies via synchronization.
            gsKernel(n, bBuffer.View, xBuffer.View, valuesBuffer.View, colIndicesBuffer.View, rowPointersBuffer.View);
            accelerator.Synchronize(); // Ensure all updates are complete before convergence check

            // Convergence check on CPU
            // Suggestion: For better GPU utilization, consider a GPU-based reduction for convergence checking.
            xBuffer.CopyToCPU(x);
            double maxDiff = 0;
            for (int i = 0; i < n; i++)
            {
                double diff = Math.Abs(x[i] - prevX[i]);
                if (diff > maxDiff) maxDiff = diff;
            }
            if (maxDiff < tolerance) break;
        }

        // Final result back to CPU
        xBuffer.CopyToCPU(x);
    }

    /// <summary>
    /// GPU kernel for Gauss-Seidel iteration: updates x for a given row.
    /// </summary>
    // Comment: Implements the Gauss-Seidel update x[i] = (b[i] - sum(A[i,j] * x[j], j != i)) / A[i,i].
    // Observation: Single-threaded per row to avoid race conditions; parallelism is across iterations, not rows.
    private static void GaussSeidelKernel(
        Index1D index,
        ArrayView1D<double, Stride1D.Dense> b,
        ArrayView1D<double, Stride1D.Dense> x,
        ArrayView1D<double, Stride1D.Dense> values,
        ArrayView1D<int, Stride1D.Dense> colIndices,
        ArrayView1D<int, Stride1D.Dense> rowPointers)
    {
        int i = index;
        if (i >= rowPointers.Length - 1) return;

        int start = rowPointers[i];
        int end = rowPointers[i + 1];
        double sum = 0.0;
        double diag = 0.0;

        // Compute off-diagonal sum and find diagonal
        for (int j = start; j < end; j++)
        {
            int col = colIndices[j];
            if (col != i)
                sum += values[j] * x[col]; // Uses latest x values from previous updates
            else
                diag = values[j];
        }

        // Update x[i] if diagonal is non-zero
        if (diag != 0.0)
            x[i] = (b[i] - sum) / diag;
    }

    // Solve: Initialize x and call Relax to compute solution
    // Comment: Consistent with LUDirectSolverGPU, delegates to Relax for iterative computation.
    public double[] Solve(double[] b, int maxIterations, double tolerance)
    {
        if (b.Length != matrix.Rows)
            throw new ArgumentException("Length of b does not match matrix rows.");

        double[] x = new double[matrix.Rows]; // Zero-initialized
        Relax(b, x, maxIterations, tolerance);
        return x;
    }
}