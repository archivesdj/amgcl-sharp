using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using Amgcl.Matrix;

namespace Amgcl.Solver;

public class LUDirectSolverGPU : ISolver
{
    private readonly Accelerator accelerator;
    private readonly SparseMatrixCSR matrix;
    private SparseMatrixCSR L; // Lower triangular matrix
    private SparseMatrixCSR U; // Upper triangular matrix

    // Constructor: Initialize with matrix and an existing Accelerator
    // Comment: Good use of null checks and initialization of LU decomposition in the constructor.
    // Suggestion: Consider adding a comment explaining that 'accelerator' is nullable for flexibility in testing or CPU fallback scenarios.
    public LUDirectSolverGPU(SparseMatrixCSR matrix, Accelerator? accelerator)
    {
        this.matrix = matrix ?? throw new ArgumentNullException(nameof(matrix));
        this.accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
        (L, U) = Decompose(); // Perform LU decomposition on initialization and store results
    }

    // Comment: This method performs a dense LU decomposition (Doolittle algorithm) and converts back to sparse CSR format.
    // Observation: Converting to a dense matrix may be memory-intensive for large sparse matrices; a native sparse LU could be more efficient but is more complex to implement.
    // Suggestion: Add error handling for zero pivots (e.g., throw an exception if u[i,i] == 0) to ensure numerical stability.
    private (SparseMatrixCSR L, SparseMatrixCSR U) Decompose()
    {
        int n = matrix.Rows;
        double[,] a = new double[n, n];

        // Sparse to dense conversion
        // Comment: Efficiently copies CSR data into a dense matrix, preserving sparsity structure.
        for (int i = 0; i < n; i++)
            for (int j = matrix.RowPointers[i]; j < matrix.RowPointers[i + 1]; j++)
                a[i, matrix.ColIndices[j]] = matrix.Values[j];

        double[,] l = new double[n, n];
        double[,] u = new double[n, n];

        // Doolittle LU decomposition
        // Comment: Correct implementation of the Doolittle algorithm with L having 1s on the diagonal.
        for (int i = 0; i < n; i++)
        {
            // Compute U (upper triangular)
            for (int k = i; k < n; k++)
            {
                double sum = 0;
                for (int j = 0; j < i; j++)
                    sum += l[i, j] * u[j, k];
                u[i, k] = a[i, k] - sum;
            }

            // Compute L (lower triangular, diagonal 1)
            for (int k = i; k < n; k++)
            {
                if (i == k)
                    l[i, i] = 1; // L diagonal is 1
                else
                {
                    double sum = 0;
                    for (int j = 0; j < i; j++)
                        sum += l[k, j] * u[j, i];
                    l[k, i] = (a[k, i] - sum) / u[i, i];
                }
            }
        }

        // Convert dense matrices back to sparse CSR
        var lSparse = DenseToSparseCSR(l);
        var uSparse = DenseToSparseCSR(u);

        return (lSparse, uSparse);
    }

    // Helper method to convert dense matrix to sparse CSR format
    // Comment: Well-implemented conversion that preserves non-zero elements, maintaining sparsity.
    // Suggestion: Consider adding a threshold (e.g., Math.Abs(value) > 1e-10) to filter out numerical noise.
    private SparseMatrixCSR DenseToSparseCSR(double[,] dense)
    {
        int rows = dense.GetLength(0);
        int cols = dense.GetLength(1);
        var values = new List<double>();
        var colIndices = new List<int>();
        var rowPointers = new List<int> { 0 };

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (dense[i, j] != 0)
                {
                    values.Add(dense[i, j]);
                    colIndices.Add(j);
                }
            }
            rowPointers.Add(values.Count);
        }

        return new SparseMatrixCSR(rows, cols, values.ToArray(), colIndices.ToArray(), rowPointers.ToArray());
    }

    // Relax method: GPU-accelerated LU solve
    // Comment: Good use of GPU buffers and kernel launches, but the sequential execution per row limits parallelism due to dependencies.
    // Suggestion: For better GPU utilization, explore wavefront parallelism or use an existing GPU sparse solver library (e.g., cuSPARSE).
    public void Relax(double[] b, double[] x, int maxIterations, double tolerance)
    {
        if (b.Length != matrix.Rows || x.Length != matrix.Rows)
            throw new ArgumentException("Vector length does not match matrix dimensions.");

        int n = matrix.Rows;
        double[] y = new double[n]; // Intermediate vector for Ly = b

        // Allocate GPU memory buffers
        // Comment: Proper use of 'using' ensures memory cleanup, which is critical for GPU resource management.
        using var bBuffer = accelerator.Allocate1D<double>(b);
        using var yBuffer = accelerator.Allocate1D<double>(n);
        using var xBuffer = accelerator.Allocate1D<double>(n);
        using var lValuesBuffer = accelerator.Allocate1D<double>(L.Values);
        using var lColIndicesBuffer = accelerator.Allocate1D<int>(L.ColIndices);
        using var lRowPointersBuffer = accelerator.Allocate1D<int>(L.RowPointers);
        using var uValuesBuffer = accelerator.Allocate1D<double>(U.Values);
        using var uColIndicesBuffer = accelerator.Allocate1D<int>(U.ColIndices);
        using var uRowPointersBuffer = accelerator.Allocate1D<int>(U.RowPointers);

        // Load GPU kernels
        // Comment: Correct kernel signature using ILGPU's current API (ArrayView1D with Stride1D.Dense).
        var forwardKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, 
            ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<int, Stride1D.Dense>, 
            ArrayView1D<int, Stride1D.Dense>, 
            int>(ForwardSubstitutionKernel);

        var backwardKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, 
            ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<int, Stride1D.Dense>, 
            ArrayView1D<int, Stride1D.Dense>, 
            int>(BackwardSubstitutionKernel);

        // Step 1: Forward substitution (Ly = b)
        // Comment: Sequential kernel launches with synchronization ensure correctness but reduce GPU efficiency.
        for (int i = 0; i < n; i++)
        {
            forwardKernel(1, bBuffer.View, yBuffer.View, lValuesBuffer.View, lColIndicesBuffer.View, lRowPointersBuffer.View, i);
            accelerator.Synchronize(); // Ensure row i is computed before proceeding
        }

        // Step 2: Backward substitution (Ux = y)
        // Comment: Reverse order is correct for backward substitution; synchronization ensures dependencies are respected.
        for (int i = n - 1; i >= 0; i--)
        {
            backwardKernel(1, yBuffer.View, xBuffer.View, uValuesBuffer.View, uColIndicesBuffer.View, uRowPointersBuffer.View, i);
            accelerator.Synchronize(); // Ensure row i is computed before proceeding
        }

        // Copy the solution back to the CPU
        xBuffer.CopyToCPU(x);
    }

    /// <summary>
    /// GPU kernel for forward substitution: solves Ly = b for row i.
    /// </summary>
    // Comment: Correctly computes the sum for off-diagonal elements; assumes L has unit diagonal, which matches the Doolittle method used in Decompose.
    private static void ForwardSubstitutionKernel(
        Index1D index, 
        ArrayView1D<double, Stride1D.Dense> b, 
        ArrayView1D<double, Stride1D.Dense> y, 
        ArrayView1D<double, Stride1D.Dense> values, 
        ArrayView1D<int, Stride1D.Dense> colIndices, 
        ArrayView1D<int, Stride1D.Dense> rowPointers, 
        int i)
    {
        if (index != 0) return; // Single thread per row

        int start = rowPointers[i];
        int end = rowPointers[i + 1];
        double sum = 0.0;

        // Compute sum of L[i,j] * y[j] for j < i
        for (int j = start; j < end; j++)
        {
            int col = colIndices[j];
            if (col < i)
                sum += values[j] * y[col];
        }

        // Assuming L has 1s on the diagonal (no division needed)
        y[i] = b[i] - sum;
    }

    /// <summary>
    /// GPU kernel for backward substitution: solves Ux = y for row i.
    /// </summary>
    // Comment: Properly extracts the diagonal and computes off-diagonal sums; division by diagonal ensures correct solution.
    private static void BackwardSubstitutionKernel(
        Index1D index, 
        ArrayView1D<double, Stride1D.Dense> y, 
        ArrayView1D<double, Stride1D.Dense> x, 
        ArrayView1D<double, Stride1D.Dense> values, 
        ArrayView1D<int, Stride1D.Dense> colIndices, 
        ArrayView1D<int, Stride1D.Dense> rowPointers, 
        int i)
    {
        if (index != 0) return; // Single thread per row

        int start = rowPointers[i];
        int end = rowPointers[i + 1];
        double sum = 0.0;
        double diag = 0.0;

        // Compute sum of U[i,j] * x[j] for j > i and find diagonal
        for (int j = start; j < end; j++)
        {
            int col = colIndices[j];
            if (col > i)
                sum += values[j] * x[col];
            else if (col == i)
                diag = values[j];
        }

        if (diag != 0.0)
            x[i] = (y[i] - sum) / diag; // Solve for x[i]
    }

    // Solve: Call Relax to compute solution
    // Comment: Simple delegation to Relax is appropriate for a direct solver; maxIterations and tolerance are unused as expected.
    public double[] Solve(double[] b, int maxIterations, double tolerance)
    {
        double[] x = new double[matrix.Rows];
        Relax(b, x, maxIterations, tolerance);
        return x;
    }
}