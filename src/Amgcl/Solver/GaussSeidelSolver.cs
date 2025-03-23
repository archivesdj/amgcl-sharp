using Amgcl.Matrix;

namespace Amgcl.Solver;

public class GaussSeidelSolver : ISolver
{
    private readonly SparseMatrixCSR _A;
    private readonly double[] _diagonal; // Diagonal elements of the matrix

    // Constructor: Initialize with sparse matrix and extract diagonal elements
    public GaussSeidelSolver(SparseMatrixCSR A)
    {
        this._A = A ?? throw new ArgumentNullException(nameof(A));
        if (A.Rows != A.Cols)
            throw new InvalidOperationException("Matrix must be square for Gauss-Seidel method.");
        this._diagonal = A.GetDiagonal();
        if (_diagonal.Any(d => d == 0))
            throw new InvalidOperationException("Matrix must have non-zero diagonal elements");
    }

    // Relax: Perform Gauss-Seidel iterations to update x
    public void Relax(double[] b, double[] x, int maxIterations, double tolerance)
    {
        if (b.Length != _A.Rows || x.Length != _A.Rows)
            throw new ArgumentException("Vector length does not match matrix dimensions.");

        for (int iter = 0; iter < maxIterations; iter++)
        {
            double maxDiff = 0; // Track maximum difference for convergence check

            for (int i = 0; i < _A.Rows; i++)
            {
                double sum = 0;
                double oldX = x[i];

                // Compute sum of off-diagonal elements times current x values
                for (int j = _A.RowPointers[i]; j < _A.RowPointers[i + 1]; j++)
                {
                    int col = _A.ColIndices[j];
                    if (col != i) // Exclude diagonal
                        sum += _A.Values[j] * x[col];
                }

                // Update x[i] using Gauss-Seidel formula
                x[i] = (b[i] - sum) / _diagonal[i];

                // Calculate difference for convergence
                double diff = Math.Abs(x[i] - oldX);
                if (diff > maxDiff)
                    maxDiff = diff;
            }

            // Check convergence
            if (maxDiff < tolerance)
                break;
        }
    }

    // Solve: Initialize x and call Relax to compute solution
    public double[] Solve(double[] b, int maxIterations, double tolerance)
    {
        double[] x = new double[b.Length];
        Relax(b, x, maxIterations, tolerance);
        return x;
    }
}