using Amgcl.Matrix;

namespace Amgcl.Solver;

public class GaussSeidelSolver : ISolver
{
    private readonly SparseMatrixCSR matrix;
    private readonly double[] diagonal; // Diagonal elements of the matrix

    // Constructor: Initialize with sparse matrix and extract diagonal elements
    public GaussSeidelSolver(SparseMatrixCSR matrix)
    {
        this.matrix = matrix ?? throw new ArgumentNullException(nameof(matrix));
        if (matrix.Rows != matrix.Cols)
            throw new InvalidOperationException("Matrix must be square for Gauss-Seidel method.");
        this.diagonal = matrix.GetDiagonal();
        if (diagonal.Any(d => d == 0))
            throw new InvalidOperationException("Matrix must have non-zero diagonal elements");
    }

    // Relax: Perform Gauss-Seidel iterations to update x
    public void Relax(double[] b, double[] x, int maxIterations, double tolerance)
    {
        if (b.Length != matrix.Rows || x.Length != matrix.Rows)
            throw new ArgumentException("Vector length does not match matrix dimensions.");

        for (int iter = 0; iter < maxIterations; iter++)
        {
            double maxDiff = 0; // Track maximum difference for convergence check

            for (int i = 0; i < matrix.Rows; i++)
            {
                double sum = 0;
                double oldX = x[i];

                // Compute sum of off-diagonal elements times current x values
                for (int j = matrix.RowPointers[i]; j < matrix.RowPointers[i + 1]; j++)
                {
                    int col = matrix.ColIndices[j];
                    if (col != i) // Exclude diagonal
                        sum += matrix.Values[j] * x[col];
                }

                // Update x[i] using Gauss-Seidel formula
                x[i] = (b[i] - sum) / diagonal[i];

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
        if (b.Length != matrix.Rows)
            throw new ArgumentException("Length of b does not match matrix rows.");

        double[] x = new double[matrix.Rows]; // Initialize solution with zeros
        Relax(b, x, maxIterations, tolerance);
        return x;
    }
}