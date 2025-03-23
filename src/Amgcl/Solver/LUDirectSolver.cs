using Amgcl.Matrix;

namespace Amgcl.Solver;

public class LUDirectSolver : ISolver
{
    private readonly SparseMatrixCSR _A;
    private double[,] _L; // Lower triangular matrix
    private double[,] _U; // Upper triangular matrix

    // Constructor: Initialize with sparse matrix and perform LU decomposition
    public LUDirectSolver(SparseMatrixCSR A)
    {
        this._A = A ?? throw new ArgumentNullException(nameof(A));
        if (A.Rows != A.Cols)
            throw new InvalidOperationException("Matrix must be square for LU decomposition.");

        (_L, _U) = DecomposeLU();
    }

    // Perform LU decomposition
    private (double[,] L, double[,] U) DecomposeLU()
    {
        double[,] A = _A.ToDense();
        int n = A.GetLength(0);
        var L = new double[n, n];
        var U = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            // Compute upper triangular part of U
            for (int k = i; k < n; k++)
            {
                double sum = 0;
                for (int j = 0; j < i; j++)
                    sum += L[i, j] * U[j, k];
                U[i, k] = A[i, k] - sum;
            }

            // Compute lower triangular part of L
            for (int k = i; k < n; k++)
            {
                if (i == k)
                    L[i, i] = 1; // Diagonal of L is 1
                else
                {
                    double sum = 0;
                    for (int j = 0; j < i; j++)
                        sum += L[k, j] * U[j, i];
                    L[k, i] = (A[k, i] - sum) / U[i, i];
                }
            }
        }

        return (L, U);
    }

    // Forward substitution: Solve Ly = b
    private double[] ForwardSubstitution(double[] b)
    {
        int n = b.Length;
        double[] y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = b[i];
            for (int j = 0; j < i; j++)
                y[i] -= _L[i, j] * y[j];
            // L[i, i] = 1, so no division needed
        }
        return y;
    }

    // Backward substitution: Solve Ux = y
    private double[] BackwardSubstitution(double[] y)
    {
        int n = y.Length;
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--)
        {
            x[i] = y[i];
            for (int j = i + 1; j < n; j++)
                x[i] -= _U[i, j] * x[j];
            x[i] /= _U[i, i];
        }
        return x;
    }

    // Relax: Compute solution directly and update x
    public void Relax(double[] b, double[] x, int maxIterations, double tolerance)
    {
        if (b.Length != _A.Rows || x.Length != _A.Rows)
            throw new ArgumentException("Vector length does not match matrix dimensions.");

        double[] y = ForwardSubstitution(b);
        double[] result = BackwardSubstitution(y);
        Array.Copy(result, x, x.Length);
    }

    // Solve: Initialize x and call Relax to compute solution
    public double[] Solve(double[] b, int maxIterations, double tolerance)
    {
        double[] x = new double[b.Length];
        Relax(b, x, maxIterations, tolerance);
        return x;
    }
}