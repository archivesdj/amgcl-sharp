using Amgcl.Matrix;

namespace Amgcl.Solver;

public class ConjugateGradientSolver : ISolver
{
    private readonly SparseMatrixCSR _A;

    public ConjugateGradientSolver(SparseMatrixCSR A)
    {
        this._A = A ?? throw new ArgumentNullException(nameof(A));
        if (A.Rows != A.Cols)
            throw new InvalidOperationException("Matrix must be square for CG");
    }

    public void Relax(double[] b, double[] x, int maxIterations, double tolerance)
    {
        if (b.Length != _A.Rows || x.Length != _A.Rows)
            throw new ArgumentException("Vector length mismatch");

        double[] r = new double[_A.Rows];
        double[] p = new double[_A.Rows];
        double[] Ap = new double[_A.Rows];

        _A.Multiply(x, Ap); // Ap = Ax
        for (int i = 0; i < _A.Rows; i++)
        {
            r[i] = b[i] - Ap[i]; // r = b - Ax
            p[i] = r[i];         // initial p = r
        }

        double rsold = DotProduct(r, r);
        for (int iter = 0; iter < maxIterations; iter++)
        {
            _A.Multiply(p, Ap); // Ap = A*p
            double alpha = rsold / DotProduct(p, Ap);
            for (int i = 0; i < _A.Rows; i++)
            {
                x[i] += alpha * p[i];
                r[i] -= alpha * Ap[i];
            }

            double rsnew = DotProduct(r, r);
            if (Math.Sqrt(rsnew) < tolerance) break;

            double beta = rsnew / rsold;
            for (int i = 0; i < _A.Rows; i++)
            {
                p[i] = r[i] + beta * p[i];
            }
            rsold = rsnew;
        }
    }

    // Solve: Initialize x and call Relax to compute solution
    public double[] Solve(double[] b, int maxIterations, double tolerance)
    {
        double[] x = new double[b.Length];
        Relax(b, x, maxIterations, tolerance);
        return x;
    }

    private static double DotProduct(double[] a, double[] b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
            sum += a[i] * b[i];
        return sum;
    }
}