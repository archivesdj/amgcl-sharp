using Amgcl.Matrix;

namespace Amgcl.Solver;

public class ConjugateGradientSolver : ISolver
{
    private readonly SparseMatrixCSR matrix;

    public ConjugateGradientSolver(SparseMatrixCSR matrix)
    {
        this.matrix = matrix ?? throw new ArgumentNullException(nameof(matrix));
        if (matrix.Rows != matrix.Cols)
            throw new InvalidOperationException("Matrix must be square for CG");
    }

    public void Relax(double[] b, double[] x, int maxIterations, double tolerance)
    {
        if (b.Length != matrix.Rows || x.Length != matrix.Rows)
            throw new ArgumentException("Vector length mismatch");

        double[] r = new double[matrix.Rows];
        double[] p = new double[matrix.Rows];
        double[] Ap = new double[matrix.Rows];

        matrix.Multiply(x, Ap); // Ap = Ax
        for (int i = 0; i < matrix.Rows; i++)
        {
            r[i] = b[i] - Ap[i]; // r = b - Ax
            p[i] = r[i];         // initial p = r
        }

        double rsold = DotProduct(r, r);
        for (int iter = 0; iter < maxIterations; iter++)
        {
            matrix.Multiply(p, Ap); // Ap = A*p
            double alpha = rsold / DotProduct(p, Ap);
            for (int i = 0; i < matrix.Rows; i++)
            {
                x[i] += alpha * p[i];
                r[i] -= alpha * Ap[i];
            }

            double rsnew = DotProduct(r, r);
            if (Math.Sqrt(rsnew) < tolerance) break;

            double beta = rsnew / rsold;
            for (int i = 0; i < matrix.Rows; i++)
            {
                p[i] = r[i] + beta * p[i];
            }
            rsold = rsnew;
        }
    }

    public double[] Solve(double[] b, int maxIterations, double tolerance)
    {
        if (b.Length != matrix.Rows)
            throw new ArgumentException("Vector length must match matrix rows");

        double[] x = new double[matrix.Rows];
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