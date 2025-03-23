using Amgcl.Matrix;

namespace Amgcl.Solver;

public class BiCGSTABSolver : ISolver
{
    private readonly SparseMatrixCSR _A;

    // Constructor: Initialize with a sparse matrix
    public BiCGSTABSolver(SparseMatrixCSR A)
    {
        this._A = A ?? throw new ArgumentNullException(nameof(A));
        if (A.Rows != A.Cols)
            throw new InvalidOperationException("Matrix must be square for BiCGSTAB");
    }

    // Relax method: Improve the initial solution x
    public void Relax(double[] b, double[] x, int maxIterations, double tolerance)
    {
        if (b.Length != _A.Rows || x.Length != _A.Rows)
            throw new ArgumentException("Vector length mismatch");

        // Initialize internal vectors
        double[] r = new double[_A.Rows];  // Residual
        double[] r0 = new double[_A.Rows]; // Initial residual
        double[] p = new double[_A.Rows];  // Search direction
        double[] v = new double[_A.Rows];  // A*p
        double[] s = new double[_A.Rows];  // Intermediate residual
        double[] t = new double[_A.Rows];  // A*s

        // Compute initial residual: r = b - Ax
        _A.Multiply(x, r);
        for (int i = 0; i < _A.Rows; i++)
        {
            r[i] = b[i] - r[i];
            r0[i] = r[i];  // r0 = r (fixed reference residual)
            p[i] = r[i];   // Initial p = r
        }

        // Initialize BiCGSTAB variables
        double rhoOld = 1;
        double alpha = 1;
        double omega = 1;

        // Iterative process
        for (int iter = 0; iter < maxIterations; iter++)
        {
            // Compute rho
            double rho = DotProduct(r0, r);
            if (Math.Abs(rho) < tolerance) break;

            // Compute beta and update p
            double beta = (rho / rhoOld) * (alpha / omega);
            for (int i = 0; i < _A.Rows; i++)
            {
                p[i] = r[i] + beta * (p[i] - omega * v[i]);
            }

            // Compute v = A*p
            _A.Multiply(p, v);
            double r0v = DotProduct(r0, v);
            alpha = rho / r0v;

            // Compute s = r - alpha*v
            for (int i = 0; i < _A.Rows; i++)
            {
                s[i] = r[i] - alpha * v[i];
            }

            // Compute t = A*s
            _A.Multiply(s, t);
            omega = DotProduct(t, s) / DotProduct(t, t);

            // Update x and r
            for (int i = 0; i < _A.Rows; i++)
            {
                x[i] += alpha * p[i] + omega * s[i];
                r[i] = s[i] - omega * t[i];
            }

            // Check residual norm
            double norm = Math.Sqrt(DotProduct(r, r));
            if (norm < tolerance) break;

            rhoOld = rho;
        }
    }

    // Solve: Initialize x and call Relax to compute solution
    public double[] Solve(double[] b, int maxIterations, double tolerance)
    {
        double[] x = new double[b.Length];
        Relax(b, x, maxIterations, tolerance);
        return x;
    }

    // Helper function to compute dot product
    private static double DotProduct(double[] a, double[] b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
            sum += a[i] * b[i];
        return sum;
    }
}