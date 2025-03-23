using Amgcl.Matrix;

namespace Amgcl.Solver;

public class DampedJacobiSolver : ISolver
{
    private readonly SparseMatrixCSR _A;
   
    private readonly double[] _diagonal;

    private double _omega = 1.0;
    public double Omega 
    {
        get { return _omega; }
        set {
            if (value <= 0 || value > 1)
                throw new ArgumentException("Omega must be between 0 and 1");
            _omega = value;
        }
    }

    public DampedJacobiSolver(SparseMatrixCSR A)
    {
        this._A = A ?? throw new ArgumentNullException(nameof(A));
        this._diagonal = A.GetDiagonal();
        if (_diagonal.Any(d => d == 0))
            throw new InvalidOperationException("Matrix must have non-zero diagonal elements");
    }

    public void Relax(double[] b, double[] x, int maxIterations, double tolerance)
    {
        if (b.Length != _A.Rows || x.Length != _A.Rows)
            throw new ArgumentException("Vector length mismatch");

        double[] temp = new double[_A.Rows];
        for (int iter = 0; iter < maxIterations; iter++)
        {
            _A.Multiply(x, temp); // temp = Ax
            double norm = 0;
            for (int i = 0; i < _A.Rows; i++)
            {
                double residual = b[i] - temp[i]; // r = b - Ax
                double delta = Omega * (residual / _diagonal[i]);
                double newX = x[i] + delta;
                norm += delta * delta;
                x[i] = newX;
            }
            if (Math.Sqrt(norm) < tolerance) break;
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