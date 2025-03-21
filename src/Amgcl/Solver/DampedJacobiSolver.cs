using Amgcl.Matrix;

namespace Amgcl.Solver;

public class DampedJacobiSolver : ISolver
{
    private readonly SparseMatrixCSR matrix;
   
    private readonly double[] diagonal;

    private double omega = 1.0;
    public double Omega 
    {
        get { return omega; }
        set {
            if (value <= 0 || value > 1)
                throw new ArgumentException("Omega must be between 0 and 1");
            omega = value;
        }
    }

    public DampedJacobiSolver(SparseMatrixCSR matrix)
    {
        this.matrix = matrix ?? throw new ArgumentNullException(nameof(matrix));
        this.diagonal = matrix.GetDiagonal();
        if (diagonal.Any(d => d == 0))
            throw new InvalidOperationException("Matrix must have non-zero diagonal elements");
    }

    public void Relax(double[] b, double[] x, int maxIterations, double tolerance)
    {
        if (b.Length != matrix.Rows || x.Length != matrix.Rows)
            throw new ArgumentException("Vector length mismatch");

        double[] temp = new double[matrix.Rows];
        for (int iter = 0; iter < maxIterations; iter++)
        {
            matrix.Multiply(x, temp); // temp = Ax
            double norm = 0;
            for (int i = 0; i < matrix.Rows; i++)
            {
                double residual = b[i] - temp[i]; // r = b - Ax
                double delta = Omega * (residual / diagonal[i]);
                double newX = x[i] + delta;
                norm += delta * delta;
                x[i] = newX;
            }
            if (Math.Sqrt(norm) < tolerance) break;
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
}