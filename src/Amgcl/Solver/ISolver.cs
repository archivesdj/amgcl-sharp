namespace Amgcl.Solver;

public interface ISolver
{
    void Relax(double[] b, double[] x, int maxIterations, double tolerance);
    double[] Solve(double[] b, int maxIterations, double tolerance);
}