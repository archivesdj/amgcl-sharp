public class AMGSolver
{
    private List<IAMGLevel> levels;
    private List<JacobiRelaxation> relaxations;

    public AMGSolver(SparseMatrix A)
    {
        levels = new List<IAMGLevel>();
        relaxations = new List<JacobiRelaxation>();
        BuildHierarchy(A);
    }

    private void BuildHierarchy(SparseMatrix A)
    {
        try
        {
            var level = new AMGLevel(A);
            levels.Add(level);
            relaxations.Add(new JacobiRelaxation(A));

            int maxLevels = 10;
            while (level.A.Rows > 4 && levels.Count < maxLevels)
            {
                level.BuildCoarseLevel(strengthThreshold: 0.05, dampingFactor: 0.5);
                if (level.CoarseA == null || level.CoarseSize >= level.A.Rows * 0.9) break; // 10% 미만 감소 시 중단
                level = new AMGLevel(level.CoarseA);
                levels.Add(level);
                relaxations.Add(new JacobiRelaxation(level.A));
            }
            Console.WriteLine($"Hierarchy built with {levels.Count} levels.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error in BuildHierarchy: {ex.Message}");
            throw;
        }
    }

    public double[] Solve(double[] b, int maxIterations = 200, double tolerance = 1e-8)
    {
        if (b == null || b.Length != levels[0].A.Rows)
            throw new ArgumentException("Invalid right-hand side vector.");

        try
        {
            double[] x = new double[b.Length];
            for (int iter = 0; iter < maxIterations; iter++)
            {
                VCycle(0, x, b);
                double[] r = Subtract(b, levels[0].A.Multiply(x));
                double residual = Norm(r);
                Console.WriteLine($"Iteration {iter + 1}: Residual = {residual}");
                if (residual < tolerance)
                {
                    Console.WriteLine($"Converged in {iter + 1} iterations.");
                    return x;
                }
            }
            Console.WriteLine("Failed to converge within max iterations.");
            return x;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error in Solve: {ex.Message}");
            throw;
        }
    }

    private void VCycle(int levelIndex, double[] x, double[] b)
    {
        if (levelIndex >= levels.Count)
            throw new ArgumentException($"Invalid level index: {levelIndex}");

        var level = levels[levelIndex];
        try
        {
            if (levelIndex == levels.Count - 1)
            {
                relaxations[levelIndex].Relax(x, b, 10);
                return;
            }

            relaxations[levelIndex].Relax(x, b, 3);
            double[] r = Subtract(b, level.A.Multiply(x));
            if (level.R == null) throw new InvalidOperationException("Restriction matrix is null.");
            double[] rc = level.R.Multiply(r);

            double[] xc = new double[level.CoarseSize];
            VCycle(levelIndex + 1, xc, rc);

            if (level.P == null) throw new InvalidOperationException("Interpolation matrix is null.");
            double[] e = level.P.Multiply(xc);
            for (int i = 0; i < x.Length; i++)
                x[i] += e[i];

            relaxations[levelIndex].Relax(x, b, 3);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error in VCycle at level {levelIndex}: {ex.Message}");
            throw;
        }
    }

    private double[] Subtract(double[] a, double[] b) => a.Zip(b, (x, y) => x - y).ToArray();
    private double Norm(double[] v) => Math.Sqrt(v.Sum(x => x * x));
}