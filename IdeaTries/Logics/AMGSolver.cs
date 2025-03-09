using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.CPU;

namespace AmgSharp.Logics;

public class AMGSolver : IDisposable
{
    private readonly List<IAMGLevel> levels;
    private readonly List<JacobiRelaxation> relaxations;
    private readonly Accelerator accelerator;
    private readonly Context context;
    private readonly int maxLevels; // 최대 레벨을 저장하는 필드 추가

    public AMGSolver(SparseMatrix A, int maxLevels = 10)
    {
        this.maxLevels = maxLevels;

        context = Context.Create(builder => builder.Cuda().CPU());
        accelerator = context.GetPreferredDevice(preferCPU: false)
                            .CreateAccelerator(context);

        // 장치 정보 출력
        string deviceType = accelerator.AcceleratorType switch
        {
            AcceleratorType.Cuda => "GPU (CUDA)",
            AcceleratorType.CPU => "CPU",
            _ => "Unknown"
        };
        Console.WriteLine($"Using {deviceType} for computation (Name: {accelerator.Name}, MaxThreads: {accelerator.MaxNumThreads})");
        Console.WriteLine($"Maximum multi-level set to: {maxLevels}");

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
            relaxations.Add(new JacobiRelaxation(A, accelerator));

            while (level.A.Rows > 4 && levels.Count < maxLevels)
            {
                level.BuildCoarseLevel(strengthThreshold: 0.9, dampingFactor: 0.8);
                if (level.CoarseA == null || level.CoarseSize > level.A.Rows * 0.6)
                {
                    Console.WriteLine($"Stopping hierarchy at {levels.Count} levels due to insufficient size reduction.");
                    break;
                }
                level = new AMGLevel(level.CoarseA);
                levels.Add(level);
                relaxations.Add(new JacobiRelaxation(level.A, accelerator));
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
            var totalStopwatch = System.Diagnostics.Stopwatch.StartNew();

            double[] initialR = Subtract(b, levels[0].A.Multiply(x));
            double bNorm = Norm(b);
            double diagSum = 0;
            for (int i = 0; i < levels[0].A.Rows; i++)
                for (int j = levels[0].A.RowPointers[i]; j < levels[0].A.RowPointers[i + 1]; j++)
                    if (levels[0].A.ColIndices[j] == i)
                        diagSum += levels[0].A.Values[j];
            //Console.WriteLine($"Initial residual norm: {Norm(initialR)}, b norm: {bNorm}, A diagonal sum: {diagSum}");

            for (int iter = 0; iter < maxIterations; iter++)
            {
                var iterStopwatch = System.Diagnostics.Stopwatch.StartNew();
                VCycle(0, x, b);
                iterStopwatch.Stop();

                double[] r = Subtract(b, levels[0].A.Multiply(x));
                double residual = Norm(r);
                Console.WriteLine($"Iteration {iter + 1}: Residual = {residual}, Time = {iterStopwatch.Elapsed.TotalSeconds:F6} seconds");

                if (residual < tolerance)
                {
                    totalStopwatch.Stop();
                    Console.WriteLine($"Converged in {iter + 1} iterations. Total time = {totalStopwatch.Elapsed.TotalSeconds:F6} seconds.");
                    return x;
                }
            }
            totalStopwatch.Stop();
            Console.WriteLine($"Failed to converge within {maxIterations} iterations. Total time = {totalStopwatch.Elapsed.TotalSeconds:F6} seconds.");
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
            int relaxIterations = 10;
            if (levelIndex == levels.Count - 1)
            {
                relaxations[levelIndex].Relax(x, b, 20);
                return;
            }

            relaxations[levelIndex].Relax(x, b, relaxIterations);
            double[] r = Subtract(b, level.A.Multiply(x));
            //Console.WriteLine($"Level {levelIndex}: Pre-restriction residual = {Norm(r)}");
            if (level.R == null) throw new InvalidOperationException("Restriction matrix is null.");
            double[] rc = level.R.Multiply(r);

            double[] xc = new double[level.CoarseSize];
            VCycle(levelIndex + 1, xc, rc);

            if (level.P == null) throw new InvalidOperationException("Interpolation matrix is null.");
            double[] e = level.P.Multiply(xc);
            for (int i = 0; i < x.Length; i++)
                x[i] += e[i];

            relaxations[levelIndex].Relax(x, b, relaxIterations);
            double[] postR = Subtract(b, level.A.Multiply(x));
            //Console.WriteLine($"Level {levelIndex}: Post-relaxation residual = {Norm(postR)}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error in VCycle at level {levelIndex}: {ex.Message}");
            throw;
        }
    }

    public void Dispose()
    {
        foreach (var relaxation in relaxations)
            relaxation.Dispose();
        accelerator?.Dispose();
        context?.Dispose();
    }

    //

    public static double[] Subtract(double[] a, double[] b) => a.Zip(b, (x, y) => x - y).ToArray();
    public static double Norm(double[] v) => Math.Sqrt(v.Sum(x => x * x));
}