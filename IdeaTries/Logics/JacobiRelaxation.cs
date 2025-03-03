
public class JacobiRelaxation
{
    private SparseMatrix A;
    private double[] diag; // 대각 행렬 값

    public JacobiRelaxation(SparseMatrix matrix)
    {
        A = matrix;
        diag = new double[A.Rows];
        for (int i = 0; i < A.Rows; i++)
        {
            for (int j = A.RowPointers[i]; j < A.RowPointers[i + 1]; j++)
            {
                if (A.ColIndices[j] == i)
                {
                    diag[i] = A.Values[j];
                    break;
                }
            }
        }
    }

    public void Relax(double[] x, double[] b, int iterations)
    {
        for (int iter = 0; iter < iterations; iter++)
        {
            double[] Ax = A.Multiply(x);
            for (int i = 0; i < A.Rows; i++)
            {
                x[i] += (b[i] - Ax[i]) / diag[i];
            }
        }
    }
}