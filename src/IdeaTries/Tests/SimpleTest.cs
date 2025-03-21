using AmgSharp.Logics;

namespace AmgSharp.Tests;

public class SimpleTest {
    public static void Run(bool usingMatrixMarket = true)
    {

        SparseMatrix A = usingMatrixMarket ? InitFromMatrixMarket() : InitFromArrays();

        // 파일로 저장
        //A.ToMatrixMarket("output.mtx");

        PrintMatrix(A);

        // 우변 벡터 (임의 설정)
        double[] b = { 1, 2, 3, 4 };

        // AMG 솔버로 해 구하기
        AMGSolver solver = new AMGSolver(A);
        double[] x = solver.Solve(b);

        // 결과 출력
        Console.WriteLine("Solution:");
        foreach (var val in x) Console.WriteLine(val);
    }

     static SparseMatrix InitFromArrays()
    {
        // 4x4 희소 행렬 (Poisson 문제 예제)
        double[] values = { 4, -1, -1, -1, 4, -1, -1, 4, -1, -1, -1, 4 };
        int[] colIndices = { 0, 1, 2, 0, 1, 2, 1, 2, 3, 0, 2, 3 };
        int[] rowPointers = { 0, 3, 6, 9, 12 };
        return new SparseMatrix(4, 4, values, colIndices, rowPointers);
    }

    static SparseMatrix InitFromMatrixMarket()
    {
        // Matrix Market 파일 읽기
        return SparseMatrix.FromMatrixMarket("./Assets/matrix.mtx");
    }

    static void PrintMatrix(SparseMatrix A)
    {
        Console.WriteLine($"Rows: {A.Rows}, Cols: {A.Cols}, Non-zeros: {A.Values.Length}");
        for (int i = 0; i < A.Rows; i++)
        {
            Console.Write($"Row {i}: ");
            for (int j = A.RowPointers[i]; j < A.RowPointers[i + 1]; j++)
            {
                Console.Write($"({A.ColIndices[j]}, {A.Values[j]}) ");
            }
            Console.WriteLine();
        }
    }
}