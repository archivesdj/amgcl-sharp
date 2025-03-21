using Amgcl.Matrix;

namespace AmgclXUnitTest.Matrix;

public class SparseMatrixTests
{
    [Fact(Skip = "done")]
    public void Test_SaveAndLoad()
    {
        Console.WriteLine("SparseMatrix Save and Load Test");

        SparseMatrix matrix = new SparseMatrix(3, 3);
        matrix[0, 0] = 1.0;
        matrix[0, 1] = 2.0;
        matrix[1, 1] = 3.0;
        matrix[2, 2] = 4.0;

        matrix.SaveToMatrixMarketCoordinate("matrix_coord.mtx");
        var coordMatrix = SparseMatrix.LoadFromMatrixMarketCoordinate("matrix_coord.mtx");

        matrix.SaveToMatrixMarketArray("matrix_array.mtx");
        var arrayMatrix = SparseMatrix.LoadFromMatrixMarketArray("matrix_array.mtx");

        var expectedString = "    1.00    2.00    0.00\n    0.00    3.00    0.00\n    0.00    0.00    4.00\n";
        Console.WriteLine("Original Matrix:");
        var matrixString = matrix.Print();
        Assert.Equal(expectedString, matrixString);

        Console.WriteLine("\nLoaded from Coordinate Format:");
        var coordMatrixString = coordMatrix.Print();
        Assert.Equal(expectedString, coordMatrixString);

        Console.WriteLine("\nLoaded from Array Format:");
        var arrayMatrixString = arrayMatrix.Print();
        Assert.Equal(expectedString, arrayMatrixString);
    }

    [Fact(Skip = "done")]
    public void Test_ToCSR()
    {
        Console.WriteLine("SparseMatrix ToCSR Test");

        SparseMatrix matrix = new SparseMatrix(4, 4);
        matrix[0, 0] = 1.0;
        matrix[0, 2] = 2.0;
        matrix[1, 1] = 3.0;
        matrix[2, 3] = 4.0;
        matrix[3, 0] = 5.0;

        var expectedString = "    1.00    0.00    2.00    0.00\n    0.00    3.00    0.00    0.00\n    0.00    0.00    0.00    4.00\n    5.00    0.00    0.00    0.00\n";
        Console.WriteLine("Original Matrix:");
        var matrixString = matrix.Print();
        Assert.Equal(expectedString, matrixString);

        var expectedCSRString = "Values: 1 2 3 4 5\nColumn Indices: 0 2 1 3 0\nRow Pointers: 0 2 3 4 5";
        Console.WriteLine("\nCSR Representation:");
        var csrString = matrix.PrintCSR();
        Console.WriteLine(expectedCSRString);
    }
}
