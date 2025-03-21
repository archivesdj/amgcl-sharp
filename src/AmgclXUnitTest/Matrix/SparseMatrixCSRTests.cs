using Amgcl.Matrix;

namespace AmgclXUnitTest.Matrix;

public class SparseMatrixCSRTests
{
    private double[,] dense = new double[,]
    {
        { 1.0, 0.0, 2.0, 0.0 },
        { 0.0, 3.0, 0.0, 0.0 },
        { 0.0, 0.0, 0.0, 4.0 },
        { 5.0, 0.0, 0.0, 0.0 }
    };

    private string expectedString = "    1.00    0.00    2.00    0.00\n    0.00    3.00    0.00    0.00\n    0.00    0.00    0.00    4.00\n    5.00    0.00    0.00    0.00\n";
    private string expectedCSRString = "Values: 1 2 3 4 5\nColumn Indices: 0 2 1 3 0\nRow Pointers: 0 2 3 4 5";

    [Fact(Skip = "done")]
    public void Test_FromDense()
    {
        Console.WriteLine("SparseMatrixCSR FromDense Test");

        var matrix = SparseMatrixCSR.FromDense(dense);
        Console.WriteLine("Original Matrix:");
        var matrixString = matrix.Print();
        Assert.Equal(expectedString, matrixString);

        Console.WriteLine("\nCSR Representation:");
        var csrString = matrix.PrintCSR();
        Console.WriteLine(expectedCSRString);
    }

    [Fact(Skip = "done")]
    public void Test_Multiply()
    {
        Console.WriteLine("SparseMatrixCSR Multiply Test");

        var matrix = SparseMatrixCSR.FromDense(dense);
        double[] vector = new double[] { 1.0, 2.0, 3.0, 4.0 };
        double[] result = new double[4];
        matrix.Multiply(vector, result);

        double[] expected = new double[] { 7.0, 6.0, 16.0, 5.0 };
        Assert.Equal(expected, result);
    }
}