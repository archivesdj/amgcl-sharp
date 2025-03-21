using System;
using System.IO;
using System.Globalization;
using System.Linq;
using System.Text;

namespace Amgcl.Matrix;

public class SparseMatrixCSR
{
    public int Rows { get; }
    public int Cols { get; }
    public double[] Values { get; }
    public int[] ColIndices { get; }
    public int[] RowPointers { get; }
    public int NonZeroCount => Values.Length;

    public SparseMatrixCSR(int rows, int cols, double[] values, int[] colIndices, int[] rowPointers)
    {
        if (rows < 0 || cols < 0 || values.Length != colIndices.Length || rowPointers.Length != rows + 1)
            throw new ArgumentException("Invalid CSR data dimensions");

        Rows = rows;
        Cols = cols;
        Values = values;
        ColIndices = colIndices;
        RowPointers = rowPointers;
    }

    public double[,] ToDense()
    {
        double[,] dense = new double[Rows, Cols];
        for (int i = 0; i < Rows; i++)
        {
            for (int j = RowPointers[i]; j < RowPointers[i + 1]; j++)
            {
                int col = ColIndices[j];
                dense[i, col] = Values[j];
            }
        }
        return dense;
    }

    public static SparseMatrixCSR FromDense(double[,] denseMatrix)
    {
        int rows = denseMatrix.GetLength(0);
        int cols = denseMatrix.GetLength(1);
        var tempValues = new System.Collections.Generic.List<double>();
        var tempColIndices = new System.Collections.Generic.List<int>();
        var tempRowPointers = new int[rows + 1];

        int nnz = 0;
        for (int i = 0; i < rows; i++)
        {
            tempRowPointers[i] = nnz;
            for (int j = 0; j < cols; j++)
            {
                double value = denseMatrix[i, j];
                if (value != 0)
                {
                    tempValues.Add(value);
                    tempColIndices.Add(j);
                    nnz++;
                }
            }
        }
        tempRowPointers[rows] = nnz;

        return new SparseMatrixCSR(rows, cols, tempValues.ToArray(), tempColIndices.ToArray(), tempRowPointers);
    }

    public double this[int row, int col]
    {
        get
        {
            if (row < 0 || row >= Rows || col < 0 || col >= Cols)
                throw new ArgumentOutOfRangeException("Row or column index out of range");

            int start = RowPointers[row];
            int end = RowPointers[row + 1];
            for (int i = start; i < end; i++)
            {
                if (ColIndices[i] == col)
                    return Values[i];
            }
            return 0;
        }
    }

    public void SaveToMatrixMarketCoordinate(string filePath)
    {
        using var writer = new StreamWriter(filePath);
        writer.WriteLine("%%MatrixMarket matrix coordinate real general");
        writer.WriteLine($"% Generated by SparseMatrixCSR on {DateTime.Now}");
        writer.WriteLine($"{Rows} {Cols} {Values.Length}");

        for (int i = 0; i < Rows; i++)
        {
            int start = RowPointers[i];
            int end = RowPointers[i + 1];
            for (int j = start; j < end; j++)
            {
                writer.WriteLine($"{i + 1} {ColIndices[j] + 1} {Values[j].ToString(CultureInfo.InvariantCulture)}");
            }
        }
    }

    public void SaveToMatrixMarketArray(string filePath)
    {
        using var writer = new StreamWriter(filePath);
        writer.WriteLine("%%MatrixMarket matrix array real general");
        writer.WriteLine($"% Generated by SparseMatrixCSR on {DateTime.Now}");
        writer.WriteLine($"{Rows} {Cols}");

        for (int j = 0; j < Cols; j++)
        {
            for (int i = 0; i < Rows; i++)
            {
                double value = this[i, j];
                writer.WriteLine(value.ToString(CultureInfo.InvariantCulture));
            }
        }
    }

    public double[] GetDiagonal()
    {
        double[] diagonal = new double[Rows];
        for (int i = 0; i < Rows; i++)
        {
            int start = RowPointers[i];
            int end = RowPointers[i + 1];
            for (int j = start; j < end; j++)
            {
                if (ColIndices[j] == i)
                {
                    diagonal[i] = Values[j];
                    break;
                }
            }
        }
        return diagonal;
    }

    // Matrix-vector multiplication
    public void Multiply(double[] x, double[] result)
    {
        if (x.Length != Cols || result.Length != Rows)
            throw new ArgumentException("Vector length does not match matrix dimensions.");

        for (int i = 0; i < Rows; i++)
        {
            result[i] = 0;
            for (int j = RowPointers[i]; j < RowPointers[i + 1]; j++)
                result[i] += Values[j] * x[ColIndices[j]];
        }
    }

    // Matrix-matrix multiplication
    public SparseMatrixCSR MultiplyMatrix(SparseMatrixCSR right)
    {
        if (Cols != right.Rows)
            throw new ArgumentException("Matrix dimensions do not match for multiplication.");

        int m = Rows;
        int n = right.Cols;
        var values = new List<double>();
        var colIndices = new List<int>();
        var rowPointers = new List<int> { 0 };
        var tempRow = new Dictionary<int, double>();

        for (int i = 0; i < m; i++)
        {
            tempRow.Clear();
            for (int j = RowPointers[i]; j < RowPointers[i + 1]; j++)
            {
                int k = ColIndices[j];
                double leftVal = Values[j];
                for (int l = right.RowPointers[k]; l < right.RowPointers[k + 1]; l++)
                {
                    int col = right.ColIndices[l];
                    double val = leftVal * right.Values[l];
                    tempRow.TryGetValue(col, out double current);
                    tempRow[col] = current + val;
                }
            }

            foreach (var kvp in tempRow.Where(kvp => kvp.Value != 0))
            {
                values.Add(kvp.Value);
                colIndices.Add(kvp.Key);
            }
            rowPointers.Add(values.Count);
        }

        return new SparseMatrixCSR(m, n, values.ToArray(), colIndices.ToArray(), rowPointers.ToArray());
    }

    // Transpose matrix
    public SparseMatrixCSR Transpose()
    {
        int nRowsTranspose = Cols;
        int nColsTranspose = Rows;
        int nnz = Values.Length;

        int[] rowCountsTranspose = new int[nRowsTranspose];
        for (int i = 0; i < Rows; i++)
        {
            for (int j = RowPointers[i]; j < RowPointers[i + 1]; j++)
                rowCountsTranspose[ColIndices[j]]++;
        }

        int[] rowPointersTranspose = new int[nRowsTranspose + 1];
        for (int i = 0; i < nRowsTranspose; i++)
            rowPointersTranspose[i + 1] = rowPointersTranspose[i] + rowCountsTranspose[i];

        double[] valuesTranspose = new double[nnz];
        int[] colIndicesTranspose = new int[nnz];
        int[] tempPos = new int[nRowsTranspose];
        for (int i = 0; i < nRowsTranspose; i++)
            tempPos[i] = rowPointersTranspose[i];

        for (int i = 0; i < Rows; i++)
        {
            for (int j = RowPointers[i]; j < RowPointers[i + 1]; j++)
            {
                int col = ColIndices[j];
                int pos = tempPos[col];
                valuesTranspose[pos] = Values[j];
                colIndicesTranspose[pos] = i;
                tempPos[col]++;
            }
        }

        return new SparseMatrixCSR(nRowsTranspose, nColsTranspose, valuesTranspose, colIndicesTranspose, rowPointersTranspose);
    }

    // Compute D^-1 (inverse of diagonal matrix) as a sparse matrix
    public SparseMatrixCSR ComputeDiagonalInverse()
    {
        int n = Rows;
        var values = new double[n];
        var colIndices = new int[n];
        var rowPointers = new int[n + 1];

        for (int i = 0; i < n; i++)
        {
            rowPointers[i] = i;
            colIndices[i] = i;
            double diag = 0;
            for (int j = RowPointers[i]; j < RowPointers[i + 1]; j++)
            {
                if (ColIndices[j] == i)
                {
                    diag = Values[j];
                    break;
                }
            }
            values[i] = (diag != 0) ? 1.0 / diag : 0; // Handle zero diagonal
        }
        rowPointers[n] = n;

        return new SparseMatrixCSR(n, n, values, colIndices, rowPointers);
    }

    // Create an identity matrix
    public SparseMatrixCSR CreateIdentityMatrix()
    {
        int size = Rows;
        var values = new double[size];
        var colIndices = new int[size];
        var rowPointers = new int[size + 1];

        for (int i = 0; i < size; i++)
        {
            values[i] = 1.0;
            colIndices[i] = i;
            rowPointers[i] = i;
        }
        rowPointers[size] = size;

        return new SparseMatrixCSR(size, size, values, colIndices, rowPointers);
    }

    // Scale a matrix by a scalar
    public SparseMatrixCSR Scale(double scalar)
    {
        var values = new double[NonZeroCount];
        Array.Copy(Values, values, values.Length);
        for (int i = 0; i < values.Length; i++)
            values[i] *= scalar;

        return new SparseMatrixCSR(Rows, Cols, values, ColIndices, RowPointers);
    }

    // Subtract one matrix from another (no NonZeroCount constraint)
    public SparseMatrixCSR Subtract(SparseMatrixCSR b)
    {
        if (Rows != b.Rows || Cols != b.Cols)
            throw new ArgumentException("Matrices must have the same dimensions.");

        var values = new List<double>();
        var colIndices = new List<int>();
        var rowPointers = new List<int> { 0 };
        var tempRow = new Dictionary<int, double>();

        for (int i = 0; i < Rows; i++)
        {
            tempRow.Clear();

            // Add contributions from this matrix (A)
            for (int j = RowPointers[i]; j < RowPointers[i + 1]; j++)
            {
                int col = ColIndices[j];
                tempRow[col] = Values[j];
            }

            // Subtract contributions from b
            for (int j = b.RowPointers[i]; j < b.RowPointers[i + 1]; j++)
            {
                int col = b.ColIndices[j];
                tempRow.TryGetValue(col, out double current);
                tempRow[col] = current - b.Values[j];
            }

            // Collect non-zero elements
            foreach (var kvp in tempRow.Where(kvp => kvp.Value != 0))
            {
                values.Add(kvp.Value);
                colIndices.Add(kvp.Key);
            }
            rowPointers.Add(values.Count);
        }

        return new SparseMatrixCSR(Rows, Cols, values.ToArray(), colIndices.ToArray(), rowPointers.ToArray());
    }

    // for testing

    public string Print()
    {
        var sb = new StringBuilder();
        for (int i = 0; i < Rows; i++)
        {
            for (int j = 0; j < Cols; j++)
            {
                sb.Append($"{this[i, j],8:F2}");
            }
            sb.AppendLine();
        }
        return sb.ToString();
    }

    public string PrintCSR()
    {
        var sb = new StringBuilder();
        sb.AppendLine("CSR Format:");
        sb.Append("Values: ");
        sb.AppendLine(string.Join(" ", Values));
        sb.Append("Column Indices: ");
        sb.AppendLine(string.Join(" ", ColIndices));
        sb.Append("Row Pointers: ");
        sb.AppendLine(string.Join(" ", RowPointers));
        return sb.ToString();
    }
}
