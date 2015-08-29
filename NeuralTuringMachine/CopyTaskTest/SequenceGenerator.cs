using System;

namespace CopyTaskTest
{
    public static class SequenceGenerator
    {
        public static Random Rand = new Random(8);

        public static Tuple<double[][], double[][]> GenerateSequence(int length, int vectorSize)
        {
            double[][] data = new double[length][];
            for (int i = 0; i < length; i++)
            {
                data[i] = new double[vectorSize];
                for (int j = 0; j < vectorSize; j++)
                {
                    data[i][j] = (Rand.Next(0, 2) * 2) - 1;
                }
            }

            int sequenceLength = (length*2) + 2;

            double[][] input = new double[sequenceLength][];
            int inputVectorSize = vectorSize + 2;

            for (int i = 0; i < sequenceLength; i++)
            {
                input[i] = new double[inputVectorSize];
                for (int j = 0; j < inputVectorSize; j++)
                {
                    input[i][j] = -1;
                }

                if (i == 0)
                {
                    input[i][vectorSize] = 1;
                }
                else if (i <= length)
                {
                    for (int j = 0; j < vectorSize; j++)
                    {
                        input[i][j] = data[i - 1][j];
                    }
                }
                else if(i == (length + 1))
                {
                    input[i][vectorSize + 1] = 1;
                }
            }

            double[][] output = new double[sequenceLength][];
            for (int i = 0; i < sequenceLength; i++)
            {
                output[i] = new double[vectorSize];
                if (i >= (length + 2))
                {
                    for (int j = 0; j < vectorSize; j++)
                    {
                        output[i][j] = (data[i - (length + 2)][j] + 1) / 2;
                    }
                }
            }

            return new Tuple<double[][], double[][]>(input, output);
        }
    }
}
