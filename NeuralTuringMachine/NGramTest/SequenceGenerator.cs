using System;
using AForge.Math.Random;

namespace NGramTest
{
    public static class SequenceGenerator
    {
        public static Random Random = new Random(10);

        public static Tuple<double[][], double[][]> GenerateSequence(double[] propabilities)
        {
            int n = (int)(Math.Log(propabilities.Length, 2));
            const int seqLen = 200;
            double[][] input = new double[seqLen + 1][];

            for (int i = 0; i < n; i++)
            {
                input[i] = new double[]{Random.Next(2)};
            }

            for (int i = n; i < seqLen + 1; i++)
            {
                double[][] ngram = new double[n][];
                for (int j = i - n; j < i; j++)
                {
                    ngram[j - i + n] = input[j];
                }
                int propIndex = Binarize(ngram);
                if (Random.NextDouble() < propabilities[propIndex])
                {
                    input[i] = new double[] { 1 };
                }
                else
                {
                    input[i] = new double[] { 0 };
                }
            }

            double[][] output = new double[seqLen][];
            for (int i = 0; i < seqLen; i++)
            {
                if (i < (n - 1))
                {
                    output[i] = new double[] {0};
                }
                else
                {
                    output[i] = input[i + 1];
                }
            }
            double[][] realInput = new double[seqLen][];
            for (int i = 0; i < seqLen; i++)
            {
                realInput[i] = input[i];
            }

            return new Tuple<double[][], double[][]>(realInput, output);
        }

        public static int Binarize(double[][] seq)
        {
            int idx = 0;
            for (int i = 0; i < seq.Length; i++)
            {
                double[] data = seq[i];
                idx += (int)data[0] * (1 << i);
            }
            return idx;
        }

        public static double[] GeneratePropabilities()
        {
            const int n = 5;
            double[] probabilities = new double[1 << n];
            for (int i = 0; i < (1 << n); i++)
            {
                probabilities[i] = Beta();
            }
            return probabilities;
        }

        public static double Beta()
        {
            GaussianGenerator generator = new GaussianGenerator(0, 1);
            double x = generator.Next();
            x *= x;
            x *= 0.5;
            double y = generator.Next();
            y *= y;
            y *= 0.5;
            return x/(x + y);
        }

    }
}
