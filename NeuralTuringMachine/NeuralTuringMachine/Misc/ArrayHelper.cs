using System;

namespace NeuralTuringMachine.Misc
{
    public static class ArrayHelper
    {
        public static double[] CloneArray(double[] source)
        {
            int length = source.Length;
            double[] newArray = new double[length];
            Array.Copy(source, newArray, length);
            return newArray;
        }

        public static double[][] CloneArray(double[][] source)
        {
            int length = source.Length;
            double[][] newArray = new double[length][];
            for (int i = 0; i < length; i++)
            {
                newArray[i] = CloneArray(source[i]);
            }
            return newArray;
        }

        public static void NormalizeVector(double[] vector)
        {
            double sum = 0;
            var length = vector.Length;
            for (int i = 0; i < length; i++)
            {
                sum += vector[i];
            }
            double multiplier = 1 / sum;
            for (int i = 0; i < vector.Length; i++)
            {
                vector[i] *= multiplier;
            }
        }
    }
}
