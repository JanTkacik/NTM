using System;
using AForge.Math.Metrics;

namespace NeuralTuringMachine.Performance
{
    public static class PerfMeter
    {
        public static double CalculateError(NeuralTuringMachine machine, double[][] input, double[][] output)
        {
            EuclideanDistance distance = new EuclideanDistance();
            double error = 0;
            int inputCount = input.Length;
            machine.Memory.ResetMemory();
            for (int i = 0; i < inputCount; i++)
            {
                double[] computedOutput = machine.Compute(input[i]);
                error += distance.GetDistance(computedOutput, output[i]);
            }

            return error;
        }
    }
}
