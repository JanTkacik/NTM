using System;
using NTM2;
using NTM2.Controller;
using NTM2.Learning;

namespace CopyTaskTest
{
    class Program
    {
        static void Main(string[] args)
        {
            //TODO args parsing shit
            Random rand = new Random(8);

            const int vectorSize = 8;
            const int controllerSize = 100;
            const int headsCount = 1;
            const int memoryN = 128;
            const int memoryM = 20;

            NTMController controller = new NTMController(vectorSize + 2, vectorSize, controllerSize, headsCount, memoryN, memoryM);
            //Randomize weights
            controller.UpdateWeights(unit => unit.Value = (rand.NextDouble() - 0.5));

            RMSPropTeacher rmsPropTeacher = new RMSPropTeacher(controller);
            for (int i = 1; i < 100000; i++)
            {
                Tuple<double[][], double[][]> sequence = SequenceGenerator.GenerateSequence(rand.Next(20) + 1, vectorSize);
                Ntm[] machines = rmsPropTeacher.Train(sequence.Item1, sequence.Item2, 0.95, 0.5, 0.001, 0.001);
                
                if (i%100 == 0)
                {
                    double error = CalculateLogLoss(sequence.Item2, machines);
                    double averageError = error/(sequence.Item2.Length*sequence.Item2[0].Length);
                    Console.WriteLine("Iteration: {0}, average error: {1}",i,averageError);
                }
            }
        }

        private static double CalculateLogLoss(double[][] knownOutput, Ntm[] trainedMachines)
        {
            double totalLoss = 0;
            int okt = knownOutput.Length - ((knownOutput.Length - 2) / 2);
            for (int t = 0; t < knownOutput.Length; t++)
            {
                for (int i = 0; i < knownOutput[t].Length; i++)
                {
                    double expected = knownOutput[t][i];
                    double real = trainedMachines[t].Controller.Output[i].Value;
                    if (t >= okt)
                    {
                        totalLoss += (expected*Math.Log(real, 2)) + ((1 - expected)*Math.Log(1 - real, 2));
                    }
                }
            }
            return -totalLoss;
        }
    }
}
