using System;
using System.IO;
using AForge.Math.Metrics;
using AForge.Neuro;
using NeuralTuringMachine.Memory.Head;
using NeuralTuringMachine.Misc;

namespace NeuralTuringMachine.Performance
{
    public static class PerfMeter
    {
        private static double _lastError = 0;
        private static double _bestError = double.PositiveInfinity;
        private static readonly StreamWriter StreamWriter = File.CreateText("NTM.txt");

        public static double CalculateError(NeuralTuringMachine machine, double[][] input, double[][] output)
        {
            EuclideanDistance distance = new EuclideanDistance();
            double error = 0;
            int inputCount = input.Length;
            machine.Memory.Randomize();

            //LOG MEMORY OPERATIONS
            double[][] computedOutputs = new double[inputCount][];
            double[][] lastReadWeights = new double[inputCount][];
            double[][] lastWriteWeights = new double[inputCount][];
            double[][] lastWriteAdds = new double[inputCount][];
            double[][] lastWriteErases = new double[inputCount][];
            double[][] readData = new double[inputCount][];

            for (int i = 0; i < inputCount; i++)
            {
                double[] computedOutput = machine.Compute(input[i]);
                computedOutputs[i] = computedOutput;
                error += distance.GetDistance(computedOutput, output[i]);
                
                //LOG MEMORY OPERATIONS
                ReadHead readHead = machine.GetReadHead(0);
                WriteHead writeHead = machine.GetWriteHead(0);
                double[] lastReadData = readHead.LastReadData ??
                                        new double[machine.Memory.MemorySettings.MemoryVectorLength];
                readData[i] = ArrayHelper.CloneArray(lastReadData);
                lastReadWeights[i] = ArrayHelper.CloneArray(readHead.LastWeights);
                lastWriteWeights[i] = ArrayHelper.CloneArray(writeHead.LastWeights);
                lastWriteAdds[i] = ArrayHelper.CloneArray(writeHead.AddVector);
                lastWriteErases[i] = ArrayHelper.CloneArray(writeHead.EraseVector);
            }

            Tuple<int, int> positionA = ConsoleWriterHelper.LogDoubleTable("Inputs", input, 0, 0);
            Tuple<int, int> positionB = ConsoleWriterHelper.LogDoubleTable("Required outputs", output, positionA.Item1 + 1, 0);
            Tuple<int, int> positionC = ConsoleWriterHelper.LogDoubleTable("Computed outputs", computedOutputs, positionB.Item1 + 1, 0);
            Tuple<int, int> positionD = ConsoleWriterHelper.LogDoubleTable("Read weights", lastReadWeights, 0, positionC.Item2 + 1);
            Tuple<int, int> positionE = ConsoleWriterHelper.LogDoubleTable("Read data", readData, positionD.Item1 + 1, positionC.Item2 + 1);
            Tuple<int, int> positionF = ConsoleWriterHelper.LogDoubleTable("Write weights", lastWriteWeights, positionE.Item1 + 1, positionC.Item2 + 1);
            Tuple<int, int> positionG = ConsoleWriterHelper.LogDoubleTable("Write erase", lastWriteErases, positionF.Item1 + 1, positionC.Item2 + 1);
            Tuple<int, int> positionH = ConsoleWriterHelper.LogDoubleTable("Write add", lastWriteAdds, positionG.Item1 + 1, positionC.Item2 + 1);

            if (error < _bestError)
            {
                _bestError = error;
                FileStream bestControllerStream = File.Create("BestController");
                machine.Controller.Save(bestControllerStream);
                bestControllerStream.Close();
            }

            Console.SetCursorPosition(0, positionH.Item2 + 1);
            Console.ForegroundColor = ConsoleColor.DarkYellow;
            Console.WriteLine("Error: {0:#######.0000}", error);
            Console.WriteLine("Improvement: {0:#####0.00000000000000}", _lastError - error);
            Console.WriteLine("Best error: {0:#######.0000}", _bestError);

            StreamWriter.WriteLine("Error: {0:#######.0000}", error);
            StreamWriter.WriteLine("Best error: {0:#######.0000}", _bestError);
            WriteController(machine.Controller);

            _lastError = error;
            return error;
        }

        private static void WriteController(Network network)
        {
            for (int i = 0; i < network.Layers.Length; i++)
            {
                Layer layer = network.Layers[i];
                StreamWriter.WriteLine("Layer: " + i);
                Neuron[] neurons = layer.Neurons;
                for (int j = 0; j < neurons.Length; j++)
                {
                    StreamWriter.WriteLine("Neuron: " + j);
                    ActivationNeuron neuron = (ActivationNeuron)neurons[j];
                    foreach (double weight in neuron.Weights)
                    {
                        StreamWriter.Write(weight);
                        StreamWriter.Write(",");
                    }
                    StreamWriter.WriteLine();
                    StreamWriter.WriteLine("Threshold: " + neuron.Threshold);
                }
            }
        }
    }
}
