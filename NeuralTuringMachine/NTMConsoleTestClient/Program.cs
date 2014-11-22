using System;
using AForge.Neuro;
using NeuralTuringMachine.Learning;
using NeuralTuringMachine.Memory.Head;
using NeuralTuringMachine.Performance;

namespace NTMConsoleTestClient
{
    class Program
    {
        static void Main(string[] args)
        {
            const int inputCount = 5;
            const int outputCount = 3;
            const int readHeadCount = 1;
            const int writeHeadCount = 1;
            const int hiddenNeuronCount = 99;
            const int hiddenLayersCount = 3;
            const int memoryCellsCount = 10;
            const int memoryVectorLength = 5;
            const int maxConvolutionalShift = 1;

            Console.WriteLine("NEURAL TURING MACHINE TEST");
            Console.WriteLine("Input count: " + inputCount);
            Console.WriteLine("Output count: " + outputCount);
            Console.WriteLine("Read head count: " + readHeadCount);
            Console.WriteLine("Write head count: " + writeHeadCount);
            Console.WriteLine("Hidden neuron count: " + hiddenNeuronCount);
            Console.WriteLine("Hidden layers count: " + hiddenLayersCount);
            Console.WriteLine("Memory cells count: " + memoryCellsCount);
            Console.WriteLine("Memory vector length: " + memoryVectorLength);
            Console.WriteLine("Max convolutional shift: " + maxConvolutionalShift);

            NeuralTuringMachine.NeuralTuringMachine neuralTuringMachine =
                new NeuralTuringMachine.NeuralTuringMachine(
                    inputCount,
                    outputCount,
                    hiddenNeuronCount,
                    hiddenLayersCount,
                    new MemorySettings(memoryCellsCount, memoryVectorLength, maxConvolutionalShift, readHeadCount, writeHeadCount)
                    );

            BpttTeacher teacher = new BpttTeacher(neuralTuringMachine);

            //DATA FOR COPY TEST
            double[][] inputs = new double[10][];
            //                   START, COPY, D0, D1, D2
            inputs[0] = new double[] { 1, 0, 0, 0, 0 };
            inputs[1] = new double[] { 0, 0, 0, 0, 1 };
            inputs[2] = new double[] { 0, 0, 0, 1, 0 };
            inputs[3] = new double[] { 0, 0, 0, 1, 1 };
            inputs[4] = new double[] { 0, 0, 1, 0, 0 };
            inputs[5] = new double[] { 0, 1, 0, 0, 0 };
            inputs[6] = new double[] { 0, 0, 0, 0, 0 };
            inputs[7] = new double[] { 0, 0, 0, 0, 0 };
            inputs[8] = new double[] { 0, 0, 0, 0, 0 };
            inputs[9] = new double[] { 0, 0, 0, 0, 0 };

            double[][] outputs = new double[10][];
            //                        D0, D1, D2
            outputs[0] = new double[] { 0, 0, 0 };
            outputs[1] = new double[] { 0, 0, 0 };
            outputs[2] = new double[] { 0, 0, 0 };
            outputs[3] = new double[] { 0, 0, 0 };
            outputs[4] = new double[] { 0, 0, 0 };
            outputs[5] = new double[] { 0, 0, 0 };
            outputs[6] = new double[] { 0, 0, 1 };
            outputs[7] = new double[] { 0, 1, 0 };
            outputs[8] = new double[] { 0, 1, 1 };
            outputs[9] = new double[] { 1, 0, 0 };

            double lastError = 0;
            for (int i = 0; i < 1000; i++)
            {
                //GenerateInputAndOutput(inputs, outputs);
                teacher.Run(inputs, outputs);
                double error = PerfMeter.CalculateError(neuralTuringMachine, inputs, outputs);
                double enhancement = lastError - error;
                Console.WriteLine("ERROR in iteration " + i + " is\t" + error + "enhancement is\t" + enhancement);
                lastError = error;
            }

            Console.ReadLine();
        }

        private static void WriteController(Network network)
        {
            for (int i = 0; i < network.Layers.Length; i++)
            {
                Layer layer = network.Layers[i];
                Console.WriteLine("Layer: " + i);
                Neuron[] neurons = layer.Neurons;
                for (int j = 0; j < neurons.Length; j++)
                {
                    Console.WriteLine("Neuron: " + j);
                    foreach (double weight in neurons[j].Weights)
                    {
                        Console.Write(weight);
                        Console.Write(",");
                    }
                    Console.WriteLine();
                }
            }
        }

        private static void GenerateInputAndOutput(double[][] input, double[][] output)
        {
            Random random = new Random(DateTime.Now.Millisecond);

            for (int i = 0; i < 10; i++)
            {
                if (i == 0)
                {
                    input[i][0] = 1;
                }
                else
                {
                    input[i][0] = 0;
                }

                if (i == 5)
                {
                    input[i][1] = 1;
                }
                else
                {
                    input[i][1] = 0;
                }

                if (i < 5 && i > 0)
                {
                    for (int j = 2; j < 5; j++)
                    {
                        double randomNum = random.NextDouble();
                        if (randomNum < 0.5)
                        {
                            input[i][j] = 0;
                            output[i + 5][j - 2] = 0;
                        }
                        else
                        {
                            input[i][j] = 1;
                            output[i + 5][j - 2] = 1;
                        }
                    }
                }
                else
                {
                    for (int j = 2; j < 5; j++)
                    {
                        input[i][j] = 0;
                    }
                }

                if (i < 6)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        output[i][j] = 0;
                    }
                }
            }
        }
    }
}
