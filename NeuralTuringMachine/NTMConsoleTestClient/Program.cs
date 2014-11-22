using System;
using AForge.Neuro;
using NeuralTuringMachine.Controller;
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
            const int hiddenNeuronCount = 10;
            const int hiddenLayersCount = 1;
            const int memoryCellsCount = 10;
            const int memoryVectorLength = 3;
            const int maxConvolutionalShift = 1;

            //Console.WriteLine("NEURAL TURING MACHINE TEST");
            //Console.WriteLine("Input count: " + inputCount);
            //Console.WriteLine("Output count: " + outputCount);
            //Console.WriteLine("Read head count: " + readHeadCount);
            //Console.WriteLine("Write head count: " + writeHeadCount);
            //Console.WriteLine("Hidden neuron count: " + hiddenNeuronCount);
            //Console.WriteLine("Hidden layers count: " + hiddenLayersCount);
            //Console.WriteLine("Memory cells count: " + memoryCellsCount);
            //Console.WriteLine("Memory vector length: " + memoryVectorLength);
            //Console.WriteLine("Max convolutional shift: " + maxConvolutionalShift);

            NeuralTuringMachine.NeuralTuringMachine neuralTuringMachine =
                new NeuralTuringMachine.NeuralTuringMachine(
                    inputCount,
                    outputCount,
                    hiddenNeuronCount,
                    hiddenLayersCount,
                    new MemorySettings(memoryCellsCount, memoryVectorLength, maxConvolutionalShift, readHeadCount, writeHeadCount)
                    );

            BpttTeacher teacher = new BpttTeacher(neuralTuringMachine);
            BpttTeacherWitKnownMemoryState teacherWitKnownMemory = new BpttTeacherWitKnownMemoryState(neuralTuringMachine);

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

            double[][] inputsWithReadData = new double[10][];
            //                   START, COPY, D0, D1, D2
            inputsWithReadData[0] = new double[] { 1, 0, 0, 0, 0, 0, 0, 0 };
            inputsWithReadData[1] = new double[] { 0, 0, 0, 0, 1, 0, 0, 0 };
            inputsWithReadData[2] = new double[] { 0, 0, 0, 1, 0, 0, 0, 0 };
            inputsWithReadData[3] = new double[] { 0, 0, 0, 1, 1, 0, 0, 0 };
            inputsWithReadData[4] = new double[] { 0, 0, 1, 0, 0, 0, 0, 0 };
            inputsWithReadData[5] = new double[] { 0, 1, 0, 0, 0, 0, 0, 0 };
            inputsWithReadData[6] = new double[] { 0, 0, 0, 0, 0, 0, 0, 1 };
            inputsWithReadData[7] = new double[] { 0, 0, 0, 0, 0, 0, 1, 0 };
            inputsWithReadData[8] = new double[] { 0, 0, 0, 0, 0, 0, 1, 1 };
            inputsWithReadData[9] = new double[] { 0, 0, 0, 0, 0, 1, 0, 0 };

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

            double[][] outputsWithHeads = new double[10][];
            //                                 D0, D1, D2  ,ReadHead -> K1, K2, K3, B, G, S1, S2, S3, H     ,WriteHead -> K1, K2, K3, B, G, S1, S2, S3, H, E1, E2, E3, A1, A2, A3
            outputsWithHeads[0] = new double[] { 0, 0, 0,                0,  0,  0, 1, 1,  0,  1,  0, 1,                   0,  0,  0, 1, 1,  0,  1,  0, 1,  1,  1,  1,  0,  0,  0 };
            outputsWithHeads[1] = new double[] { 0, 0, 0,                0,  0,  0, 1, 0,  0,  1,  0, 1,                   0,  0,  0, 1, 0,  1,  0,  0, 1,  1,  1,  1,  0,  0,  1 };
            outputsWithHeads[2] = new double[] { 0, 0, 0,                0,  0,  0, 1, 0,  0,  1,  0, 1,                   0,  0,  0, 1, 0,  1,  0,  0, 1,  1,  1,  1,  0,  1,  0 };
            outputsWithHeads[3] = new double[] { 0, 0, 0,                0,  0,  0, 1, 0,  0,  1,  0, 1,                   0,  0,  0, 1, 0,  1,  0,  0, 1,  1,  1,  1,  0,  1,  1 };
            outputsWithHeads[4] = new double[] { 0, 0, 0,                0,  0,  0, 1, 0,  0,  1,  0, 1,                   0,  0,  0, 1, 0,  1,  0,  0, 1,  1,  1,  1,  1,  0,  0 };
            outputsWithHeads[5] = new double[] { 0, 0, 0,                0,  0,  0, 1, 0,  1,  0,  0, 1,                   0,  0,  0, 1, 0,  0,  1,  0, 1,  0,  0,  0,  0,  0,  0 };
            outputsWithHeads[6] = new double[] { 0, 0, 1,                0,  0,  0, 1, 0,  1,  0,  0, 1,                   0,  0,  0, 1, 0,  0,  1,  0, 1,  0,  0,  0,  0,  0,  0 };
            outputsWithHeads[7] = new double[] { 0, 1, 0,                0,  0,  0, 1, 0,  1,  0,  0, 1,                   0,  0,  0, 1, 0,  0,  1,  0, 1,  0,  0,  0,  0,  0,  0 };
            outputsWithHeads[8] = new double[] { 0, 1, 1,                0,  0,  0, 1, 0,  1,  0,  0, 1,                   0,  0,  0, 1, 0,  0,  1,  0, 1,  0,  0,  0,  0,  0,  0 };
            outputsWithHeads[9] = new double[] { 1, 0, 0,                0,  0,  0, 1, 0,  1,  0,  0, 1,                   0,  0,  0, 1, 0,  0,  1,  0, 1,  0,  0,  0,  0,  0,  0 };


            Console.Title = "Neural turing machine - COPY";
            Console.WindowHeight = 58;
            Console.BufferHeight = 58;
            Console.WindowWidth = 170;
            Console.BufferWidth = 170;


            //MockController mockController = new MockController(inputCount, outputsWithHeads);
            //neuralTuringMachine.SetController(mockController);
            //PerfMeter.CalculateError(neuralTuringMachine, inputs, outputs);

            int i = 0;
            int iterations = 1;
            bool stop = false;
            while(!stop)
            {
                //GenerateInputAndOutput(inputs, outputs);
                for (int j = 0; j < iterations; j++)
                {
                    //teacher.Run(inputs, outputs);
                    teacherWitKnownMemory.Run(inputsWithReadData, outputsWithHeads);
                    i++;
                }
                PerfMeter.CalculateError(neuralTuringMachine, inputs, outputs);
                Console.WriteLine("Iteration " + i);
                Console.WriteLine("Press ESC to end, N for next iteration, B for 10 iterations, V for 100 iterations");

                while (true)
                {
                    ConsoleKeyInfo consoleKeyInfo = Console.ReadKey();
                    ConsoleKey consoleKey = consoleKeyInfo.Key;
                    if (consoleKey == ConsoleKey.Escape)
                    {
                        stop = true;
                        break;
                    }
                    if (consoleKey == ConsoleKey.N)
                    {
                        iterations = 1;
                        break;
                    }
                    if (consoleKey == ConsoleKey.B)
                    {
                        iterations = 10;
                        break;
                    }
                    if (consoleKey == ConsoleKey.V)
                    {
                        iterations = 100;
                        break;
                    }
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
