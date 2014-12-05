using System;
using NeuralTuringMachine;
using NeuralTuringMachine.Learning;
using NeuralTuringMachine.Memory.Head;
using NeuralTuringMachine.Performance;

namespace NTMConsoleTestClient
{
    class Program
    {
        static void Main()
        {
            const int inputCount = 3;
            const int outputCount = 1;
            const int readHeadCount = 1;
            const int writeHeadCount = 1;
            const int hiddenNeuronCount = 99;
            const int hiddenLayersCount = 3;
            const int memoryCellsCount = 10;
            const int memoryVectorLength = 1;
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

            NTMFactory factory = new NTMFactory();
            NTM neuralTuringMachine = factory.CreateNTM(inputCount, outputCount, hiddenNeuronCount, hiddenLayersCount, new MemorySettings(memoryCellsCount, memoryVectorLength, maxConvolutionalShift, readHeadCount, writeHeadCount));
            
            //FileStream fileStream = File.OpenRead("BestControllerA");
            //ActivationNetwork bestController = (ActivationNetwork)Network.Load(fileStream);
            //neuralTuringMachine.SetController(bestController);
            //fileStream.Close();

            BpttTeacher teacher = new BpttTeacher(factory, neuralTuringMachine);
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
                    GenerateInputAndOutputWithHeads(out inputsWithReadData,out outputsWithHeads,out inputs,out outputs, 4, 1);
                    teacherWitKnownMemory.Run(inputsWithReadData, outputsWithHeads);
                    i++;
                }

                //for (int j = 0; j < iterations; j++)
                //{
                //    GenerateInputAndOutput(inputs, outputs);
                //    teacher.Run(inputs, outputs);
                //    //PerfMeter.CalculateError(neuralTuringMachine, inputs, outputs);
                //    i++;
                //}
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

        private static void GenerateInputAndOutputWithHeads(out double[][] input, out double[][] output, out double[][] inputWithoutHeads, out double[][] outputWithoutHeads, int length, int inputCount)
        {
            Random random = new Random(DateTime.Now.Millisecond);
            if (length < 1)
            {
                throw new ArgumentOutOfRangeException("length", length, "Length must be larger than 1");
            }

            int usedLength = (length*2) + 2; 

            int inputVectorLength = (2*inputCount) + 2;
            int outputVecotrLength = 12 + (5 * inputCount);

            input = new double[usedLength][];
            output = new double[usedLength][];
            inputWithoutHeads = new double[usedLength][];
            outputWithoutHeads = new double[usedLength][];

            for (int i = 0; i < usedLength; i++)
            {
                input[i] = new double[inputVectorLength];
                output[i] = new double[outputVecotrLength];
                inputWithoutHeads[i] = new double[2 + inputCount];
                outputWithoutHeads[i] = new double[inputCount];
            }
            
            double[][] randomNumbers = new double[length][];
            for (int i = 0; i < length; i++)
            {
                randomNumbers[i] = new double[inputCount];
                for (int j = 0; j < inputCount; j++)
                {
                    double randomNumber = random.NextDouble();
                    if (randomNumber > 0.5)
                    {
                        randomNumbers[i][j] = 1;
                        inputWithoutHeads[i + 1][j + 2] = 1;
                        outputWithoutHeads[length + 2 + i][j] = 1;
                    }
                }
            }

            inputWithoutHeads[0][0] = 1;
            inputWithoutHeads[1 + length][1] = 1;


            //START
            input[0][0] = 1; //Start flag

            output[0][2*inputCount] = 1; //Read head beta
            output[0][(2*inputCount) + 1] = 1; //Read head gate
            output[0][(2*inputCount) + 3] = 1; //Read head Convolution 0
            output[0][(2*inputCount) + 5] = 1; //Read head gama 

            output[0][(3*inputCount) + 6] = 1; //Write head beta
            output[0][(3*inputCount) + 7] = 1; //Write head gate
            output[0][(3*inputCount) + 9] = 1; //Write head Convolution 0
            output[0][(3*inputCount) + 11] = 1; //Write head gama 

            //Erase vector
            for (int i = 0; i < inputCount; i++)
            {
                output[0][(3*inputCount) + i + 12] = 1; 
            }

            //COPY INPUT
            for (int i = 0; i < length; i++)
            {
                //INPUT VECTOR
                for (int j = 0; j < inputCount; j++)
                {
                    input[i + 1][j + 2] = randomNumbers[i][j];
                }

                //OUTPUT VECTOR
                output[i + 1][2 * inputCount] = 1; //Read head beta
                output[i + 1][(2 * inputCount) + 3] = 1; //Read head Convolution 0
                output[i + 1][(2 * inputCount) + 5] = 1; //Read head gama 

                output[i + 1][(3 * inputCount) + 6] = 1; //Write head beta
                output[i + 1][(3 * inputCount) + 8] = 1; //Write head Convolution +1
                output[i + 1][(3 * inputCount) + 11] = 1; //Write head gama 

                //Erase vector
                for (int j = 0; j < inputCount; j++)
                {
                    output[i + 1][(3 * inputCount) + j + 12] = 1;
                }
                //Add vector
                for (int j = 0; j < inputCount; j++)
                {
                    output[i + 1][(4 * inputCount) + j + 12] = randomNumbers[i][j];
                }
            }

            //COPY START
            input[length + 1][1] = 1; //Copy flag

            output[length + 1][2 * inputCount] = 1; //Read head beta
            output[length + 1][(2 * inputCount) + 2] = 1; //Read head Convolution +1
            output[length + 1][(2 * inputCount) + 5] = 1; //Read head gama 

            output[length + 1][(3 * inputCount) + 6] = 1; //Write head beta
            output[length + 1][(3 * inputCount) + 9] = 1; //Write head Convolution 0
            output[length + 1][(3 * inputCount) + 11] = 1; //Write head gama 

            //REPLAY
            for (int i = 0; i < length; i++)
            {
                //INPUT VECTOR
                for (int j = 0; j < inputCount; j++)
                {
                    input[i + 2 + length][j + 2 + inputCount] = randomNumbers[i][j];
                }

                //OUTPUT VECTOR
                for (int j = 0; j < inputCount; j++)
                {
                    output[i + 2 + length][j] = randomNumbers[i][j];
                }

                output[i + 2 + length][2 * inputCount] = 1; //Read head beta
                output[i + 2 + length][(2 * inputCount) + 2] = 1; //Read head Convolution +1
                output[i + 2 + length][(2 * inputCount) + 5] = 1; //Read head gama 

                output[i + 2 + length][(3 * inputCount) + 6] = 1; //Write head beta
                output[i + 2 + length][(3 * inputCount) + 9] = 1; //Write head Convolution 0
                output[i + 2 + length][(3 * inputCount) + 11] = 1; //Write head gama 
            }
        }
    }
}
