using System;
using System.Diagnostics;
using System.Linq;
using NTM2;
using NTM2.Learning;
using NTM2.Memory.Addressing;
using YoVisionClient;
using YoVisionCore;
using YoVisionCore.DataTypes;

namespace NGramTest
{
    class Program
    {
        static void Main()
        {
            DataStream reportStream = null;
            try
            {
                YoVisionClientHelper yoVisionClientHelper = new YoVisionClientHelper();
                yoVisionClientHelper.Connect(EndpointType.NetTcp, 8081, "localhost", "YoVisionServer");
                reportStream = yoVisionClientHelper.RegisterDataStream("NGram task training",
                    new Int32DataType("Iteration"),
                    new DoubleDataType("Average data loss"),
                    new Double2DArrayType("Input"),
                    new Double2DArrayType("Known output"),
                    new Double2DArrayType("Real output"),
                    new Double2DArrayType("Head addressings"));
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }

            const int controllerSize = 100;
            const int headsCount = 1;
            const int memoryN = 128;
            const int memoryM = 20;
            const int inputSize = 1;
            const int outputSize = 1;

            Random rand = new Random(42);
            NeuralTuringMachine machine = new NeuralTuringMachine(inputSize, outputSize, controllerSize, headsCount, memoryN, memoryM, new RandomWeightInitializer(rand));

            int headUnitSize = Head.GetUnitSize(memoryM);

             var weightsCount =
                (headsCount * memoryN) +
                (memoryN * memoryM) +
                (controllerSize * headsCount * memoryM) +
                (controllerSize * inputSize) +
                (controllerSize) +
                (outputSize * (controllerSize + 1)) +
                (headsCount * headUnitSize * (controllerSize + 1));

            Console.WriteLine(weightsCount);

            RMSPropWeightUpdater rmsPropWeightUpdater = new RMSPropWeightUpdater(weightsCount, 0.95, 0.5, 0.001, 0.001);

            BPTTTeacher teacher = new BPTTTeacher(machine, rmsPropWeightUpdater);

            long[] times = new long[100];

            for (int i = 1; i < 10000000; i++)
            {
                Tuple<double[][], double[][]> data = SequenceGenerator.GenerateSequence(SequenceGenerator.GeneratePropabilities());

                Stopwatch stopwatch = new Stopwatch();
                stopwatch.Start();
                double[][] headAddressings;
                double[][] output = teacher.TrainVerbose(data.Item1, data.Item2, out headAddressings);
                stopwatch.Stop();
                times[i % 100] = stopwatch.ElapsedMilliseconds;
                
                if (i%10 == 0)
                {
                    double loss = CalculateLogLoss(output, data.Item2);
                    if (reportStream != null)
                    {
                        reportStream.Set("Iteration", i);
                        reportStream.Set("Average data loss", loss);
                        reportStream.Set("Input", data.Item1);
                        reportStream.Set("Known output", data.Item2);
                        reportStream.Set("Real output", output);
                        reportStream.Set("Head addressings", headAddressings);
                        reportStream.SendData();
                    }
                }

                if (i%100 == 0)
                {
                    Console.WriteLine("Iteration: {0}, iterations per second: {1:0.0}", i, 1000 / times.Average());
                }


                if (i%1000 == 0)
                {
                    double[] props = SequenceGenerator.GeneratePropabilities();
                    
                    const int sampleCount = 100;

                    double[] losses = new double[sampleCount];

                    for (int j = 0; j < sampleCount; j++)
                    {
                        Tuple<double[][], double[][]> sequence = SequenceGenerator.GenerateSequence(props);
                        var machineOutput = teacher.Train(sequence.Item1, sequence.Item2);
                        double[][] knownOutput = sequence.Item2;
                        double loss = CalculateLogLoss(machineOutput, knownOutput);
                        losses[j] = -loss;
                    }

                    Console.WriteLine("Loss [bits per sequence]: {0}", losses.Average());
                }

                if (i % 1000 == 0)
                {
                    machine.Save("NTM_" + i + DateTime.Now.ToString("s").Replace(":", ""));
                }
            }
        }

        private static double CalculateLogLoss(double[][] machineOutput, double[][] knownOutput)
        {
            double loss = 0;
            int sequenceLength = machineOutput.Length;
            for (int k = 0; k < sequenceLength; k++)
            {
                double y = knownOutput[k][0];
                double p = machineOutput[k][0];
                loss += (y*Math.Log(p)) + ((1 - y)*Math.Log(1 - p));
            }
            return loss;
        }
    }
}
