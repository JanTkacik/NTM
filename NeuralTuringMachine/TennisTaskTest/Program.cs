using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using NTM2;
using NTM2.Learning;
using NTM2.Memory.Addressing;
// ReSharper disable CompareOfFloatsByEqualityOperator

namespace TennisTaskTest
{
    class Program
    {
        static void Main(string[] args)
        {
            Random rand = new Random(DateTime.Now.Millisecond);

            int headCount = int.Parse(args[0]);
            int controllerSize = int.Parse(args[1]);
            int memoryN = int.Parse(args[2]);
            int memoryM = int.Parse(args[3]);
            bool parallel = bool.Parse(args[4]);

            List<double[][]> inputs;
            List<double[][]> outputs;
            LoadData(out inputs, out outputs);

            if (!parallel)
            {
                ClassicIterations(controllerSize, headCount, memoryN, memoryM, rand, inputs, outputs);   
            }
            else
            {
                ParallelIterations(controllerSize, headCount, memoryN, memoryM, rand, inputs, outputs);
            }
        }

        private static void ParallelIterations(int controllerSize, int headCount, int memoryN, int memoryM, Random rand, List<double[][]> inputs, List<double[][]> outputs)
        {
            string directoryName = string.Format("{0}_{1}_{2}_{3}", controllerSize, headCount, memoryM, memoryN);

            if (Directory.Exists(directoryName))
            {
                DateTime newestDateTime = new DateTime();
                string newestFileName = null;
                string[] filenames = Directory.GetFiles(directoryName);
                if (filenames.Length > 0)
                {
                    foreach (string fileName in filenames)
                    {
                        DateTime creationTime = File.GetCreationTime(fileName);
                        if (creationTime > newestDateTime)
                        {
                            newestFileName = fileName;
                            newestDateTime = creationTime;
                        }
                    }
                    int weightsCount;
                    NeuralTuringMachine neuralTuringMachine = GetRandomMachine(out weightsCount, controllerSize,
                        headCount, memoryN, memoryM, rand);
                    ParallelTasks.Run(
                        () =>
                            new Tuple<NeuralTuringMachine, int>(NeuralTuringMachine.Load(newestFileName), weightsCount),
                        () => ExampleFactory(inputs, outputs, rand), directoryName);
                    return;
                }
            }
            
            ParallelTasks.Run(() => MachineFactory(headCount, controllerSize, memoryN, memoryM, rand), () => ExampleFactory(inputs, outputs, rand), directoryName);
            
        }

        private static Tuple<double[][], double[][]> ExampleFactory(List<double[][]> inputs, List<double[][]> outputs, Random rand)
        {
            int index = rand.Next(inputs.Count);
            return new Tuple<double[][], double[][]>(inputs[index], outputs[index]);
        }

        private static Tuple<NeuralTuringMachine, int> MachineFactory(int headCount, int controllerSize, int memoryN, int memoryM, Random rand)
        {

            int weightsCount;
            NeuralTuringMachine neuralTuringMachine = GetRandomMachine(out weightsCount, controllerSize, headCount, memoryN, memoryM, rand);
            return new Tuple<NeuralTuringMachine, int>(neuralTuringMachine, weightsCount);
        }

        private static void ClassicIterations(int controllerSize, int headCount, int memoryN, int memoryM, Random rand, List<double[][]> inputs, List<double[][]> outputs)
        {
            int weightsCount;
            var machine = GetRandomMachine(out weightsCount, controllerSize, headCount, memoryN, memoryM, rand);
            RMSPropWeightUpdater rmsPropWeightUpdater = new RMSPropWeightUpdater(weightsCount, 0.95, 0.5, 0.001);
            BPTTTeacher teacher = new BPTTTeacher(machine, rmsPropWeightUpdater);

            double[][] errors = new double[100][];
            long[] times = new long[100];
            for (int i = 0; i < 100; i++)
            {
                errors[i] = new double[4];
                for (int j = 0; j < 4; j++)
                {
                    errors[i][j] = 1;
                }
            }

            int count = inputs.Count;
            for (int i = 1; i < 10000000; i++)
            {
                int index = rand.Next(count);
                double[][] input = inputs[index];
                double[][] output = outputs[index];
                Stopwatch stopwatch = new Stopwatch();
                stopwatch.Start();
                double[][] machinesOutput = teacher.Train(input, output);
                stopwatch.Stop();
                times[i%100] = stopwatch.ElapsedMilliseconds;

                double[] error = CalculateLoss(output, machinesOutput);

                errors[i%100][0] = error[0];
                errors[i%100][1] = error[1];
                errors[i%100][2] = error[2];
                errors[i%100][3] = error[3];

                double averageError = errors.Average(doubles => doubles.Average());

                if (i%100 == 0)
                {
                    WriteExample(i, times, errors, averageError, output, machinesOutput);
                }
                if (i%2500 == 0)
                {
                    string directoryName = string.Format("{0}_{1}_{2}_{3}", controllerSize, headCount, memoryM, memoryN);
                    if (!Directory.Exists(directoryName))
                    {
                        Directory.CreateDirectory(directoryName);
                    }
                    string filename = string.Format("NTM_{0}_{1}_{2}.ntm", i, DateTime.Now.ToString("s").Replace(":", ""),
                        averageError);
                    machine.Save(Path.Combine(directoryName, filename));
                }
            }
        }

        private static void WriteExample(int i, long[] times, double[][] errors, double averageError, double[][] output, double[][] machinesOutput)
        {
            Console.WriteLine(
                "Iteration: {0}, iterations per second: {1:0.0} Match error: {2} Sets error: {3} Games error: {4} Points error: {5} Average error: {6}",
                i,
                1000/times.Average(),
                errors[0].Average(),
                errors[1].Average(),
                errors[2].Average(),
                errors[3].Average(),
                averageError);
            Console.WriteLine("Last example:");
            Console.WriteLine("Match:");
            WriteSequence(output, 0);
            WriteSequence(machinesOutput, 0);

            Console.WriteLine("Set:");
            WriteSequence(output, 1);
            WriteSequence(machinesOutput, 1);

            Console.WriteLine("Game:");
            WriteSequence(output, 2);
            WriteSequence(machinesOutput, 2);

            Console.WriteLine("Point:");
            WriteSequence(output, 3);
            WriteSequence(machinesOutput, 3);
        }

        private static void WriteSequence(double[][] output, int index)
        {
            foreach (double[] known in output)
            {
                Console.Write("{0:0.00} ", known[index]);
            }
            Console.WriteLine();
        }

        private static NeuralTuringMachine GetRandomMachine(out int weightsCount, int controllerSize, int headsCount, int memoryN, int memoryM, Random rand)
        {
            const int inputSize = 8;
            const int outputSize = 1;

            int headUnitSize = Head.GetUnitSize(memoryM);

            weightsCount = (headsCount*memoryN) +
                           (memoryN*memoryM) +
                           (controllerSize*headsCount*memoryM) +
                           (controllerSize*inputSize) +
                           (controllerSize) +
                           (outputSize*(controllerSize + 1)) +
                           (headsCount*headUnitSize*(controllerSize + 1));

            NeuralTuringMachine machine = new NeuralTuringMachine(inputSize, outputSize, controllerSize, headsCount, memoryN,
                memoryM, new RandomWeightInitializer(rand));
            return machine;
        }

        private static double[] CalculateLoss(double[][] output, double[][] machinesOutput)
        {
            double matchError = 0;
            double setsError = 0;
            double gamesError = 0;
            double pointsError = 0;
            for (int i = 0; i < output.Length; i++)
            {
                double[] known = output[i];
                double[] computed = machinesOutput[i];
                matchError += Math.Abs(known[0] - computed[0]);
                setsError += Math.Abs(known[1] - computed[1]);
                gamesError += Math.Abs(known[2] - computed[2]);
                pointsError += Math.Abs(known[3] - computed[3]);
            }

            return new[] { matchError / output.Length, setsError / output.Length, gamesError / output.Length, pointsError / output.Length };
        }

        [SuppressMessage("ReSharper", "PossibleNullReferenceException")]
        private static void LoadData(out List<double[][]> inputs, out List<double[][]> outputs)
        {
            string[] lines = File.ReadAllLines(@"data1.txt");
            inputs = new List<double[][]>();
            outputs = new List<double[][]>();
            List<double[]> currentMatchInput = null;
            List<double[]> currentMatchOutput = null;
            List<double[]> currentMatchInvInput = null;
            List<double[]> currentMatchInvOutput = null;

            for (int i = 1; i < lines.Length; i++)
            {
                string line = lines[i];
                string[] data = line.Split(',');
                double serve = double.Parse(data[0]);
                double lastPoint = double.Parse(data[1]);
                double hsets = double.Parse(data[2]);
                double asets = double.Parse(data[3]);
                double hgames = double.Parse(data[4]);
                double agames = double.Parse(data[5]);
                double hpoints = double.Parse(data[6]);
                double apoints = double.Parse(data[7]);
                double cgame = double.Parse(data[10]);
                //double cpoint = double.Parse(data[11]);

                if (lastPoint == 0)
                {
                    if (currentMatchInput != null)
                    {
                        inputs.Add(currentMatchInput.ToArray());
                        inputs.Add(currentMatchInvInput.ToArray());
                    }
                    if (currentMatchOutput != null)
                    {
                        outputs.Add(currentMatchOutput.ToArray());
                        outputs.Add(currentMatchInvOutput.ToArray());
                    }
                    currentMatchInput = new List<double[]>();
                    currentMatchOutput = new List<double[]>();
                    currentMatchInvInput = new List<double[]>();
                    currentMatchInvOutput = new List<double[]>();
                }

                currentMatchInput.Add(new[] {serve, lastPoint, hsets, asets, hgames, agames, hpoints, apoints});
                currentMatchInvInput.Add(new [] {-serve, -lastPoint, asets, hsets, agames, hgames, apoints, hpoints});
                //currentMatchOutput.Add(new[] {cmatch, cset, cgame, cpoint});
                //currentMatchOutput.Add(new[] {cpoint});
                currentMatchOutput.Add(new [] {cgame});
                //currentMatchInvOutput.Add(new double[] {cmatch == 1 ? 0 : 1, cset == 1 ? 0 : 1, cgame == 1 ? 0 : 1, cpoint == 1 ? 0 : 1});
                //currentMatchInvOutput.Add(new double[] {cpoint == 1 ? 0 : 1});
                currentMatchInvOutput.Add(new double[] {cgame == 1 ? 0 : 1});
            }

            inputs.Add(currentMatchInput.ToArray());
            inputs.Add(currentMatchInvInput.ToArray());
            outputs.Add(currentMatchOutput.ToArray());
            outputs.Add(currentMatchInvOutput.ToArray());
        }
    }
}
