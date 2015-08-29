using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using NTM2;
using NTM2.Learning;
using NTM2.Memory.Addressing;
using YoVisionClient;
using YoVisionCore;
using YoVisionCore.DataTypes;

namespace CopyTaskTest
{
    class Program
    {
        static void Main()
        {
            //DataStream reportStream = null;
            //try
            //{
            //    YoVisionClientHelper yoVisionClientHelper = new YoVisionClientHelper();
            //    yoVisionClientHelper.Connect(EndpointType.NetTcp, 8081, "localhost", "YoVisionServer");
            //    reportStream = yoVisionClientHelper.RegisterDataStream("Copy task training",
            //        new Int32DataType("Iteration"),
            //        new DoubleDataType("Average data loss"),
            //        new Int32DataType("Training time"),
            //        new Int32DataType("Sequence length"),
            //        new Double2DArrayType("Input"),
            //        new Double2DArrayType("Known output"),
            //        new Double2DArrayType("Real output"),
            //        new Double2DArrayType("Head addressings"));
            //}
            //catch (Exception ex)
            //{
            //    Console.WriteLine(ex.Message);
            //}

            //StandardCopyTask(reportStream);
            MultipleSimultaniousCopyTasks();
            //MultipleSimultaniousAvgCopyTasks();
        }

        private static void MultipleSimultaniousCopyTasks()
        {
            const int numberOfThreads = 2;
            const int numberOfParallelTasks = 32;
            bool end = false;

            List<LearningTask> tasks = new List<LearningTask>();

            BlockingCollection<Tuple<Action<int>, int>> work = new BlockingCollection<Tuple<Action<int>, int>>();
            Thread[] threads = new Thread[numberOfThreads];

            SemaphoreSlim[] semaphores = new SemaphoreSlim[numberOfParallelTasks];

            for (int i = 0; i < numberOfParallelTasks; i++)
            {
                semaphores[i] = new SemaphoreSlim(0);
            }

            for (int i = 0; i < numberOfThreads; i++)
            {
                threads[i] = new Thread(
                    () =>
                    {
                        while (!end)
                        {
                            var action = work.Take();
                            action.Item1(action.Item2);
                            semaphores[action.Item2].Release();
                        }
                    });
                threads[i].Start();
            }

            int weightsCount = 0;
            for (int i = 0; i < numberOfParallelTasks; i++)
            {
                NeuralTuringMachine machine = GetRandomMachine(out weightsCount);
                RMSPropWeightUpdater updater = new RMSPropWeightUpdater(weightsCount, 0.95, 0.5, 0.001);
                tasks.Add(new LearningTask(machine, updater, i));
            }

            int k = 1;
            double bestLongTermError = double.MaxValue;

            while (!end)
            {
                Stopwatch stopwatch = new Stopwatch();
                stopwatch.Start();
                for (int i = 0; i < tasks.Count; i++)
                {
                    work.Add(new Tuple<Action<int>, int>(
                        id =>
                        {
                            tasks[id].Run();
                        }, i));
                }

                for (int i = 0; i < tasks.Count; i++)
                {
                    semaphores[i].Wait();
                }

                Console.WriteLine("Best copy");
                Console.WriteLine("Iteration: {0}", k);

                double[] longAverages = tasks.Select(task => task.GetLongTermErrorAverage()).ToArray();
                Console.WriteLine("Long averages");
                WriteSorted(tasks);

                var min = longAverages.Min();
                Console.WriteLine("Long term average min < best long term error: {0}", min < bestLongTermError);

                if (min < bestLongTermError)
                {
                    double maxError = double.MinValue;
                    double minError = double.MaxValue;
                    int maxIndex = 0;
                    int minIndex = 0;

                    for (int i = 0; i < longAverages.Length; i++)
                    {
                        if (longAverages[i] > maxError)
                        {
                            maxError = longAverages[i];
                            maxIndex = i;
                        }
                        if (longAverages[i] < minError)
                        {
                            minError = longAverages[i];
                            minIndex = i;
                        }
                    }

                    if (minIndex == maxIndex)
                    {
                        break;
                    }

                    Console.WriteLine("Copying {0} to {1}", tasks[minIndex].ID, tasks[maxIndex].ID);

                    tasks[maxIndex].CopyFrom(tasks[minIndex], weightsCount);

                    ResetPriorities(tasks);

                    bestLongTermError = min;
                }
                Console.WriteLine("Remaining tasks count: {0}", tasks.Count);
                Console.WriteLine("Minimum long term error: {0}", min);
                Console.WriteLine("Best long term error: {0}", bestLongTermError);
                k++;
                stopwatch.Stop();
                double seconds = stopwatch.ElapsedMilliseconds / (double)1000;
                Console.WriteLine("Time: {0}[s] per task: {1}[s]", seconds, seconds / tasks.Count);
            }
        }

        private static void WriteSorted(List<LearningTask> tasks)
        {
            double[] currentErrors = tasks.Select(task => task.GetCurrentError()).ToArray();
            List<double> avgs = new List<double>(currentErrors);
            avgs.Sort();
            for (int i = 0; i < tasks.Count; i++)
            {
                for (int q = 0; q < tasks.Count; q++)
                {
                    // ReSharper disable once CompareOfFloatsByEqualityOperator
                    if (currentErrors[q] == avgs[i])
                    {
                        Console.WriteLine("Average: {0}, id: {1}", avgs[i], tasks[q].ID);
                    }
                }
            }
        }

        private static void ResetPriorities(List<LearningTask> tasks)
        {
            double[] errors = tasks.Select(task => 1 / task.GetLongTermErrorAverage()).ToArray();
            double min = errors.Min();
            for (int i = 0; i < tasks.Count; i++)
            {
                errors[i] -= min;
            }
            double max = errors.Max();
            for (int i = 0; i < tasks.Count; i++)
            {
                errors[i] /= max;
            }
            double sum = errors.Sum();
            double normalizator = 100 / sum;
            for (int i = 0; i < tasks.Count; i++)
            {
                errors[i] *= normalizator;
            }
            for (int i = 0; i < tasks.Count; i++)
            {
                tasks[i].Priority = (int)errors[i];
            }
            tasks.RemoveAll(task => task.Priority == 0);
        }

        private static void MultipleSimultaniousAvgCopyTasks()
        {
            const int numberOfThreads = 1;
            const int numberOfParallelTasks = 16;
            bool end = false;
            BlockingCollection<Tuple<Action<int>, int>> work = new BlockingCollection<Tuple<Action<int>, int>>();
            Thread[] threads = new Thread[numberOfThreads];

            SemaphoreSlim[] semaphores = new SemaphoreSlim[numberOfParallelTasks];

            for (int i = 0; i < numberOfParallelTasks; i++)
            {
                semaphores[i] = new SemaphoreSlim(0);
            }

            for (int i = 0; i < numberOfThreads; i++)
            {
                threads[i] = new Thread(
                    () =>
                    {
                        while (!end)
                        {
                            var action = work.Take();
                            action.Item1(action.Item2);
                            semaphores[action.Item2].Release();
                        }
                    });
                threads[i].Start();
            }

            double[][] errorss = new double[numberOfParallelTasks][];
            long[][] timess = new long[numberOfParallelTasks][];
            NeuralTuringMachine[] machines = new NeuralTuringMachine[numberOfParallelTasks];
            BPTTTeacher[] teachers = new BPTTTeacher[numberOfParallelTasks];

            int weightsCount = 0;
            for (int i = 0; i < numberOfParallelTasks; i++)
            {
                errorss[i] = new double[100];
                timess[i] = new long[100];
                for (int j = 0; j < 100; j++)
                {
                    errorss[i][j] = 1;
                }
                machines[i] = GetRandomMachine(out weightsCount);
                teachers[i] = GetTeacher(weightsCount, machines[i]);
            }

            int k = 1;
            while (!end)
            {
                Stopwatch stopwatch = new Stopwatch();
                stopwatch.Start();
                for (int i = 0; i < numberOfParallelTasks; i++)
                {
                    var index = i;
                    work.Add(new Tuple<Action<int>, int>(id => Iterate(teachers[index], errorss[index], timess[index], id), index));
                }

                for (int i = 0; i < numberOfParallelTasks; i++)
                {
                    semaphores[i].Wait();
                }

                Console.WriteLine("Average NTMs");

                double[] errors = errorss.Select(doubles => doubles.Average()).ToArray();
                AverageMachineWeightUpdater averageWeightUpdater = new AverageMachineWeightUpdater(weightsCount, machines);

                foreach (NeuralTuringMachine machine in machines)
                {
                    machine.UpdateWeights(averageWeightUpdater);
                    averageWeightUpdater.Reset();
                }

                for (int i = 0; i < numberOfParallelTasks; i++)
                {
                    teachers[i] = GetTeacher(weightsCount, machines[i]);
                }

                Console.WriteLine("Iteration: {0}", k);
                Console.WriteLine("Average error: {0}", errors.Average());
                Console.WriteLine("Best error: {0}", errors.Min());
                k++;

                stopwatch.Stop();
                double seconds = stopwatch.ElapsedMilliseconds / (double)1000;
                Console.WriteLine("Time: {0}[s] per task: {1}[s]", seconds, seconds / numberOfParallelTasks);
            }
        }

        private static void Iterate(BPTTTeacher teacher, double[] errors, long[] times, int id)
        {
            const int vectorSize = 8;
            const int minSeqLen = 1;
            const int maxSeqLen = 20;
            //double savingThreshold = 0.0005;
            Random rand = new Random(DateTime.Now.Millisecond);

            for (int i = 1; i <= 100; i++)
            {
                var sequence = SequenceGenerator.GenerateSequence(rand.Next(minSeqLen, maxSeqLen), vectorSize);

                Stopwatch stopwatch = new Stopwatch();
                stopwatch.Start();
                double[][] machinesOutput = teacher.Train(sequence.Item1, sequence.Item2);
                stopwatch.Stop();
                times[i % 100] = stopwatch.ElapsedMilliseconds;

                double error = CalculateLoss(sequence.Item2, machinesOutput);

                errors[i % 100] = error;

                if (i % 100 == 0)
                {
                    double averageError2 = errors.Average();
                    Console.WriteLine("Iteration: {0}, error: {1}, id: {2}", i, averageError2, id);

                    //if (averageError2 < savingThreshold)
                    //{
                    //    savingThreshold /= 2;
                    //    machine.Save("NTM_" + averageError2 + "_" + DateTime.Now.ToString("s").Replace(":", ""));
                    //    maxSeqLen++;
                    //    minSeqLen++;
                    //}
                }
                //if (i % 100000 == 0)
                //{
                //    machine.Save("NTM_" + i + DateTime.Now.ToString("s").Replace(":", ""));
                //}
            }
        }

        private static BPTTTeacher GetTeacher(int weightsCount, NeuralTuringMachine machine)
        {
            RMSPropWeightUpdater rmsPropWeightUpdater = new RMSPropWeightUpdater(weightsCount, 0.95, 0.5, 0.001, 0.001);

            BPTTTeacher teacher = new BPTTTeacher(machine, rmsPropWeightUpdater);
            return teacher;
        }

        private static NeuralTuringMachine GetRandomMachine(out int weightsCount)
        {
            Random rand = new Random(DateTime.Now.Millisecond);

            const int vectorSize = 8;
            const int controllerSize = 100;
            const int headsCount = 1;
            const int memoryN = 128;
            const int memoryM = 20;
            const int inputSize = vectorSize + 2;
            const int outputSize = vectorSize;

            int headUnitSize = Head.GetUnitSize(memoryM);

            weightsCount = (headsCount * memoryN) +
                           (memoryN * memoryM) +
                           (controllerSize * headsCount * memoryM) +
                           (controllerSize * inputSize) +
                           (controllerSize) +
                           (outputSize * (controllerSize + 1)) +
                           (headsCount * headUnitSize * (controllerSize + 1));

            //TODO remove rand
            NeuralTuringMachine machine = new NeuralTuringMachine(vectorSize + 2, vectorSize, controllerSize, headsCount, memoryN, memoryM, new RandomWeightInitializer(rand));
            return machine;
        }

        private static void StandardCopyTask(DataStream reportStream)
        {
            double[] errors = new double[100];
            long[] times = new long[100];
            for (int i = 0; i < 100; i++)
            {
                errors[i] = 1;
            }

            const int seed = 32702;
            Console.WriteLine(seed);
            //TODO args parsing shit
            Random rand = new Random(seed);

            const int vectorSize = 8;
            const int controllerSize = 100;
            const int headsCount = 1;
            const int memoryN = 128;
            const int memoryM = 20;
            const int inputSize = vectorSize + 2;
            const int outputSize = vectorSize;

            //TODO remove rand
            NeuralTuringMachine machine = new NeuralTuringMachine(vectorSize + 2, vectorSize, controllerSize, headsCount, memoryN, memoryM, new RandomWeightInitializer(rand));

            //TODO extract weight count calculation
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

            //NeuralTuringMachine machine = NeuralTuringMachine.Load(@"NTM_0.000583637804331003_2015-04-18T223455");

            BPTTTeacher teacher = new BPTTTeacher(machine, rmsPropWeightUpdater);


            //for (int i = 1; i < 256; i++)
            //{
            //    var sequence = SequenceGenerator.GenerateSequence(i, vectorSize);
            //    double[][] machineOutput = teacher.Train(sequence.Item1, sequence.Item2);
            //    double error = CalculateLoss(sequence.Item2, machineOutput);
            //    Console.WriteLine("{0},{1}", i, error);
            //}

            int minSeqLen = 200;
            int maxSeqLen = 200;
            double savingThreshold = 0.0005;
            for (int i = 1; i < 10000000; i++)
            {
                //var sequence = SequenceGenerator.GenerateSequence(rand.Next(20) + 1, vectorSize);
                var sequence = SequenceGenerator.GenerateSequence(rand.Next(minSeqLen, maxSeqLen), vectorSize);

                Stopwatch stopwatch = new Stopwatch();
                stopwatch.Start();
                double[][] headAddressings;
                double[][] machinesOutput = teacher.TrainVerbose(sequence.Item1, sequence.Item2, out headAddressings);
                stopwatch.Stop();
                times[i % 100] = stopwatch.ElapsedMilliseconds;

                double error = CalculateLoss(sequence.Item2, machinesOutput);
                double averageError = error / (sequence.Item2.Length * sequence.Item2[0].Length);

                errors[i % 100] = error;

                if (reportStream != null)
                {
                    reportStream.Set("Iteration", i);
                    reportStream.Set("Average data loss", averageError);
                    reportStream.Set("Training time", stopwatch.ElapsedMilliseconds);
                    reportStream.Set("Sequence length", (sequence.Item1.Length - 2) / 2);
                    reportStream.Set("Input", sequence.Item1);
                    reportStream.Set("Known output", sequence.Item2);
                    reportStream.Set("Real output", machinesOutput);
                    reportStream.Set("Head addressings", headAddressings);
                    reportStream.SendData();
                }

                if (i % 100 == 0)
                {
                    double averageError2 = errors.Average();
                    Console.WriteLine(
                        "Iteration: {0}, error: {1}, iterations per second: {2:0.0} MinSeqLen: {3} MaxSeqLen: {4}", i,
                        averageError2, 1000 / times.Average(), minSeqLen, maxSeqLen);

                    if (averageError2 < savingThreshold)
                    {
                        savingThreshold /= 2;
                        machine.Save("NTM_" + averageError2 + "_" + DateTime.Now.ToString("s").Replace(":", ""));
                        maxSeqLen++;
                        minSeqLen++;
                    }
                }
                if (i % 100000 == 0)
                {
                    machine.Save("NTM_" + i + DateTime.Now.ToString("s").Replace(":", ""));
                }
            }
        }

        private static double CalculateLoss(double[][] knownOutput, double[][] machinesOutput)
        {
            double totalLoss = 0;
            int okt = knownOutput.Length - ((knownOutput.Length - 2) / 2);
            for (int t = 0; t < knownOutput.Length; t++)
            {
                for (int i = 0; i < knownOutput[t].Length; i++)
                {
                    double expected = knownOutput[t][i];
                    double real = machinesOutput[t][i];
                    if (t >= okt)
                    {
                        totalLoss += Math.Pow(expected - real, 2);
                    }
                }
            }
            return Math.Sqrt(totalLoss);
        }
    }
}
