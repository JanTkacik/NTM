using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using NTM2;
using NTM2.Learning;

namespace TennisTaskTest
{
    static class ParallelTasks
    {
        public static bool End = false;

        public static void Run(
            Func<Tuple<NeuralTuringMachine, int>> machineFactory,
            Func<Tuple<double[][], double[][]>> exampleFactory,
            string directoryName)
        {
            const int numberOfThreads = 8;
            const int numberOfParallelTasks = 32;

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
                        while (!End)
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
                Tuple<NeuralTuringMachine, int> factory = machineFactory();
                NeuralTuringMachine machine = factory.Item1;
                weightsCount = factory.Item2;
                RMSPropWeightUpdater updater = new RMSPropWeightUpdater(weightsCount, 0.95, 0.5, 0.001);
                tasks.Add(new LearningTask(machine, updater, exampleFactory, directoryName, i));
            }

            int k = 1;
            double bestLongTermError = double.MaxValue;

            while (!End)
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
                Console.WriteLine("Current averages");
                WriteCurrentSorted(tasks);

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
                    
                    bestLongTermError = min;
                }
                ResetPriorities(tasks);
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
            double[] currentErrors = tasks.Select(task => task.GetLongTermErrorAverage()).ToArray();
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

        private static void WriteCurrentSorted(List<LearningTask> tasks)
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
            double max = errors.Max();

            // ReSharper disable once CompareOfFloatsByEqualityOperator
            if (min != max)
            {
                for (int i = 0; i < tasks.Count; i++)
                {
                    errors[i] -= min;
                }
                for (int i = 0; i < tasks.Count; i++)
                {
                    errors[i] /= max;
                }
                double sum = errors.Sum();
                double normalizator = 100/sum;
                for (int i = 0; i < tasks.Count; i++)
                {
                    errors[i] *= normalizator;
                }
                for (int i = 0; i < tasks.Count; i++)
                {
                    tasks[i].Priority = (int) errors[i];
                }
            }
            else
            {
                foreach (LearningTask t in tasks)
                {
                    t.Priority = 100 / tasks.Count;
                }
            }
            if (tasks.Count > 8)
            {
                while (tasks.Count > 8)
                {
                    int findIndex = tasks.FindIndex(task => task.Priority == 0);
                    if (findIndex == -1)
                    {
                        break;
                    }
                    tasks.RemoveAt(findIndex);
                }
            }
            else
            {
                foreach (LearningTask task in tasks)
                {
                    if (task.Priority == 0)
                    {
                        task.Priority = 1;
                    }
                }
            }
        }
    }
}
