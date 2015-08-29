using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NTM2;
using NTM2.Learning;

namespace TennisTaskTest
{
    class LearningTask
    {
        private int _iterations;
        private const int EpochLength = 10;
        private readonly NeuralTuringMachine _machine;
        private RMSPropWeightUpdater _weightUpdater;
        private readonly Func<Tuple<double[][], double[][]>> _exampleGenerator;
        private readonly string _directoryName;
        private readonly int _id;
        private BPTTTeacher _teacher;
        private List<double> _longTermAverageErrors;

        public LearningTask(NeuralTuringMachine machine, RMSPropWeightUpdater weightUpdater, Func<Tuple<double[][], double[][]>> exampleGenerator, string directoryName, int id)
        {
            _iterations = 0;
            _machine = machine;
            _weightUpdater = weightUpdater;
            _exampleGenerator = exampleGenerator;
            _directoryName = directoryName;
            _id = id;
            _teacher = new BPTTTeacher(_machine, weightUpdater);
            _longTermAverageErrors = new List<double>();
            Priority = 100 / 32;
        }

        public int ID
        {
            get
            {
                return _id;
            }
        }

        public int Priority { get; set; }

        public void Run()
        {
            Console.WriteLine("Running task {0} with priority {1}", ID, Priority);
            double[][] lastKnown = null;
            double[][] lastComputed = null;
            double[] lastErrors = null;
            for (int j = 0; j < Priority; j++)
            {
                for (int i = 0; i < EpochLength; i++)
                {
                    var sequence = _exampleGenerator();
                    double[][] machinesOutput = _teacher.Train(sequence.Item1, sequence.Item2);
                    double[] errors = CalculateLoss(sequence.Item2, machinesOutput);
                    _longTermAverageErrors.Add(errors.Average());
                    if (_longTermAverageErrors.Count > 1000)
                    {
                        _longTermAverageErrors.RemoveAt(0);
                    }
                    _iterations++;
                    if (_iterations % 2500 == 0)
                    {
                        if (!Directory.Exists(_directoryName))
                        {
                            Directory.CreateDirectory(_directoryName);
                        }
                        string filename = string.Format("NTM_{0}_{1}_{2}.ntm", _iterations, DateTime.Now.ToString("s").Replace(":", ""), GetLongTermErrorAverage());
                        _machine.Save(Path.Combine(_directoryName, filename));
                    }
                    lastKnown = sequence.Item2;
                    lastComputed = machinesOutput;
                    lastErrors = errors;
                }
            }
            if (lastErrors != null && lastKnown != null && lastComputed != null)
            {
                WriteExample(lastErrors, lastKnown, lastComputed);
            }
        }

        private void WriteExample(double[] errors, double[][] output, double[][] machinesOutput)
        {
            //Console.WriteLine(
            //    "TaskID: {0} Iteration: {1}, Match error: {2} Sets error: {3} Games error: {4} Points error: {5}",
            //    ID,
            //    _iterations,
            //    errors[0],
            //    errors[1],
            //    errors[2],
            //    errors[3]);

            Console.WriteLine(
                "TaskID: {0} Iteration: {1}, Game error: {2}",
                ID,
                _iterations,
                errors[0]);

            Console.WriteLine("Last example:");
            //Console.WriteLine("Match:");
            //WriteSequence(output, 0);
            //WriteSequence(machinesOutput, 0);

            //Console.WriteLine("Set:");
            //WriteSequence(output, 1);
            //WriteSequence(machinesOutput, 1);

            //Console.WriteLine("Game:");
            //WriteSequence(output, 2);
            //WriteSequence(machinesOutput, 2);

            Console.WriteLine("Point:");
            WriteSequence(output, 0);
            WriteSequence(machinesOutput, 0);
        }

        private static void WriteSequence(double[][] output, int index)
        {
            foreach (double[] known in output)
            {
                Console.Write("{0:0.00} ", known[index]);
            }
            Console.WriteLine();
        }


        private double[] CalculateLoss(double[][] output, double[][] machinesOutput)
        {
            double matchError = 0;
            //double setsError = 0;
            //double gamesError = 0;
            //double pointsError = 0;
            for (int i = 0; i < output.Length; i++)
            {
                double[] known = output[i];
                double[] computed = machinesOutput[i];
                matchError += Math.Abs(known[0] - computed[0]);
                //setsError += Math.Abs(known[1] - computed[1]);
                //gamesError += Math.Abs(known[2] - computed[2]);
                //pointsError += Math.Abs(known[3] - computed[3]);
            }

            //return new[] { matchError / output.Length, setsError / output.Length, gamesError / output.Length, pointsError / output.Length };
            return new[] { matchError / output.Length };
        }

        public double GetLongTermErrorAverage()
        {
            return _longTermAverageErrors.Average();
        }

        public void CopyFrom(LearningTask task, int weightsCount)
        {
            CopyMachine copyMachine = new CopyMachine(weightsCount, task._machine);
            _machine.UpdateWeights(copyMachine);
            _weightUpdater = task._weightUpdater.Clone();
            _teacher = new BPTTTeacher(_machine, _weightUpdater);
            _iterations = task._iterations;
            _longTermAverageErrors = new List<double>(task._longTermAverageErrors);
            Priority = task.Priority;
        }

        public double GetCurrentError()
        {
            return _longTermAverageErrors.Skip(Math.Max(0, _longTermAverageErrors.Count() - 100)).Average();
        }
    }
}
