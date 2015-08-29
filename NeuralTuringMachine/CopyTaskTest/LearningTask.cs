using System;
using System.Collections.Generic;
using System.Linq;
using NTM2;
using NTM2.Learning;

namespace CopyTaskTest
{
    class LearningTask
    {
        private int _iterations;
        private const int EpochLength = 10;
        private readonly Random _rand = new Random(DateTime.Now.Millisecond);
        private readonly NeuralTuringMachine _machine;
        private RMSPropWeightUpdater _weightUpdater;
        private readonly int _id;
        private BPTTTeacher _teacher;
        private List<double> _longTermAverageErrors;

        public LearningTask(NeuralTuringMachine machine, RMSPropWeightUpdater weightUpdater, int id)
        {
            _iterations = 0;
            _machine = machine;
            _weightUpdater = weightUpdater;
            _id = id;
            _teacher = new BPTTTeacher(_machine, weightUpdater);
            _longTermAverageErrors = new List<double>();
            Priority = 10;
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
            const int vectorSize = 8;
            const int minSeqLen = 1;
            const int maxSeqLen = 20;

            for (int j = 0; j < Priority; j++)
            {
                for (int i = 0; i < EpochLength; i++)
                {
                    var sequence = SequenceGenerator.GenerateSequence(_rand.Next(minSeqLen, maxSeqLen), vectorSize);
                    double[][] machinesOutput = _teacher.Train(sequence.Item1, sequence.Item2);
                    double error = CalculateLoss(sequence.Item2, machinesOutput);
                    _longTermAverageErrors.Add(error);
                    if (_longTermAverageErrors.Count > 1000)
                    {
                        _longTermAverageErrors.RemoveAt(0);
                    }
                    _iterations++;
                }
            }
        }

        private double CalculateLoss(double[][] knownOutput, double[][] machinesOutput)
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
