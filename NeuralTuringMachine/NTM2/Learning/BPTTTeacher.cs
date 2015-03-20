using System;
using NTM2.Controller;

namespace NTM2.Learning
{
    public class BPTTTeacher : INTMTeacher
    {
        private readonly NeuralTuringMachine _controller;
        private readonly IWeightUpdater _weightUpdater;
        private readonly IWeightUpdater _gradientResetter;

        public BPTTTeacher(NeuralTuringMachine controller, IWeightUpdater weightUpdater)
        {
            _controller = controller;
            _weightUpdater = weightUpdater;
            _gradientResetter = new GradientResetter();
        }

        public double[][] Train(double[][] input, double[][] knownOutput)
        {
            TrainableNTM[] machines = new TrainableNTM[input.Length];
            TrainableNTM empty = new TrainableNTM(_controller);

            machines[0] = new TrainableNTM(empty, input[0], _controller.UnitFactory);

            for (int i = 1; i < input.Length; i++)
            {
                machines[i] = new TrainableNTM(machines[i - 1], input[i], _controller.UnitFactory);
            }

            _gradientResetter.Reset();
            _controller.UpdateWeights(_gradientResetter);

            for (int i = input.Length - 1; i >= 0; i--)
            {
                machines[i].BackwardErrorPropagation(knownOutput[i]);
            }

            empty.DataBackwardPropagation();

            _weightUpdater.Reset();
            _controller.UpdateWeights(_weightUpdater);
            
            return GetMachineOutputs(machines);
        }

        private double[][] GetMachineOutputs(TrainableNTM[] machines)
        {
            double[][] realOutputs = new double[machines.Length][];
            for (int i = 0; i < machines.Length; i++)
            {
                TrainableNTM machine = machines[i];
                realOutputs[i] = machine.Controller.GetOutput();
            }
            return realOutputs;
        }

        public double[][][] Train(double[][][] inputs, double[][][] knownOutputs)
        {
            throw new NotImplementedException();
        }
    }
}
