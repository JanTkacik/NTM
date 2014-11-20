using AForge.Genetic;
using AForge.Math.Metrics;
using AForge.Neuro;
using NeuralTuringMachine.Controller;

namespace NeuralTuringMachine.GeneticsOptimalization
{
    public class IdealInputFitnessFunction : IFitnessFunction
    {
        private readonly double[] _input;
        private readonly double[] _idealOutput;
        private readonly Network _ntm;
        private readonly int _controllerInputsCount;
        private readonly IDistance _distance;

        public IdealInputFitnessFunction(double[] input, double[] idealOutput, Network network)
        {
            _input = input;
            _idealOutput = idealOutput;
            _ntm = network;
            _controllerInputsCount = network.InputsCount;
            _distance = new EuclideanDistance();
        }

        public double Evaluate(IChromosome chromosome)
        {
            DoubleArrayChromosome doubleArrayChromosome = (DoubleArrayChromosome)chromosome;
            double[] readFromMemoryInput = doubleArrayChromosome.Value;

            ControllerInput input = new ControllerInput(_input, readFromMemoryInput, _controllerInputsCount);
            double[] controllerOutput = _ntm.Compute(input.Input);

            return _distance.GetDistance(controllerOutput, _idealOutput);
        }
    }
}
