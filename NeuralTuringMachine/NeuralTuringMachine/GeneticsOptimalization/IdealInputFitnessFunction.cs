using AForge.Genetic;
using AForge.Math.Metrics;
using AForge.Neuro;
using NeuralTuringMachine.Controller;

namespace NeuralTuringMachine.GeneticsOptimalization
{
    public class IdealInputFitnessFunction : IFitnessFunction
    {
        private readonly double[] _input;
        private readonly ControllerOutput _idealControllerOutput;
        private readonly NTM _machine;
        private readonly Network _controller;
        private readonly int _controllerInputsCount;

        public IdealInputFitnessFunction(double[] input, ControllerOutput idealControllerOutput, NTM machine)
        {
            _input = input;
            _idealControllerOutput = idealControllerOutput;
            _machine = machine;
            _controller = _machine.Controller;
            _controllerInputsCount = _controller.InputsCount;
        }

        public double Evaluate(IChromosome chromosome)
        {
            DoubleArrayChromosome doubleArrayChromosome = (DoubleArrayChromosome)chromosome;
            double[] readFromMemoryInput = doubleArrayChromosome.Value;

            ControllerInput input = new ControllerInput(_input, readFromMemoryInput, _controllerInputsCount);
            ControllerOutput computedOutput = new ControllerOutput(_controller.Compute(input.Input), _machine.OutputCount,  _machine.Memory.MemorySettings);

            return ControllerOutput.GetSimilarityScore(computedOutput, _idealControllerOutput);
        }
    }
}
