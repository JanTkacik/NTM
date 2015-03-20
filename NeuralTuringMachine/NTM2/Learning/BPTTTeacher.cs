using System;
using NTM2.Controller;

namespace NTM2.Learning
{
    public class BPTTTeacher : INTMTeacher
    {
        private readonly NTMController _controller;
        private readonly IWeightUpdater _weightUpdater;

        public BPTTTeacher(NTMController controller, IWeightUpdater weightUpdater)
        {
            _controller = controller;
            _weightUpdater = weightUpdater;
        }

        public TrainableNTM[] Train(double[][] input, double[][] knownOutput)
        {
            TrainableNTM[] machines = _controller.ProcessAndUpdateErrors(input, knownOutput);
            _weightUpdater.Reset();
            _controller.UpdateWeights(_weightUpdater);
            return machines;
        }

        public TrainableNTM[] Train(double[][][] inputs, double[][][] knownOutputs)
        {
            throw new System.NotImplementedException();
        }
    }
}
