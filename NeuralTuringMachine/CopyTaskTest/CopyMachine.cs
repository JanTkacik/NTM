using NTM2;
using NTM2.Controller;
using NTM2.Learning;

namespace CopyTaskTest
{
    class CopyMachine : WeightUpdaterBase
    {
        private int _i;
        private readonly double[] _weights;
        private readonly double[] _gradients;

        public CopyMachine(int weightsCount, NeuralTuringMachine machine)
        {
            _weights = new double[weightsCount];
            _gradients = new double[weightsCount];

            var extractor = new WeightsExtractor(_weights, _gradients);
            machine.UpdateWeights(extractor);
        }

        public override void UpdateWeight(Unit data)
        {
            data.Value = _weights[_i];
            data.Gradient = _gradients[_i];
            _i++;
        }

        public override void Reset()
        {
            _i = 0;
        }

        private class WeightsExtractor : WeightUpdaterBase
        {
            private readonly double[] _weights;
            private readonly double[] _gradients;
            private int _i;

            public WeightsExtractor(double[] weights, double[] gradients)
            {
                _weights = weights;
                _gradients = gradients;
            }

            public override void Reset()
            {
                _i = 0;
            }

            public override void UpdateWeight(Unit data)
            {
                _weights[_i] = data.Value;
                _gradients[_i] = data.Gradient;
                _i++;
            }
        }
    }


}
