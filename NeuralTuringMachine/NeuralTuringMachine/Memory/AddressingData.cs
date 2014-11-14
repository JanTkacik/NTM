using System;
using System.Diagnostics;

namespace NeuralTuringMachine.Memory
{
    class AddressingData
    {
        private readonly double[] _keyVector;
        private readonly double _beta;
        private readonly double _g;
        private readonly double _s;
        private readonly double _gama;

        public AddressingData(double[] output, int outputOffset, int keyVectorLength)
        {
            _keyVector = new double[keyVectorLength];
            Array.Copy(output, outputOffset, _keyVector, 0, keyVectorLength);
            _beta = output[keyVectorLength + outputOffset];
            _g = output[keyVectorLength + outputOffset + 1];
            _s = output[keyVectorLength + outputOffset + 2];
            _gama = output[keyVectorLength + outputOffset + 3];
        }

        public double[] KeyVector
        {
            get { return _keyVector; }
        }

        public double KeyStrengthBeta
        {
            get { return _beta; }
        }

        public double InterpolationGate
        {
            get { return _g; }
        }

        public double ShiftWeighting
        {
            get { return _s; }
        }

        public double Sharpening
        {
            get
            {
                Debug.Assert(_gama >= 1);
                return _gama;
            }
        }
    }
}
