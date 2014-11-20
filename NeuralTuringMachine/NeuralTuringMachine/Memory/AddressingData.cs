using System;

namespace NeuralTuringMachine.Memory
{
    class AddressingData
    {
        private readonly double[] _keyVector;
        private readonly double _beta;
        private readonly double _g;
        private readonly double[] _s;
        private readonly double _gama;

        public AddressingData(double[] output, int keyVectorLength, int maxConvShift)
        {
            var convShiftLen = (maxConvShift * 2) + 1;
            _keyVector = new double[keyVectorLength];
            _s = new double[convShiftLen];
            Array.Copy(output, 0, _keyVector, 0, keyVectorLength);
            _beta = output[keyVectorLength];
            _g = output[keyVectorLength + 1];
            Array.Copy(output, keyVectorLength + 2, _s, 0, convShiftLen);
            _gama = output[keyVectorLength + convShiftLen + 3];

            NormalizeConvolutionVector();
        }

        private void NormalizeConvolutionVector()
        {
            double sumOfSquares = 0;
            foreach (double t in _s)
            {
                sumOfSquares += Math.Pow(t, 2);
            }
            sumOfSquares = Math.Pow(sumOfSquares, 0.5);
            for (int i = 0; i < _s.Length; i++)
            {
                _s[i] = _s[i]/sumOfSquares;
            }
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

        public double[] ShiftWeighting
        {
            get { return _s; }
        }

        public double Sharpening
        {
            get
            {
                // SHARPENING MUST BE LARGER THAN ONE - SHARPENING SMALLER THAN ONE MEANS BLURRING
                return _gama + 1;
            }
        }
    }
}
