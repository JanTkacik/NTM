using System;

namespace NeuralTuringMachine.Memory
{
    class AddressingData
    {
        private readonly double[] _keyVector;
        private double _beta;
        private double _g;
        private double _s;
        private double _gama;

        public AddressingData(double[] output, int outputOffset, int keyVectorLength)
        {
            _keyVector = new double[keyVectorLength];
            Array.Copy(output, outputOffset, _keyVector, 0, keyVectorLength);
            _beta = output[keyVectorLength + outputOffset];
            _g = output[keyVectorLength + outputOffset + 1];
            _s = output[keyVectorLength + outputOffset + 2];
            _gama = output[keyVectorLength + outputOffset + 3];
        }
    }
}
