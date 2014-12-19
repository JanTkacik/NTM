using System;
using AForge.Math.Metrics;
using NeuralTuringMachine.Memory.Head;
using NeuralTuringMachine.Misc;

namespace NeuralTuringMachine.Memory
{
    public class AddressingData
    {
        //FOCUS CONSTANT IS NOT MENTIONED IN TEXT
        private static ISimilarity _similarity = new EuclideanSimilarity();

        private const double FocusConstant = 1;
        private const double SharpeningConstant = 1;

        private readonly double[] _keyVector;
        private readonly double _beta;
        private readonly double _g;
        private readonly double[] _s;
        private readonly double _gama;
        
        private AddressingData(double[] keyVector, double beta, double g, double[] s, double gama)
        {
            _keyVector = keyVector;
            _beta = beta;
            _g = g;
            _s = s;
            _gama = gama;
        }

        public AddressingData(double[] readHeadsOutput, MemorySettings memorySettings)
        {
            int maxConvShift = memorySettings.MaxConvolutionalShift;
            int keyVectorLength = memorySettings.MemoryVectorLength;
            var convShiftLen = (maxConvShift * 2) + 1;
            _keyVector = new double[keyVectorLength];
            _s = new double[convShiftLen];
            Array.Copy(readHeadsOutput, 0, _keyVector, 0, keyVectorLength);
            _beta = readHeadsOutput[keyVectorLength];
            _g = readHeadsOutput[keyVectorLength + 1];
            Array.Copy(readHeadsOutput, keyVectorLength + 2, _s, 0, convShiftLen);
            _gama = readHeadsOutput[keyVectorLength + convShiftLen + 2];
            NormalizeConvolutionVector();
        }

        private void NormalizeConvolutionVector()
        {
            ArrayHelper.NormalizeVector(_s);
        }

        public double[] KeyVector
        {
            get { return _keyVector; }
        }

        public double KeyStrengthBeta
        {
            get { return _beta * FocusConstant; }
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
                return Math.Exp(_gama * SharpeningConstant);
            }
        }

        public AddressingData Clone()
        {
            return new AddressingData(ArrayHelper.CloneArray(_keyVector), _beta, _g, ArrayHelper.CloneArray(_s), _gama);
        }

        public static double GetSimilarityScore(AddressingData dataA, AddressingData dataB)
        {
            double keySimilarity = GetKeyVectorSimilarity(dataA.KeyVector, dataA._g, dataB.KeyVector, dataB._g);
            double betaSimilarity = GetBhattacharyyaSimilarity(dataA._beta, dataA._g, dataB._beta, dataB._g);
            double convolutionSimilarity = _similarity.GetSimilarityScore(dataA._s, dataB._s);
            double sharpeningSimilarity = 1/(1 + Math.Abs(dataA._gama - dataB._gama));

            return (keySimilarity + betaSimilarity + convolutionSimilarity + sharpeningSimilarity)/4;
        }

        private static double GetKeyVectorSimilarity(double[] keyA, double gA, double[] keyB, double gB)
        {
            int length = keyA.Length;
            double similaritySum = 0;
            for (int i = 0; i < length; i++)
            {
                similaritySum += GetBhattacharyyaSimilarity(keyA[i], gA, keyB[i], gB);
            }
            return similaritySum/length;
        }

        private static double GetBhattacharyyaSimilarity(double a, double gA, double b, double gB)
        {
            double gaR = 1 - gA;
            double gbR = 1 - gB;

            double distance =  (0.25*((Math.Pow(a - b, 2))/(gaR + gbR))) + 
                   (0.25 * Math.Log(0.25 * ((gaR / gbR) + (gbR/gaR) + 2)));

            return 1/(1 + distance);
        }
    }
}
