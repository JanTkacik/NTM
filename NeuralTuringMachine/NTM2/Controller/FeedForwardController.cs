using System;
using NTM2.Learning;
using NTM2.Memory;

namespace NTM2.Controller
{
    class FeedForwardController 
    {
        #region Fields and variables

        internal readonly HiddenLayer HiddenLayer;
        internal readonly OutputLayer OutputLayer;

        #endregion

        #region Ctor

        public FeedForwardController(int controllerSize, int inputSize, int outputSize, int headCount, int memoryUnitSizeM)
        {
            HiddenLayer = new HiddenLayer(controllerSize, inputSize, headCount, memoryUnitSizeM);
            OutputLayer = new OutputLayer(outputSize, controllerSize, headCount, memoryUnitSizeM);
        }

        private FeedForwardController(HiddenLayer hiddenLayer, OutputLayer outputLayer)
        {
            HiddenLayer = hiddenLayer;
            OutputLayer = outputLayer;
        }

        public double[] GetOutput()
        {
            return OutputLayer.GetOutput();
        }

        public FeedForwardController Clone()
        {
            HiddenLayer newHiddenLayer = HiddenLayer.Clone();
            OutputLayer newOutputLayer = OutputLayer.Clone();
            return new FeedForwardController(newHiddenLayer, newOutputLayer);
        }

        #endregion

        #region Forward propagation

        public void ForwardPropagation(double[] input, MemoryState memoryState)
        {
            HiddenLayer.ForwardPropagation(input, memoryState.ReadData);
            OutputLayer.ForwardPropagation(HiddenLayer, memoryState.HeadSettings);
        }

        #endregion

        #region Update weights

        public void UpdateWeights(IWeightUpdater weightUpdater)
        {
            OutputLayer.UpdateWeights(weightUpdater);
            HiddenLayer.UpdateWeights(weightUpdater);
        }

        #endregion

        #region BackwardErrorPropagation

        public void BackwardErrorPropagation(double[] knownOutput, double[] input, ReadData[] reads)
        {
            OutputLayer.BackwardErrorPropagation(knownOutput, HiddenLayer);
            HiddenLayer.BackwardErrorPropagation(input, reads);
        }

        #endregion
    }
}
