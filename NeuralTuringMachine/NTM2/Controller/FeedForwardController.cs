using System;
using NTM2.Memory;

namespace NTM2.Controller
{
    class FeedForwardController : IController
    {
        #region Fields and variables

        internal readonly HiddenLayer HiddenLayer;
        internal readonly OutputLayer OutputLayer;

        #endregion

        #region Ctor

        public FeedForwardController(int controllerSize, int inputSize, int outputSize, int headCount, int memoryUnitSizeM, UnitFactory unitFactory)
        {
            HiddenLayer = new HiddenLayer(controllerSize, inputSize, headCount, memoryUnitSizeM, unitFactory);
            OutputLayer = new OutputLayer(outputSize, controllerSize, headCount, memoryUnitSizeM, unitFactory);
        }

        private FeedForwardController(HiddenLayer hiddenLayer, OutputLayer outputLayer)
        {
            HiddenLayer = hiddenLayer;
            OutputLayer = outputLayer;
        }

        public IController Clone()
        {
            HiddenLayer newHiddenLayer = HiddenLayer.Clone();
            OutputLayer newOutputLayer = OutputLayer.Clone();
            return new FeedForwardController(newHiddenLayer, newOutputLayer);
        }

        #endregion

        #region Forward propagation

        public void ForwardPropagation(double[] input, ReadData[] readData)
        {
            HiddenLayer.ForwardPropagation(input, readData);
            OutputLayer.ForwardPropagation(HiddenLayer);
        }

        #endregion

        #region Update weights

        public void UpdateWeights(Action<Unit> updateAction)
        {
            OutputLayer.UpdateWeights(updateAction);
            HiddenLayer.UpdateWeights(updateAction);
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
