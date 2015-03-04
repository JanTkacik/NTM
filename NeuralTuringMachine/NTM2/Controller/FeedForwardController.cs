using System;

namespace NTM2.Controller
{
    //TODO refactor extract layers - input, hidden and output
    class FeedForwardController : IController
    {
        #region Fields and variables
        
        private readonly int _controllerSize;
        private readonly UnitFactory _unitFactory;

        //Controller hidden layer threshold weights
        private readonly Unit[] _hiddenLayerThresholds;

        //Weights from input to controller
        private readonly Unit[][] _inputToHiddenLayerWeights;
        
        //Weights from read data to controller
        private readonly Unit[][][] _wh1r;

        #endregion
        
        #region Ctor

        public FeedForwardController(int controllerSize, int inputSize, UnitFactory unitFactory)
        {
            _controllerSize = controllerSize;
            _unitFactory = unitFactory;

            _inputToHiddenLayerWeights = _unitFactory.GetTensor2(controllerSize, inputSize);
            _hiddenLayerThresholds = _unitFactory.GetVector(controllerSize);
        }

        #endregion

        #region Forward propagation

        //TODO refactor - do not use tempsum - but beware of rounding issues

        public double ForwardPropagation(double tempSum, int neuronIndex, double[] input)
        {
            double sum = tempSum;
            sum = GetInputContributionToHiddenLayer(neuronIndex, input, sum);
            sum = GetThresholdContributionToHiddenLayer(neuronIndex, sum);
            return sum;
        }

        private double GetInputContributionToHiddenLayer(int neuronIndex, double[] input, double tempSum)
        {
            Unit[] inputWeights = _inputToHiddenLayerWeights[neuronIndex];
            for (int j = 0; j < inputWeights.Length; j++)
            {
                tempSum += inputWeights[j].Value*input[j];
            }
            return tempSum;
        }

        private double GetThresholdContributionToHiddenLayer(int neuronIndex, double tempSum)
        {
            tempSum += _hiddenLayerThresholds[neuronIndex].Value;
            return tempSum;
        }

        #endregion

        #region Update weights

        public void UpdateWeights(Action<Unit> updateAction)
        {
            Action<Unit[]> vectorUpdateAction = Unit.GetVectorUpdateAction(updateAction);
            Action<Unit[][]> tensor2UpdateAction = Unit.GetTensor2UpdateAction(updateAction);

            tensor2UpdateAction(_inputToHiddenLayerWeights);
            vectorUpdateAction(_hiddenLayerThresholds);
        }

        #endregion

        #region BackwardErrorPropagation

		public void BackwardErrorPropagation(double[] hiddenLayerGradients, double[] input)
        {
            UpdateInputToHiddenWeightsGradients(hiddenLayerGradients, input);

            UpdateHiddenLayerThresholdsGradients(hiddenLayerGradients);
        }

        private void UpdateInputToHiddenWeightsGradients(double[] hiddenLayerGradients, double[] input)
        {
            for (int i = 0; i < _controllerSize; i++)
            {
                double hiddenGradient = hiddenLayerGradients[i];
                Unit[] inputToHiddenNeuronWeights = _inputToHiddenLayerWeights[i];

                UpdateInputGradient(hiddenGradient, inputToHiddenNeuronWeights, input);
            }
        }

        private void UpdateInputGradient(double hiddenLayerGradient, Unit[] inputToHiddenNeuronWeights, double[] input)
        {
            int inputLength = input.Length;
            for (int j = 0; j < inputLength; j++)
            {
                inputToHiddenNeuronWeights[j].Gradient += hiddenLayerGradient * input[j];
            }
        }

        private void UpdateHiddenLayerThresholdsGradients(double[] hiddenLayerGradients)
        {
            for (int i = 0; i < _controllerSize; i++)
            {
                _hiddenLayerThresholds[i].Gradient += hiddenLayerGradients[i];
            }
        }
        
        #endregion
    }
}
