using System;
using NTM2.Memory;

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
        
        public int HiddenLayerSize
        {
            get { return _controllerSize; }
        }

        #endregion
        
        #region Ctor

        public FeedForwardController(int controllerSize, int inputSize, int headCount, int memoryCellSizeM, UnitFactory unitFactory)
        {
            _controllerSize = controllerSize;
            _unitFactory = unitFactory;

            _wh1r = _unitFactory.GetTensor3(controllerSize, headCount, memoryCellSizeM);
            _inputToHiddenLayerWeights = _unitFactory.GetTensor2(controllerSize, inputSize);
            _hiddenLayerThresholds = _unitFactory.GetVector(controllerSize);
        }

        #endregion

        #region Forward propagation

        //TODO refactor - do not use tempsum - but beware of rounding issues
        
        public double ForwardPropagation(double tempSum, int neuronIndex, double[] input, ReadData[] readData)
        {
            double sum = tempSum;
            sum = GetReadDataContributionToHiddenLayer(neuronIndex, readData, sum);
            sum = GetInputContributionToHiddenLayer(neuronIndex, input, sum);
            sum = GetThresholdContributionToHiddenLayer(neuronIndex, sum);
            return sum;
        }

        private double GetReadDataContributionToHiddenLayer(int neuronIndex, ReadData[] readData, double sum)
        {
            //TODO continue to refactor
            //Foreach head
            Unit[][] headsWeights = _wh1r[neuronIndex];
            for (int j = 0; j < headsWeights.Length; j++)
            {
                //Foreach read data
                Unit[] weights = headsWeights[j];
                ReadData read = readData[j];

                for (int k = 0; k < weights.Length; k++)
                {
                    sum += weights[k].Value*read.Data[k].Value;
                }
            }
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
            Action<Unit[][][]> tensor3UpdateAction = Unit.GetTensor3UpdateAction(updateAction);
            
            tensor3UpdateAction(_wh1r);
            tensor2UpdateAction(_inputToHiddenLayerWeights);
            vectorUpdateAction(_hiddenLayerThresholds);
        }

        #endregion

        #region BackwardErrorPropagation

		public void BackwardErrorPropagation(double[] hiddenLayerGradients, double[] input, ReadData[] reads)
        {
            UpdateReadDataGradient(hiddenLayerGradients, reads);

            UpdateReadDataToHiddenWeightsGradient(hiddenLayerGradients, reads);

            UpdateInputToHiddenWeightsGradients(hiddenLayerGradients, input);

            UpdateHiddenLayerThresholdsGradients(hiddenLayerGradients);
        }

        private void UpdateReadDataGradient(double[] hiddenLayerGradients, ReadData[] reads)
        {
            //TODO continue refactoring
            for (int k = 0; k < hiddenLayerGradients.Length; k++)
            {
                Unit[][] wh1rk = _wh1r[k];
                for (int i = 0; i < reads.Length; i++)
                {
                    ReadData readData = reads[i];
                    Unit[] wh1rki = wh1rk[i];
                    for (int j = 0; j < wh1rki.Length; j++)
                    {
                        readData.Data[j].Gradient += hiddenLayerGradients[k]*wh1rki[j].Value;
                    }
                }
            }
        }

        private void UpdateReadDataToHiddenWeightsGradient(double[] hiddenLayerGradients, ReadData[] reads)
        {
            for (int neuronIndex = 0; neuronIndex < _controllerSize; neuronIndex++)
            {
                Unit[][] hiddenLayerNeuronToReadDataWeights = _wh1r[neuronIndex];
                double hiddenGradient = hiddenLayerGradients[neuronIndex];

                //TODO change to headcount 
                //TODO finish refactoring
                for (int headIndex = 0; headIndex < hiddenLayerNeuronToReadDataWeights.Length; headIndex++)
                {
                    Unit[] wh1rij = hiddenLayerNeuronToReadDataWeights[headIndex];
                    for (int k = 0; k < reads[headIndex].Data.Length; k++)
                    {
                        Unit read = reads[headIndex].Data[k];
                        wh1rij[k].Gradient += hiddenGradient*read.Value;
                    }
                }
            }
        }

        private void UpdateInputToHiddenWeightsGradients(double[] hiddenLayerGradients, double[] input)
        {
            for (int neuronIndex = 0; neuronIndex < _controllerSize; neuronIndex++)
            {
                double hiddenGradient = hiddenLayerGradients[neuronIndex];
                Unit[] inputToHiddenNeuronWeights = _inputToHiddenLayerWeights[neuronIndex];

                UpdateInputGradient(hiddenGradient, inputToHiddenNeuronWeights, input);
            }
        }

        private void UpdateInputGradient(double hiddenLayerGradient, Unit[] inputToHiddenNeuronWeights, double[] input)
        {
            //TODO change to use stored value
            int inputLength = input.Length;
            for (int inputIndex = 0; inputIndex < inputLength; inputIndex++)
            {
                inputToHiddenNeuronWeights[inputIndex].Gradient += hiddenLayerGradient * input[inputIndex];
            }
        }

        private void UpdateHiddenLayerThresholdsGradients(double[] hiddenLayerGradients)
        {
            for (int neuronIndex = 0; neuronIndex < _controllerSize; neuronIndex++)
            {
                _hiddenLayerThresholds[neuronIndex].Gradient += hiddenLayerGradients[neuronIndex];
            }
        }
        
        #endregion
    }
}
