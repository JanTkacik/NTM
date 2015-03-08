using System;

namespace NTM2.Controller
{
    internal class OutputLayer
    {
        private readonly UnitFactory _unitFactory;

        private readonly int _outputSize;
        private readonly int _controllerSize;
        private readonly int _headCount;
        private readonly HiddenLayer _hiddenLayer;

        //Weights from controller to output
        internal readonly Unit[][] _wyh1;

        //Weights from controller to head
        internal readonly Unit[][][] _wuh1;
        
        internal readonly Unit[] _outputLayer;

        public OutputLayer(int outputSize, int controllerSize, int headCount, int headUnitSize, HiddenLayer hiddenLayer, UnitFactory unitFactory)
        {
            _outputSize = outputSize;
            _controllerSize = controllerSize;
            _headCount = headCount;
            _hiddenLayer = hiddenLayer;
            _unitFactory = unitFactory;
            _wyh1 = _unitFactory.GetTensor2(outputSize, controllerSize + 1);
            _wuh1 = _unitFactory.GetTensor3(headCount, headUnitSize, controllerSize + 1);
        }

        private OutputLayer(Unit[][] wyh1, Unit[][][] wuh1, Unit[] outputLayer, int headCount, int outputSize, int controllerSize, HiddenLayer hiddenLayer, UnitFactory unitFactory)
        {
            _wyh1 = wyh1;
            _wuh1 = wuh1;
            _controllerSize = controllerSize;
            _outputSize = outputSize;
            _outputLayer = outputLayer;
            _headCount = headCount;
            _hiddenLayer = hiddenLayer;
            _unitFactory = unitFactory;
        }

        public void ForwardPropagation()
        {
            ////Foreach neuron in classic output layer
            //for (int i = 0; i < _outputSize; i++)
            //{
            //    double sum = 0;
            //    Unit[] weights = _wyh1[i];

            //    //Foreach input from hidden layer
            //    for (int j = 0; j < _controllerSize; j++)
            //    {
            //        sum += weights[j].Value * _hiddenLayer.HiddenLayerNeurons[j].Value;
            //    }

            //    //Plus threshold
            //    sum += weights[_controllerSize].Value;
            //    _outputLayer[i].Value = Sigmoid.GetValue(sum);
            //}
        }

        public OutputLayer Clone()
        {
            Unit[] outputLayer = _unitFactory.GetVector(_outputSize);

            return new OutputLayer(_wyh1, _wuh1, outputLayer, _headCount, _outputSize, _controllerSize, _hiddenLayer, _unitFactory);
        }

        public void BackwardErrorPropagation()
        {
            
        }

        public void UpdateWeights(Action<Unit> updateAction)
        {
            Action<Unit[][]> tensor2UpdateAction = Unit.GetTensor2UpdateAction(updateAction);
            Action<Unit[][][]> tensor3UpdateAction = Unit.GetTensor3UpdateAction(updateAction);

            tensor2UpdateAction(_wyh1);
            tensor3UpdateAction(_wuh1);
        }
    }
}
