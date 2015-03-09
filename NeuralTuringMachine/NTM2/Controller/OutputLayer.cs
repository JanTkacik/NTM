using System;
using NTM2.Memory.Addressing;

namespace NTM2.Controller
{
    internal class OutputLayer
    {
        private readonly UnitFactory _unitFactory;

        private readonly int _outputSize;
        private readonly int _controllerSize;
        private readonly int _headCount;
        private readonly int _memoryUnitSizeM;
        private readonly HiddenLayer _hiddenLayer;

        //Weights from controller to output
        internal readonly Unit[][] _wyh1;

        //Weights from controller to head
        internal readonly Unit[][][] _wuh1;
        
        internal readonly Unit[] _outputLayer;
        
        internal readonly Head[] _heads;
        private readonly int _headUnitSize;

        public OutputLayer(int outputSize, int controllerSize, int headCount, int memoryUnitSizeM, HiddenLayer hiddenLayer, UnitFactory unitFactory)
        {
            _outputSize = outputSize;
            _controllerSize = controllerSize;
            _headCount = headCount;
            _memoryUnitSizeM = memoryUnitSizeM;
            _hiddenLayer = hiddenLayer;
            _unitFactory = unitFactory;
            _headUnitSize = Head.GetUnitSize(memoryUnitSizeM);
            _wyh1 = _unitFactory.GetTensor2(outputSize, controllerSize + 1);
            _wuh1 = _unitFactory.GetTensor3(headCount, _headUnitSize, controllerSize + 1);
            _heads = new Head[headCount];
        }

        private OutputLayer(Unit[][] wyh1, Unit[][][] wuh1, Unit[] outputLayer, Head[] heads, int headCount, int outputSize, int controllerSize, int memoryUnitSizeM, int headUnitSize, HiddenLayer hiddenLayer, UnitFactory unitFactory)
        {
            _wyh1 = wyh1;
            _wuh1 = wuh1;
            _heads = heads;
            _controllerSize = controllerSize;
            _outputSize = outputSize;
            _outputLayer = outputLayer;
            _headCount = headCount;
            _hiddenLayer = hiddenLayer;
            _unitFactory = unitFactory;
            _memoryUnitSizeM = memoryUnitSizeM;
            _headUnitSize = headUnitSize;
        }

        public void ForwardPropagation()
        {
            //Foreach neuron in classic output layer
            for (int i = 0; i < _outputSize; i++)
            {
                double sum = 0;
                Unit[] weights = _wyh1[i];

                //Foreach input from hidden layer
                for (int j = 0; j < _controllerSize; j++)
                {
                    sum += weights[j].Value * _hiddenLayer.HiddenLayerNeurons[j].Value;
                }

                //Plus threshold
                sum += weights[_controllerSize].Value;
                _outputLayer[i].Value = Sigmoid.GetValue(sum);
            }

            //Foreach neuron in head output layer
            for (int i = 0; i < _headCount; i++)
            {
                Unit[][] headsWeights = _wuh1[i];
                Head head = _heads[i];

                for (int j = 0; j < headsWeights.Length; j++)
                {
                    double sum = 0;
                    Unit[] headWeights = headsWeights[j];
                    //Foreach input from hidden layer
                    for (int k = 0; k < _controllerSize; k++)
                    {
                        sum += headWeights[k].Value * _hiddenLayer.HiddenLayerNeurons[k].Value;
                    }
                    //Plus threshold
                    sum += headWeights[_controllerSize].Value;
                    head[j].Value += sum;
                }
            }
        }

        public OutputLayer Clone(HiddenLayer newHiddenLayer)
        {
            Unit[] outputLayer = _unitFactory.GetVector(_outputSize);
            Head[] heads = Head.GetVector(_headCount, i => _memoryUnitSizeM, _unitFactory);

            return new OutputLayer(_wyh1, _wuh1, outputLayer, heads, _headCount, _outputSize, _controllerSize, _memoryUnitSizeM, _headUnitSize, newHiddenLayer, _unitFactory);
        }

        public void BackwardErrorPropagation()
        {
            //Output error backpropagation
            for (int j = 0; j < _outputSize; j++)
            {
                Unit unit = _outputLayer[j];
                Unit[] weights = _wyh1[j];
                for (int i = 0; i < _controllerSize; i++)
                {
                    _hiddenLayer.HiddenLayerNeurons[i].Gradient += weights[i].Value * unit.Gradient;
                }
            }

            //Heads error backpropagation
            for (int j = 0; j < _headCount; j++)
            {
                Head head = _heads[j];
                Unit[][] weights = _wuh1[j];
                for (int k = 0; k < _headUnitSize; k++)
                {
                    Unit unit = head[k];
                    Unit[] weightsK = weights[k];
                    for (int i = 0; i < _controllerSize; i++)
                    {
                        _hiddenLayer.HiddenLayerNeurons[i].Gradient += unit.Gradient * weightsK[i].Value;
                    }
                }
            }

            //Wyh1 error backpropagation
            for (int i = 0; i < _outputSize; i++)
            {
                Unit[] wyh1I = _wyh1[i];
                double yGrad = _outputLayer[i].Gradient;
                for (int j = 0; j < _controllerSize; j++)
                {
                    wyh1I[j].Gradient += yGrad * _hiddenLayer.HiddenLayerNeurons[j].Value;
                }
                wyh1I[_controllerSize].Gradient += yGrad;
            }

            //TODO refactor names
            //Wuh1 error backpropagation
            for (int i = 0; i < _headCount; i++)
            {
                Head head = _heads[i];
                Unit[][] units = _wuh1[i];
                for (int j = 0; j < _headUnitSize; j++)
                {
                    Unit headUnit = head[j];
                    Unit[] wuh1ij = units[j];

                    for (int k = 0; k < _controllerSize; k++)
                    {
                        Unit unit = _hiddenLayer.HiddenLayerNeurons[k];
                        wuh1ij[k].Gradient += headUnit.Gradient * unit.Value;
                    }
                    wuh1ij[_controllerSize].Gradient += headUnit.Gradient;
                }
            }
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
