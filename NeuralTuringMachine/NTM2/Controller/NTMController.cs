using System;
using NTM2.Memory;
using NTM2.Memory.Addressing;

namespace NTM2.Controller
{
    public class NTMController
    {
        private readonly UnitFactory _unitFactory;
        
        private readonly int _memoryColumnsN;
        private readonly int _memoryRowsM;
        private readonly int _weightsCount;
        private readonly Head[] _heads;
        private readonly Unit[] _outputLayer;
        private readonly double[] _input;
        private readonly ReadData[] _reads;
        private readonly NTMMemory _memory;

        //Weights from controller to head
        private readonly Unit[][][] _wuh1;
        //Weights from controller to output
        private readonly Unit[][] _wyh1;
        
        private readonly IController _controller;

        //Old similarities
        private readonly BetaSimilarity[][] _wtm1s;

        public int WeightsCount
        {
            get { return _weightsCount; }
        }

        public int HeadCount
        {
            get { return _heads.Length; }
        }

        public Head[] Heads
        {
            get { return _heads; }
        }

        public Unit[] Output
        {
            get { return _outputLayer; }
        }

        public NTMController(int inputSize, int outputSize, int controllerSize, int headCount, int memoryColumnsN, int memoryRowsM)
        {
            _unitFactory = new UnitFactory();
            _memoryColumnsN = memoryColumnsN;
            _memoryRowsM = memoryRowsM;
            int headUnitSize = Head.GetUnitSize(memoryRowsM);
            _heads = new Head[headCount];
            _wtm1s = BetaSimilarity.GetTensor2(headCount, memoryColumnsN);
            _memory = new NTMMemory(memoryColumnsN, memoryRowsM, _unitFactory);
            
            _controller = new FeedForwardController(controllerSize, inputSize, headCount, memoryRowsM, _unitFactory);

            _wyh1 = _unitFactory.GetTensor2(outputSize, controllerSize + 1);
            _wuh1 = _unitFactory.GetTensor3(headCount, headUnitSize, controllerSize + 1);

            _weightsCount =
                (headCount * memoryColumnsN) +
                (memoryColumnsN * memoryRowsM) +
                (controllerSize * headCount * memoryRowsM) +
                (controllerSize * inputSize) +
                (controllerSize) +
                (outputSize * (controllerSize + 1)) +
                (headCount * headUnitSize * (controllerSize + 1));
        }

        private NTMController(
            int memoryColumnsN,
            int memoryRowsM,
            Unit[][] wyh1,
            Unit[][][] wuh1,
            int weightsCount,
            ReadData[] readDatas,
            double[] input,
            IController controller,
            Unit[] outputLayer,
            Head[] heads,
            UnitFactory unitFactory)
        {
            _unitFactory = unitFactory;
            _memoryColumnsN = memoryColumnsN;
            _memoryRowsM = memoryRowsM;
            _wyh1 = wyh1;
            _wuh1 = wuh1;
            _weightsCount = weightsCount;
            _reads = readDatas;
            _input = input;
            _outputLayer = outputLayer;
            _heads = heads;
            _controller = controller;
        }

        public TrainableNTM[] ProcessAndUpdateErrors(double[][] input, double[][] knownOutput)
        {
            //FOREACH HEAD - SET WEIGHTS TO BIAS VALUES
            ContentAddressing[] contentAddressings = ContentAddressing.GetVector(HeadCount, i => _wtm1s[i], _unitFactory);

            HeadSetting[] oldSettings = HeadSetting.GetVector(HeadCount, i => new Tuple<int, ContentAddressing>(_memory.MemoryColumnsN, contentAddressings[i]), _unitFactory);
            ReadData[] readDatas = ReadData.GetVector(HeadCount, i => new Tuple<HeadSetting, NTMMemory>(oldSettings[i], _memory));

            TrainableNTM[] machines = new TrainableNTM[input.Length];
            TrainableNTM empty = new TrainableNTM(this, new MemoryState(oldSettings, readDatas, _memory));

            //BPTT
            machines[0] = new TrainableNTM(empty, input[0], _unitFactory);
            for (int i = 1; i < input.Length; i++)
            {
                machines[i] = new TrainableNTM(machines[i - 1], input[i], _unitFactory);
            }

            UpdateWeights(unit => unit.Gradient = 0);

            for (int i = input.Length - 1; i >= 0; i--)
            {
                TrainableNTM machine = machines[i];
                double[] output = knownOutput[i];

                for (int j = 0; j < output.Length; j++)
                {
                    //Delta
                    machine.Controller._outputLayer[j].Gradient = machine.Controller._outputLayer[j].Value - output[j];
                }
                machine.BackwardErrorPropagation();
            }

            //Compute gradients for the bias values of internal memory and weights
            for (int i = 0; i < readDatas.Length; i++)
            {
                readDatas[i].BackwardErrorPropagation();
                for (int j = 0; j < readDatas[i].HeadSetting.Data.Length; j++)
                {
                    contentAddressings[i].Data[j].Gradient += readDatas[i].HeadSetting.Data[j].Gradient;
                }
                contentAddressings[i].BackwardErrorPropagation();
            }

            return machines;
        }

        public NTMController Process(ReadData[] readData, double[] input)
        {
            NTMController newController = new NTMController(
                _memoryColumnsN,
                _memoryRowsM,
                _wyh1,
                _wuh1,
                _weightsCount,
                readData,
                input,
                _controller.Clone(),
                _unitFactory.GetVector(_wyh1.Length),
                Head.GetVector(readData.Length, i => _memoryRowsM, _unitFactory),
                _unitFactory);

            newController.ForwardPropagation(readData, input);
            return newController;
        }

        //TODO readData Units are maybe not important
        private void ForwardPropagation(ReadData[] readData, double[] input)
        {
            //Foreach neuron in hidden layer
            for (int i = 0; i < _controller.HiddenLayerSize; i++)
            {
                double sum = 0;
                
                sum = _controller.ForwardPropagation(sum, i, input, readData);
                
                //Set new controller unit value
                ((FeedForwardController)_controller)._hiddenLayer1[i].Value = Sigmoid.GetValue(sum);
            }

            //Foreach neuron in classic output layer
            for (int i = 0; i < _wyh1.Length; i++)
            {
                double sum = 0;
                Unit[] weights = _wyh1[i];

                //Foreach input from hidden layer
                for (int j = 0; j < ((FeedForwardController)_controller)._hiddenLayer1.Length; j++)
                {
                    sum += weights[j].Value * ((FeedForwardController)_controller)._hiddenLayer1[j].Value;
                }

                //Plus threshold
                sum += weights[((FeedForwardController)_controller)._hiddenLayer1.Length].Value;
                _outputLayer[i].Value = Sigmoid.GetValue(sum);
            }

            //Foreach neuron in head output layer
            for (int i = 0; i < _wuh1.Length; i++)
            {
                Unit[][] headsWeights = _wuh1[i];
                Head head = _heads[i];

                for (int j = 0; j < headsWeights.Length; j++)
                {
                    double sum = 0;
                    Unit[] headWeights = headsWeights[j];
                    //Foreach input from hidden layer
                    for (int k = 0; k < ((FeedForwardController)_controller)._hiddenLayer1.Length; k++)
                    {
                        sum += headWeights[k].Value * ((FeedForwardController)_controller)._hiddenLayer1[k].Value;
                    }
                    //Plus threshold
                    sum += headWeights[((FeedForwardController)_controller)._hiddenLayer1.Length].Value;
                    head[j].Value += sum;
                }
            }
        }

        public void UpdateWeights(Action<Unit> updateAction)
        {
            foreach (BetaSimilarity[] betaSimilarities in _wtm1s)
            {
                foreach (BetaSimilarity betaSimilarity in betaSimilarities)
                {
                    updateAction(betaSimilarity.Data);
                }
            }

            Action<Unit[][]> tensor2UpdateAction = Unit.GetTensor2UpdateAction(updateAction);
            Action<Unit[][][]> tensor3UpdateAction = Unit.GetTensor3UpdateAction(updateAction);

            tensor2UpdateAction(_memory.Data);
            tensor2UpdateAction(_wyh1);
            tensor3UpdateAction(_wuh1);
            
            _controller.UpdateWeights(updateAction);
        }
        
        public void BackwardErrorPropagation()
        {
            //Output error backpropagation
            for (int j = 0; j < _outputLayer.Length; j++)
            {
                Unit unit = _outputLayer[j];
                Unit[] weights = _wyh1[j];
                for (int i = 0; i < ((FeedForwardController)_controller)._hiddenLayer1.Length; i++)
                {
                    ((FeedForwardController)_controller)._hiddenLayer1[i].Gradient += weights[i].Value * unit.Gradient;
                }
            }

            //Heads error backpropagation
            for (int j = 0; j < _heads.Length; j++)
            {
                Head head = _heads[j];
                Unit[][] weights = _wuh1[j];
                for (int k = 0; k < head.GetUnitSize(); k++)
                {
                    Unit unit = head[k];
                    Unit[] weightsK = weights[k];
                    for (int i = 0; i < ((FeedForwardController)_controller)._hiddenLayer1.Length; i++)
                    {
                        ((FeedForwardController)_controller)._hiddenLayer1[i].Gradient += unit.Gradient * weightsK[i].Value;
                    }
                }
            }

            //Wyh1 error backpropagation
            for (int i = 0; i < _wyh1.Length; i++)
            {
                Unit[] wyh1I = _wyh1[i];
                double yGrad = _outputLayer[i].Gradient;
                for (int j = 0; j < ((FeedForwardController)_controller)._hiddenLayer1.Length; j++)
                {
                    wyh1I[j].Gradient += yGrad * ((FeedForwardController)_controller)._hiddenLayer1[j].Value;
                }
                wyh1I[((FeedForwardController)_controller)._hiddenLayer1.Length].Gradient += yGrad;
            }

            //Wuh1 error backpropagation
            for (int i = 0; i < _wuh1.Length; i++)
            {
                for (int j = 0; j < _heads[i].GetUnitSize(); j++)
                {
                    Unit headUnit = _heads[i][j];
                    Unit[] wuh1ij = _wuh1[i][j];
                    for (int k = 0; k < ((FeedForwardController)_controller)._hiddenLayer1.Length; k++)
                    {
                        Unit unit = ((FeedForwardController)_controller)._hiddenLayer1[k];
                        wuh1ij[k].Gradient += headUnit.Gradient * unit.Value;
                    }
                    wuh1ij[((FeedForwardController)_controller)._hiddenLayer1.Length].Gradient += headUnit.Gradient;
                }
            }

            double[] hiddenGradients = new double[((FeedForwardController)_controller)._hiddenLayer1.Length];
            for (int i = 0; i < ((FeedForwardController)_controller)._hiddenLayer1.Length; i++)
            {
                Unit unit = ((FeedForwardController)_controller)._hiddenLayer1[i];
                hiddenGradients[i] = unit.Gradient * unit.Value * (1 - unit.Value);
            }
            
            _controller.BackwardErrorPropagation(hiddenGradients, _input, _reads);
        }
    }
}
