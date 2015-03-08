﻿using System;
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
        private readonly double[] _input;
        private readonly ReadData[] _reads;
        private readonly NTMMemory _memory;
        
        private readonly IController _controller;

        //Old similarities
        private readonly BetaSimilarity[][] _wtm1s;

        public int WeightsCount
        {
            get { return _weightsCount; }
        }

        public int HeadCount
        {
            get { return ((FeedForwardController)_controller).OutputLayer._heads.Length; }
        }

        public Head[] Heads
        {
            get { return ((FeedForwardController)_controller).OutputLayer._heads; }
        }

        public Unit[] Output
        {
            get { return ((FeedForwardController)_controller).OutputLayer._outputLayer; }
        }

        public NTMController(int inputSize, int outputSize, int controllerSize, int headCount, int memoryColumnsN, int memoryRowsM)
        {
            _unitFactory = new UnitFactory();
            _memoryColumnsN = memoryColumnsN;
            _memoryRowsM = memoryRowsM;
            int headUnitSize = Head.GetUnitSize(memoryRowsM);
            _wtm1s = BetaSimilarity.GetTensor2(headCount, memoryColumnsN);
            _memory = new NTMMemory(memoryColumnsN, memoryRowsM, _unitFactory);
            
            _controller = new FeedForwardController(controllerSize, inputSize, outputSize, headCount, memoryRowsM, _unitFactory);
            
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
            int weightsCount,
            ReadData[] readDatas,
            double[] input,
            IController controller,
            UnitFactory unitFactory)
        {
            _unitFactory = unitFactory;
            _memoryColumnsN = memoryColumnsN;
            _memoryRowsM = memoryRowsM;
            _weightsCount = weightsCount;
            _reads = readDatas;
            _input = input;
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
                    ((FeedForwardController)(machine.Controller._controller)).OutputLayer._outputLayer[j].Gradient = ((FeedForwardController)(machine.Controller._controller)).OutputLayer._outputLayer[j].Value - output[j];
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
                _weightsCount,
                readData,
                input,
                _controller.Clone(),
                _unitFactory);

            newController.ForwardPropagation(readData, input);
            return newController;
        }

        //TODO readData Units are maybe not important
        private void ForwardPropagation(ReadData[] readData, double[] input)
        {
            _controller.ForwardPropagation(input, readData);

            //Foreach neuron in classic output layer
            for (int i = 0; i < ((FeedForwardController)_controller).OutputLayer._wyh1.Length; i++)
            {
                double sum = 0;
                Unit[] weights = ((FeedForwardController)_controller).OutputLayer._wyh1[i];

                //Foreach input from hidden layer
                for (int j = 0; j < ((FeedForwardController)_controller).HiddenLayer.HiddenLayerNeurons.Length; j++)
                {
                    sum += weights[j].Value * ((FeedForwardController)_controller).HiddenLayer.HiddenLayerNeurons[j].Value;
                }

                //Plus threshold
                sum += weights[((FeedForwardController)_controller).HiddenLayer.HiddenLayerNeurons.Length].Value;
                ((FeedForwardController)_controller).OutputLayer._outputLayer[i].Value = Sigmoid.GetValue(sum);
            }

            //Foreach neuron in head output layer
            for (int i = 0; i < ((FeedForwardController)_controller).OutputLayer._wuh1.Length; i++)
            {
                Unit[][] headsWeights = ((FeedForwardController)_controller).OutputLayer._wuh1[i];
                Head head = ((FeedForwardController)_controller).OutputLayer._heads[i];

                for (int j = 0; j < headsWeights.Length; j++)
                {
                    double sum = 0;
                    Unit[] headWeights = headsWeights[j];
                    //Foreach input from hidden layer
                    for (int k = 0; k < ((FeedForwardController)_controller).HiddenLayer.HiddenLayerNeurons.Length; k++)
                    {
                        sum += headWeights[k].Value * ((FeedForwardController)_controller).HiddenLayer.HiddenLayerNeurons[k].Value;
                    }
                    //Plus threshold
                    sum += headWeights[((FeedForwardController)_controller).HiddenLayer.HiddenLayerNeurons.Length].Value;
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

            tensor2UpdateAction(_memory.Data);
            
            _controller.UpdateWeights(updateAction);
        }
        
        public void BackwardErrorPropagation()
        {
            //Output error backpropagation
            for (int j = 0; j < ((FeedForwardController)_controller).OutputLayer._outputLayer.Length; j++)
            {
                Unit unit = ((FeedForwardController)_controller).OutputLayer._outputLayer[j];
                Unit[] weights = ((FeedForwardController)_controller).OutputLayer._wyh1[j];
                for (int i = 0; i < ((FeedForwardController)_controller).HiddenLayer.HiddenLayerNeurons.Length; i++)
                {
                    ((FeedForwardController)_controller).HiddenLayer.HiddenLayerNeurons[i].Gradient += weights[i].Value * unit.Gradient;
                }
            }

            //Heads error backpropagation
            for (int j = 0; j < ((FeedForwardController)_controller).OutputLayer._heads.Length; j++)
            {
                Head head = ((FeedForwardController)_controller).OutputLayer._heads[j];
                Unit[][] weights = ((FeedForwardController)_controller).OutputLayer._wuh1[j];
                for (int k = 0; k < head.GetUnitSize(); k++)
                {
                    Unit unit = head[k];
                    Unit[] weightsK = weights[k];
                    for (int i = 0; i < ((FeedForwardController)_controller).HiddenLayer.HiddenLayerNeurons.Length; i++)
                    {
                        ((FeedForwardController)_controller).HiddenLayer.HiddenLayerNeurons[i].Gradient += unit.Gradient * weightsK[i].Value;
                    }
                }
            }

            //Wyh1 error backpropagation
            for (int i = 0; i < ((FeedForwardController)_controller).OutputLayer._wyh1.Length; i++)
            {
                Unit[] wyh1I = ((FeedForwardController)_controller).OutputLayer._wyh1[i];
                double yGrad = ((FeedForwardController)_controller).OutputLayer._outputLayer[i].Gradient;
                for (int j = 0; j < ((FeedForwardController)_controller).HiddenLayer.HiddenLayerNeurons.Length; j++)
                {
                    wyh1I[j].Gradient += yGrad * ((FeedForwardController)_controller).HiddenLayer.HiddenLayerNeurons[j].Value;
                }
                wyh1I[((FeedForwardController)_controller).HiddenLayer.HiddenLayerNeurons.Length].Gradient += yGrad;
            }

            //Wuh1 error backpropagation
            for (int i = 0; i < ((FeedForwardController)_controller).OutputLayer._wuh1.Length; i++)
            {
                for (int j = 0; j < ((FeedForwardController)_controller).OutputLayer._heads[i].GetUnitSize(); j++)
                {
                    Unit headUnit = ((FeedForwardController)_controller).OutputLayer._heads[i][j];
                    Unit[] wuh1ij = ((FeedForwardController)_controller).OutputLayer._wuh1[i][j];
                    for (int k = 0; k < ((FeedForwardController)_controller).HiddenLayer.HiddenLayerNeurons.Length; k++)
                    {
                        Unit unit = ((FeedForwardController)_controller).HiddenLayer.HiddenLayerNeurons[k];
                        wuh1ij[k].Gradient += headUnit.Gradient * unit.Value;
                    }
                    wuh1ij[((FeedForwardController)_controller).HiddenLayer.HiddenLayerNeurons.Length].Gradient += headUnit.Gradient;
                }
            }
            
            _controller.BackwardErrorPropagation(_input, _reads);
        }
    }
}
