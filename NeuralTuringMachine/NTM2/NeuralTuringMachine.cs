using System;
using NTM2.Controller;
using NTM2.Learning;
using NTM2.Memory;
using NTM2.Memory.Addressing;

namespace NTM2
{
    public class NeuralTuringMachine
    {
        private readonly UnitFactory _unitFactory;
        
        private readonly int _memoryColumnsN;
        private readonly int _memoryRowsM;
        private readonly int _weightsCount;

        private readonly double[] _input;
        private readonly ReadData[] _reads;

        private readonly NTMMemory _memory;
        private readonly IController _controller;
        
        public int WeightsCount
        {
            get { return _weightsCount; }
        }

        public int HeadCount
        {
            get { return ((FeedForwardController)_controller).OutputLayer.HeadsNeurons.Length; }
        }

        public Head[] HeadsNeurons
        {
            get { return ((FeedForwardController)_controller).OutputLayer.HeadsNeurons; }
        }
        
        public NeuralTuringMachine(int inputSize, int outputSize, int controllerSize, int headCount, int memoryColumnsN, int memoryRowsM)
        {
            _unitFactory = new UnitFactory();
            _memoryColumnsN = memoryColumnsN;
            _memoryRowsM = memoryRowsM;
            int headUnitSize = Head.GetUnitSize(memoryRowsM);
            _memory = new NTMMemory(memoryColumnsN, memoryRowsM, headCount, _unitFactory);
            
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

        private NeuralTuringMachine(
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
            ContentAddressing[] contentAddressings = _memory.GetContentAddressing();
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
                machines[i].BackwardErrorPropagation(knownOutput[i]);
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

        public NeuralTuringMachine Process(ReadData[] readData, double[] input)
        {
            NeuralTuringMachine newController = new NeuralTuringMachine(
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
        
        private void ForwardPropagation(ReadData[] readData, double[] input)
        {
            _controller.ForwardPropagation(input, readData);
        }

        public void UpdateWeights(Action<Unit> updateAction)
        {
            _memory.UpdateWeights(updateAction);
            _controller.UpdateWeights(updateAction);
        }
        
        public void BackwardErrorPropagation(double[] knownOutput)
        {
            _controller.BackwardErrorPropagation(knownOutput, _input, _reads);
        }

        public void UpdateWeights(IWeightUpdater weightUpdater)
        {
            _memory.UpdateWeights(weightUpdater);
            _controller.UpdateWeights(weightUpdater);
        }

        public double[] GetOutput()
        {
            return _controller.GetOutput();
        }
    }
}
