using System;
using NTM2.Memory;
using NTM2.Memory.Addressing;

namespace NTM2.Controller
{
    public class TrainableNTM
    {
        private readonly NeuralTuringMachine _controller;
        private readonly MemoryState _oldMemoryState;

        private MemoryState _memoryState;
        private readonly ContentAddressing[] _contentAddressings;
        private readonly HeadSetting[] _headSettings;
        private readonly ReadData[] _readDatas;

        public TrainableNTM(NeuralTuringMachine controller)
        {
            _controller = controller;

            //FOREACH HEAD - SET WEIGHTS TO BIAS VALUES
            _contentAddressings = _controller.Memory.GetContentAddressing();
            _headSettings = HeadSetting.GetVector(_controller.HeadCount, i => new Tuple<int, ContentAddressing>(_controller.Memory.MemoryColumnsN, _contentAddressings[i]), _controller.UnitFactory);
            _readDatas = ReadData.GetVector(_controller.HeadCount, i => new Tuple<HeadSetting, NTMMemory>(_headSettings[i], _controller.Memory));

            _memoryState = new MemoryState(_headSettings, _readDatas, _controller.Memory);
        }

        public TrainableNTM(TrainableNTM oldMachine)
        {
            _controller = oldMachine._controller.Clone();
            _oldMemoryState = oldMachine._memoryState;
        }

        public void ForwardPropagation(double[] input)
        {
            _controller.ForwardPropagation(_oldMemoryState.ReadData, input);

            for (int i = 0; i < _controller.HeadCount; i++)
            {
                _controller.HeadsNeurons[i].OldHeadSettings = _oldMemoryState.HeadSettings[i];
            }
            _memoryState = new MemoryState(_controller.HeadsNeurons, _oldMemoryState.Memory, _controller.UnitFactory);
        }
        
        public void BackwardErrorPropagation(double[] knownOutput)
        {
            _memoryState.BackwardErrorPropagation();
            _controller.BackwardErrorPropagation(knownOutput);
        }

        public void BackwardErrorPropagation()
        {
            //Compute gradients for the bias values of internal memory and weights
            for (int i = 0; i < _readDatas.Length; i++)
            {
                _readDatas[i].BackwardErrorPropagation();
                for (int j = 0; j < _readDatas[i].HeadSetting.Data.Length; j++)
                {
                    _contentAddressings[i].Data[j].Gradient += _readDatas[i].HeadSetting.Data[j].Gradient;
                }
                _contentAddressings[i].BackwardErrorPropagation();
            }
        }

        public double[] GetOutput()
        {
            return _controller.GetOutput();
        }
    }
}
