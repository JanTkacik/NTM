using System;
using NTM2.Memory;
using NTM2.Memory.Addressing;

namespace NTM2.Controller
{
    public class TrainableNTM
    {
        private readonly NeuralTuringMachine _controller;
        private readonly MemoryState _memoryState;
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

        public TrainableNTM(TrainableNTM oldMachine, double[] input, UnitFactory unitFactory)
        {
            _controller = oldMachine._controller.Process(oldMachine._memoryState.ReadData, input);
            for (int i = 0; i < _controller.HeadCount; i++)
            {
                _controller.HeadsNeurons[i].OldHeadSettings = oldMachine._memoryState.HeadSettings[i];
            }
            _memoryState = new MemoryState(_controller.HeadsNeurons, oldMachine._memoryState.Memory, unitFactory);
        }

        public NeuralTuringMachine Controller
        {
            get { return _controller; }
        }

        public void BackwardErrorPropagation(double[] knownOutput)
        {
            _memoryState.BackwardErrorPropagation();
            _controller.BackwardErrorPropagation(knownOutput);
        }

        public void DataBackwardPropagation()
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
    }
}
