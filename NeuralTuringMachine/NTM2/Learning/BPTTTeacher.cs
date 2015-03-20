using System;
using NTM2.Controller;
using NTM2.Memory;
using NTM2.Memory.Addressing;

namespace NTM2.Learning
{
    public class BPTTTeacher : INTMTeacher
    {
        private readonly NeuralTuringMachine _controller;
        private readonly IWeightUpdater _weightUpdater;
        private readonly IWeightUpdater _gradientResetter;

        public BPTTTeacher(NeuralTuringMachine controller, IWeightUpdater weightUpdater)
        {
            _controller = controller;
            _weightUpdater = weightUpdater;
            _gradientResetter = new GradientResetter();
        }

        public double[][] Train(double[][] input, double[][] knownOutput)
        {
            //FOREACH HEAD - SET WEIGHTS TO BIAS VALUES
            ContentAddressing[] contentAddressings = _controller.Memory.GetContentAddressing();
            HeadSetting[] oldSettings = HeadSetting.GetVector(_controller.HeadCount, i => new Tuple<int, ContentAddressing>(_controller.Memory.MemoryColumnsN, contentAddressings[i]), _controller.UnitFactory);
            ReadData[] readDatas = ReadData.GetVector(_controller.HeadCount, i => new Tuple<HeadSetting, NTMMemory>(oldSettings[i], _controller.Memory));

            TrainableNTM[] machines = new TrainableNTM[input.Length];
            TrainableNTM empty = new TrainableNTM(_controller, new MemoryState(oldSettings, readDatas, _controller.Memory));

            //BPTT
            machines[0] = new TrainableNTM(empty, input[0], _controller.UnitFactory);

            for (int i = 1; i < input.Length; i++)
            {
                machines[i] = new TrainableNTM(machines[i - 1], input[i], _controller.UnitFactory);
            }

            _gradientResetter.Reset();
            _controller.UpdateWeights(_gradientResetter);

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

            _weightUpdater.Reset();
            _controller.UpdateWeights(_weightUpdater);
            
            return GetMachineOutputs(machines);
        }

        private double[][] GetMachineOutputs(TrainableNTM[] machines)
        {
            double[][] realOutputs = new double[machines.Length][];
            for (int i = 0; i < machines.Length; i++)
            {
                TrainableNTM machine = machines[i];
                realOutputs[i] = machine.Controller.GetOutput();
            }
            return realOutputs;
        }

        public double[][][] Train(double[][][] inputs, double[][][] knownOutputs)
        {
            throw new NotImplementedException();
        }
    }
}
