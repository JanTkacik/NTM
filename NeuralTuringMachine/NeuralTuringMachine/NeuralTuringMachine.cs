using System;
using System.Collections.Generic;
using System.Linq;
using AForge.Neuro;
using NeuralTuringMachine.Memory;

namespace NeuralTuringMachine
{
    public class NeuralTuringMachine
    {
        private readonly int _outputCount;
        //INPUT IS IN ORDER "Input" "ReadHead1" "ReadHead2" ... "ReadHeadN"
        //OUTPUT IS IN ORDER "Output" "ReadHead1" "ReadHead2" ... "ReadHeadN" "WriteHead1" "WriteHead2" ... "WriteHeadN"
        //HEAD ADDRESSING DATA IS IN ORDER "KeyVector" "beta" "g" "s" "gama"
        private readonly ActivationNetwork _controller;
        private readonly NTMMemory _memory;
        private readonly List<ReadHead> _readHeads;
        private readonly List<WriteHead> _writeHeads;
        private readonly int _inputsCount;

        public NeuralTuringMachine(int inputCount, int outputCount, int readHeadCount, int writeHeadCount, int hiddenNeuronsCount, int hiddenLayersCount, int memoryCellCount, int memoryVectorLength)
        {
            _outputCount = outputCount;
            _readHeads = new List<ReadHead>(readHeadCount);
            _writeHeads = new List<WriteHead>(writeHeadCount);

            int writeHeadOffset = inputCount + _readHeads.Sum(head => head.OutputNeuronCount);

            for (int i = 0; i < readHeadCount; i++)
            {
                _readHeads.Add(new ReadHead(memoryVectorLength, i, inputCount));
            }

            for (int i = 0; i < writeHeadCount; i++)
            {
                _writeHeads.Add(new WriteHead(memoryVectorLength, i, writeHeadOffset));
            }

            int outputNeuronsCount = outputCount + _readHeads.Sum(head => head.OutputNeuronCount) + _writeHeads.Sum(head => head.OutputNeuronCount);
            List<int> neuronsCounts = new List<int>(hiddenNeuronsCount + 1);
            for (int i = 0; i < hiddenLayersCount; i++)
            {
                neuronsCounts.Add(hiddenNeuronsCount / hiddenLayersCount);
            }
            neuronsCounts.Add(outputNeuronsCount);

            _inputsCount = inputCount + (readHeadCount * memoryVectorLength);
            _controller = new ActivationNetwork(new SigmoidFunction(), _inputsCount, neuronsCounts.ToArray());
            _memory = new NTMMemory(memoryCellCount, memoryVectorLength);
        }

        public double[] Compute(double[] input)
        {
            double[] ntmInput = GetInput(input);

            double[] output = _controller.Compute(ntmInput);

            UpdateMemory();

            var ntmOutput = GetOutput(output);

            return ntmOutput;
        }

        private void UpdateMemory()
        {
            foreach (WriteHead writeHead in _writeHeads)
            {
                writeHead.UpdateAddressingData(_controller.Output);
                writeHead.UpdateEraseVector(_controller.Output);
                writeHead.UpdateAddVector(_controller.Output);
                writeHead.UpdateMemory(_memory);
            }
        }

        private double[] GetOutput(double[] output)
        {
            double[] ntmOutput = new double[_outputCount];
            Array.Copy(output, ntmOutput, _outputCount);
            return ntmOutput;
        }

        private double[] GetInput(double[] input)
        {
            double[] ntmInput = new double[_inputsCount];
            Array.Copy(input, ntmInput, input.Length);
            int actualOffset = input.Length;
            foreach (ReadHead readHead in _readHeads)
            {
                readHead.UpdateAddressingData(_controller.Output);
                double[] vectorFromMemory = readHead.GetVectorFromMemory(_memory);
                Array.Copy(vectorFromMemory, 0, ntmInput, actualOffset, vectorFromMemory.Length);
                actualOffset += vectorFromMemory.Length;
            }
            return ntmInput;
        }
    }
}
