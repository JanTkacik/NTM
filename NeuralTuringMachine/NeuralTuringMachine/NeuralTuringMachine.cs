using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AForge.Neuro;
using NeuralTuringMachine.Memory;

namespace NeuralTuringMachine
{
    public class NeuralTuringMachine
    {
        private readonly int _inputCount;
        private readonly int _outputCount;
        private readonly int _readHeadCount;
        private readonly int _writeHeadCount;
        private readonly int _hiddenNeuronsCount;
        private readonly int _hiddenLayersCount;
        private readonly int _memoryCellCount;
        private readonly int _memoryVectorLength;
        private readonly int _maxConvolutialShift;
        //INPUT IS IN ORDER "Input" "ReadHead1" "ReadHead2" ... "ReadHeadN"
        //OUTPUT IS IN ORDER "Output" "ReadHead1" "ReadHead2" ... "ReadHeadN" "WriteHead1" "WriteHead2" ... "WriteHeadN"
        //HEAD ADDRESSING DATA IS IN ORDER "KeyVector" "beta" "g" "s-vector" "gama"
        private readonly Network _controller;
        private readonly NtmMemory _memory;
        private readonly List<ReadHead> _readHeads;
        private readonly List<WriteHead> _writeHeads;
        private readonly int _inputsCount;

        //TODO REFACTOR
        public NeuralTuringMachine(
            int inputCount,
            int outputCount, 
            int readHeadCount, 
            int writeHeadCount, 
            int hiddenNeuronsCount, 
            int hiddenLayersCount, 
            int memoryCellCount, 
            int memoryVectorLength,
            int maxConvolutialShift)
        {
            _inputCount = inputCount;
            _outputCount = outputCount;
            _readHeadCount = readHeadCount;
            _writeHeadCount = writeHeadCount;
            _hiddenNeuronsCount = hiddenNeuronsCount;
            _hiddenLayersCount = hiddenLayersCount;
            _memoryCellCount = memoryCellCount;
            _memoryVectorLength = memoryVectorLength;
            _maxConvolutialShift = maxConvolutialShift;
            _readHeads = new List<ReadHead>(readHeadCount);
            _writeHeads = new List<WriteHead>(writeHeadCount);

            InitializeReadHeads();
            InitializeWriteHeads();
            List<int> neuronsCounts = GetNeuronsCount();
            _inputsCount = inputCount + (readHeadCount * memoryVectorLength);

            _controller = new ActivationNetwork(new SigmoidFunction(), _inputsCount, neuronsCounts.ToArray());
            _memory = new NtmMemory(memoryCellCount, memoryVectorLength);
        }

        private NeuralTuringMachine(
            int inputCount,
            int outputCount,
            int readHeadCount,
            int writeHeadCount,
            int hiddenNeuronsCount,
            int hiddenLayersCount,
            int memoryCellCount,
            int memoryVectorLength,
            int maxConvolutialShift,
            Network controller,
            NtmMemory memory)
        {
            _inputCount = inputCount;
            _outputCount = outputCount;
            _readHeadCount = readHeadCount;
            _writeHeadCount = writeHeadCount;
            _hiddenNeuronsCount = hiddenNeuronsCount;
            _hiddenLayersCount = hiddenLayersCount;
            _memoryCellCount = memoryCellCount;
            _memoryVectorLength = memoryVectorLength;
            _maxConvolutialShift = maxConvolutialShift;
            _readHeads = new List<ReadHead>(readHeadCount);
            _writeHeads = new List<WriteHead>(writeHeadCount);
            
            InitializeReadHeads();
            InitializeWriteHeads();
            _inputsCount = inputCount + (readHeadCount * memoryVectorLength);

            _controller = controller;
            _memory = memory;
        }

        public ActivationNetwork Controller
        {
            get { return (ActivationNetwork)_controller; }
        }

        private List<int> GetNeuronsCount()
        {
            int outputNeuronsCount = _outputCount + _readHeads.Sum(head => head.OutputNeuronCount) +
                                     _writeHeads.Sum(head => head.OutputNeuronCount);
            List<int> neuronsCounts = new List<int>(_hiddenNeuronsCount + 1);
            for (int i = 0; i < _hiddenLayersCount; i++)
            {
                neuronsCounts.Add(_hiddenNeuronsCount/_hiddenLayersCount);
            }
            neuronsCounts.Add(outputNeuronsCount);
            return neuronsCounts;
        }

        private void InitializeWriteHeads()
        {
            int writeHeadOffset = _outputCount + _readHeads.Sum(head => head.OutputNeuronCount);

            for (int i = 0; i < _writeHeadCount; i++)
            {
                _writeHeads.Add(new WriteHead(_memoryCellCount, _memoryVectorLength, i, writeHeadOffset, _maxConvolutialShift));
            }
        }

        private void InitializeReadHeads()
        {
            for (int i = 0; i < _readHeadCount; i++)
            {
                _readHeads.Add(new ReadHead(_memoryCellCount, _memoryVectorLength, i, _outputCount, _maxConvolutialShift));
            }
        }

        public double[] Compute(double[] input)
        {
            UpdateMemory(_controller.Output);

            double[] ntmInput = GetInputForController(input, _controller.Output);

            double[] output = _controller.Compute(ntmInput);
            
            var ntmOutput = GetOutput(output);

            return ntmOutput;
        }

        //USAGE FOR BPTT
        internal double[] Compute(double[] input, NeuralTuringMachine previousMachine)
        {
            double[] previousControllerOutput = previousMachine._controller.Output;
            UpdateMemory(previousControllerOutput);

            double[] ntmInput = GetInputForController(input, previousControllerOutput);

            double[] output = _controller.Compute(ntmInput);

            var ntmOutput = GetOutput(output);

            return ntmOutput;
        }

        private void UpdateMemory(double[] controllerOutput)
        {
            if (controllerOutput != null)
            {
                foreach (WriteHead writeHead in _writeHeads)
                {
                    writeHead.UpdateAddressingData(controllerOutput);
                    writeHead.UpdateEraseVector(controllerOutput);
                    writeHead.UpdateAddVector(controllerOutput);
                    writeHead.UpdateMemory(_memory);
                }
            }
        }

        private double[] GetOutput(double[] output)
        {
            double[] ntmOutput = new double[_outputCount];
            Array.Copy(output, ntmOutput, _outputCount);
            return ntmOutput;
        }

        public double[] GetInputForController(double[] input, double[] controllerOutput)
        {
            double[] ntmInput = new double[_inputsCount];
            Array.Copy(input, ntmInput, input.Length);
            int actualOffset = input.Length;
            if (controllerOutput != null)
            {
                foreach (ReadHead readHead in _readHeads)
                {
                    readHead.UpdateAddressingData(controllerOutput);
                    double[] vectorFromMemory = readHead.GetVectorFromMemory(_memory);
                    Array.Copy(vectorFromMemory, 0, ntmInput, actualOffset, vectorFromMemory.Length);
                    actualOffset += vectorFromMemory.Length;
                }
            }
            return ntmInput;
        }

        public NeuralTuringMachine Clone()
        {
            MemoryStream controllerStream = new MemoryStream();
            _controller.Save(controllerStream);
            controllerStream.Seek(0, SeekOrigin.Begin);
            Network networkClone = Network.Load(controllerStream);

            NtmMemory memoryClone = _memory.Clone();

            return new NeuralTuringMachine(
                _inputCount, 
                _outputCount, 
                _readHeadCount, 
                _writeHeadCount, 
                _hiddenNeuronsCount, 
                _hiddenLayersCount, 
                _memoryCellCount, 
                _memoryVectorLength, 
                _maxConvolutialShift, 
                networkClone, 
                memoryClone);
        }
    }
}
