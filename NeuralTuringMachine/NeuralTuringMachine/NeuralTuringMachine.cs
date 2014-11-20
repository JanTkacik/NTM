using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AForge.Neuro;
using NeuralTuringMachine.Controller;
using NeuralTuringMachine.Memory;
using NeuralTuringMachine.Memory.Head;

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
        private int _readHeadLength;
        private int _writeHeadLength;
        //INPUT IS IN ORDER "Input" "ReadHead1" "ReadHead2" ... "ReadHeadN"
        //OUTPUT IS IN ORDER "Output" "ReadHead1" "ReadHead2" ... "ReadHeadN" "WriteHead1" "WriteHead2" ... "WriteHeadN"
        //HEAD ADDRESSING DATA IS IN ORDER "KeyVector" "beta" "g" "s-vector" "gama"
        private readonly Network _controller;
        private readonly NtmMemory _memory;
        private readonly List<ReadHead> _readHeads;
        private readonly List<WriteHead> _writeHeads;
        private readonly int _controllerInputCount;

        public ControllerOutput LastControllerOutput { get; private set; }

        public ActivationNetwork Controller
        {
            get
            {
                return (ActivationNetwork)_controller;
            }
        }

        public int InputCount
        {
            get { return _inputCount; }
        }

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
            _controllerInputCount = inputCount + (readHeadCount * memoryVectorLength);

            _controller = new ActivationNetwork(new SigmoidFunction(), _controllerInputCount, neuronsCounts.ToArray());
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
            _controllerInputCount = inputCount + (readHeadCount * memoryVectorLength);

            _controller = controller;
            _memory = memory;
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
                WriteHead writeHead = new WriteHead(_memoryCellCount, _memoryVectorLength, i, _maxConvolutialShift);
                _writeHeads.Add(writeHead);
                if (i == 0)
                {
                    _writeHeadLength = writeHead.OutputNeuronCount;
                }
            }
        }

        private void InitializeReadHeads()
        {
            for (int i = 0; i < _readHeadCount; i++)
            {
                ReadHead readHead = new ReadHead(_memoryCellCount, _memoryVectorLength, i, _maxConvolutialShift);
                _readHeads.Add(readHead);
                if (i == 0)
                {
                    _readHeadLength = readHead.OutputNeuronCount;
                }
            }
        }

        public double[] Compute(double[] input)
        {
            UpdateMemory(LastControllerOutput);

            ControllerInput ntmInput = GetInputForController(input, LastControllerOutput);

            LastControllerOutput = new ControllerOutput(_controller.Compute(ntmInput.Input), _outputCount, _readHeadCount, _readHeadLength, _writeHeadCount, _writeHeadLength);

            return LastControllerOutput.DataOutput;
        }

        //USAGE FOR BPTT
        internal double[] Compute(double[] input, NeuralTuringMachine previousMachine)
        {
            UpdateMemory(previousMachine.LastControllerOutput);

            ControllerInput ntmInput = GetInputForController(input, previousMachine.LastControllerOutput);

            LastControllerOutput = new ControllerOutput(_controller.Compute(ntmInput.Input), _outputCount, _readHeadCount, _readHeadLength, _writeHeadCount, _writeHeadLength);

            return LastControllerOutput.DataOutput;
        }

        private void UpdateMemory(ControllerOutput controllerOutput)
        {
            if (controllerOutput != null)
            {
                for (int i = 0; i < _writeHeads.Count; i++)
                {
                    WriteHead writeHead = _writeHeads[i];
                    writeHead.UpdateAddressingData(controllerOutput.WriteHeadsOutputs[i]);
                    writeHead.UpdateEraseVector(controllerOutput.WriteHeadsOutputs[i]);
                    writeHead.UpdateAddVector(controllerOutput.WriteHeadsOutputs[i]);
                    writeHead.UpdateMemory(_memory);
                }
            }
        }

        public ControllerInput GetInputForController(double[] input, ControllerOutput controllerOutput)
        {
            if (controllerOutput != null)
            {
                double[][] readHeadOutputs = new double[_readHeadCount][];
                for (int i = 0; i < _readHeadCount; i++)
                {
                    ReadHead readHead = _readHeads[i];
                    readHead.UpdateAddressingData(controllerOutput.ReadHeadsOutputs[i]);
                    readHeadOutputs[i] = readHead.GetVectorFromMemory(_memory);
                }
                return new ControllerInput(input, readHeadOutputs, _controller.InputsCount);
            }
            return new ControllerInput(input, _controller.InputsCount);
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
