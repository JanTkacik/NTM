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
        //INPUT IS IN ORDER "Input" "ReadHead1" "ReadHead2" ... "ReadHeadN"
        //OUTPUT IS IN ORDER "Output" "ReadHead1" "ReadHead2" ... "ReadHeadN" "WriteHead1" "WriteHead2" ... "WriteHeadN"
        //HEAD ADDRESSING DATA IS IN ORDER "KeyVector" "beta" "g" "s-vector" "gama"
        private readonly Network _controller;
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
        
        public NtmMemory Memory { get; set; }

        public int InputCount
        {
            get { return _inputCount; }
        }
        
        public int ReadHeadLength { get; private set; }
        
        public int WriteHeadLength { get; private set; }
        
        public int MaxConvolutialShift { get; private set; }
        
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
            MaxConvolutialShift = maxConvolutialShift;
            _readHeads = new List<ReadHead>(readHeadCount);
            _writeHeads = new List<WriteHead>(writeHeadCount);

            InitializeReadHeads();
            ReadHeadLength = _readHeads[0].OutputNeuronCount;
            InitializeWriteHeads();
            WriteHeadLength = _writeHeads[0].OutputNeuronCount;

            List<int> neuronsCounts = GetNeuronsCount();
            _controllerInputCount = inputCount + (readHeadCount * memoryVectorLength);

            _controller = new ActivationNetwork(new SigmoidFunction(), _controllerInputCount, neuronsCounts.ToArray());
            Memory = new NtmMemory(memoryCellCount, memoryVectorLength);
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
            NtmMemory memory,
            List<ReadHead> readHeads,
            List<WriteHead> writeHeads,
            ControllerOutput lastControllerOutput)
        {
            _inputCount = inputCount;
            _outputCount = outputCount;
            _readHeadCount = readHeadCount;
            _writeHeadCount = writeHeadCount;
            _hiddenNeuronsCount = hiddenNeuronsCount;
            _hiddenLayersCount = hiddenLayersCount;
            _memoryCellCount = memoryCellCount;
            _memoryVectorLength = memoryVectorLength;
            MaxConvolutialShift = maxConvolutialShift;
            _readHeads = new List<ReadHead>(readHeadCount);
            _writeHeads = new List<WriteHead>(writeHeadCount);

            _readHeads = readHeads;
            ReadHeadLength = _readHeads[0].OutputNeuronCount;
            _writeHeads = writeHeads;
            WriteHeadLength = _writeHeads[0].OutputNeuronCount;

            _controllerInputCount = inputCount + (readHeadCount * memoryVectorLength);

            _controller = controller;
            Memory = memory;

            LastControllerOutput = lastControllerOutput;
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
            for (int i = 0; i < _writeHeadCount; i++)
            {
                WriteHead writeHead = new WriteHead(_memoryCellCount, _memoryVectorLength, MaxConvolutialShift);
                _writeHeads.Add(writeHead);
            }
        }

        private void InitializeReadHeads()
        {
            for (int i = 0; i < _readHeadCount; i++)
            {
                ReadHead readHead = new ReadHead(_memoryCellCount, _memoryVectorLength, MaxConvolutialShift);
                _readHeads.Add(readHead);
            }
        }

        public double[] Compute(double[] input)
        {
            UpdateMemory(LastControllerOutput);

            ControllerInput ntmInput = GetInputForController(input, LastControllerOutput);

            LastControllerOutput = new ControllerOutput(_controller.Compute(ntmInput.Input), _outputCount, _readHeadCount, ReadHeadLength, _writeHeadCount, WriteHeadLength);

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
                    writeHead.UpdateMemory(Memory);
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
                    readHeadOutputs[i] = readHead.GetVectorFromMemory(Memory);
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
            NtmMemory memoryClone = Memory.Clone();

            List<ReadHead> readHeadsClone = new List<ReadHead>();
            foreach (ReadHead head in _readHeads)
            {
                readHeadsClone.Add(head.Clone());
            }

            List<WriteHead> writeHeadClone = new List<WriteHead>();
            foreach (WriteHead head in _writeHeads)
            {
                writeHeadClone.Add(head.Clone());
            }

            ControllerOutput controllerOutputClone = null;
            if (LastControllerOutput != null)
            {
                controllerOutputClone = LastControllerOutput.Clone();
            }

            return new NeuralTuringMachine(
                _inputCount, 
                _outputCount, 
                _readHeadCount, 
                _writeHeadCount, 
                _hiddenNeuronsCount, 
                _hiddenLayersCount, 
                _memoryCellCount, 
                _memoryVectorLength, 
                MaxConvolutialShift, 
                networkClone, 
                memoryClone,
                readHeadsClone,
                writeHeadClone,
                controllerOutputClone);
        }

        public ReadHead GetReadHead(int i)
        {
            return _readHeads[i];
        }

        public WriteHead GetWriteHead(int i)
        {
            return _writeHeads[i];
        }
    }
}
