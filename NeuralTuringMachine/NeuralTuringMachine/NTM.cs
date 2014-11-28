using System.Collections.Generic;
using System.IO;
using AForge.Neuro;
using CsvHelper;
using NeuralTuringMachine.Controller;
using NeuralTuringMachine.Memory;
using NeuralTuringMachine.Memory.Head;

namespace NeuralTuringMachine
{
    public class NTM
    {
        private static readonly StreamWriter _streamWriter = new StreamWriter("NTMLog", true);
        private static readonly CsvWriter _logger = new CsvWriter(_streamWriter);

        private readonly int _inputCount;
        public int OutputCount { get; private set; }
        public int NTMId { get; private set; }
        //INPUT IS IN ORDER "Input" "ReadHead1" "ReadHead2" ... "ReadHeadN"
        //OUTPUT IS IN ORDER "Output" "ReadHead1" "ReadHead2" ... "ReadHeadN" "WriteHead1" "WriteHead2" ... "WriteHeadN"
        //HEAD ADDRESSING DATA IS IN ORDER "KeyVector" "beta" "g" "s-vector" "gama"
        private Network _controller;
        private readonly List<ReadHead> _readHeads;
        private readonly List<WriteHead> _writeHeads;
        private readonly int _controllerInputCount;
        public long Time { get; private set; }

        public ControllerOutput LastControllerOutput { get; private set; }

        public ActivationNetwork Controller
        {
            get
            {
                return (ActivationNetwork)_controller;
            }
        }

        public void SetController(Network controller)
        {
            _controller = controller;
        }
        
        public NtmMemory Memory { get; set; }

        public int InputCount
        {
            get { return _inputCount; }
        }
        
        //TODO REFACTOR
        public NTM(
            int inputCount,
            int outputCount, 
            int hiddenNeuronsCount, 
            int hiddenLayersCount, 
            MemorySettings settings,
            int ntmId)
        {
            _inputCount = inputCount;
            OutputCount = outputCount;
            NTMId = ntmId;
            _readHeads = new List<ReadHead>(settings.ReadHeadCount);
            _writeHeads = new List<WriteHead>(settings.WriteHeadCount);

            Memory = new NtmMemory(settings);
            InitializeReadHeads();
            InitializeWriteHeads();

            List<int> neuronsCounts = GetNeuronsCount(hiddenLayersCount, hiddenNeuronsCount, settings);
            _controllerInputCount = inputCount + (settings.ReadHeadCount * settings.MemoryVectorLength);

            _controller = new ActivationNetwork(new SigmoidFunction(), _controllerInputCount, neuronsCounts.ToArray());
            Time = 0;
        }

        private NTM(
            int inputCount,
            int outputCount,
            Network controller,
            NtmMemory memory,
            List<ReadHead> readHeads,
            List<WriteHead> writeHeads,
            ControllerOutput lastControllerOutput,
            long actualTime, int ntmId)
        {
            _inputCount = inputCount;
            OutputCount = outputCount;

            _readHeads = readHeads;
            _writeHeads = writeHeads;
            
            _controller = controller;
            Memory = memory;

            LastControllerOutput = lastControllerOutput;
            Time = actualTime;
            NTMId = ntmId;
        }

        #region Initialization
        private List<int> GetNeuronsCount(int hiddenLayersCount, int hiddenNeuronsCount, MemorySettings settings)
        {
            int outputNeuronsCount = OutputCount + (settings.ReadHeadCount * settings.ReadHeadLength) + (settings.WriteHeadCount * settings.WriteHeadLength);
            List<int> neuronsCounts = new List<int>(hiddenLayersCount + 1);
            for (int i = 0; i < hiddenLayersCount; i++)
            {
                neuronsCounts.Add(hiddenNeuronsCount / hiddenLayersCount);
            }
            neuronsCounts.Add(outputNeuronsCount);
            return neuronsCounts;
        }

        private void InitializeWriteHeads()
        {
            MemorySettings memorySettings = Memory.MemorySettings;
            int writeHeadCount = memorySettings.WriteHeadCount;
            for (int i = 0; i < writeHeadCount; i++)
            {
                _writeHeads.Add(new WriteHead(memorySettings));
            }
        }

        private void InitializeReadHeads()
        {
            MemorySettings memorySettings = Memory.MemorySettings;
            int readHeadCount = memorySettings.ReadHeadCount;
            for (int i = 0; i < readHeadCount; i++)
            {
                _readHeads.Add(new ReadHead(memorySettings));
            }
        }
        
        #endregion

        #region Computation
        public double[] Compute(double[] input)
        {
            ControllerInput ntmInput = GetInputForController(input, LastControllerOutput);

            LastControllerOutput = new ControllerOutput(_controller.Compute(ntmInput.Input), OutputCount, Memory.MemorySettings);

            WriteCSVLog(_logger);
            ntmInput.WriteCSVLog(_logger);
            LastControllerOutput.WriteCSVLog(_logger);
            Memory.WriteCSVLog(_logger);
            _logger.NextRecord();
            _streamWriter.Flush();

            UpdateMemory(LastControllerOutput);

            Time++;

            return LastControllerOutput.DataOutput;
        }

        private void WriteCSVLog(CsvWriter logger)
        {
            logger.WriteField("NTM");
            logger.WriteField(NTMId);
            logger.WriteField(Time);
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
            int readHeadCount = Memory.MemorySettings.ReadHeadCount;
            if (controllerOutput != null)
            {
                double[][] readHeadOutputs = new double[readHeadCount][];
                for (int i = 0; i < readHeadCount; i++)
                {
                    ReadHead readHead = _readHeads[i];
                    readHead.UpdateAddressingData(controllerOutput.ReadHeadsOutputs[i]);
                    readHeadOutputs[i] = readHead.GetVectorFromMemory(Memory);
                }
                return new ControllerInput(input, readHeadOutputs, _controller.InputsCount);
            }
            return new ControllerInput(input, _controller.InputsCount);
        } 
        #endregion

        #region Clone
        public NTM Clone(int ntmId)
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

            return new NTM(
                _inputCount,
                OutputCount,
                networkClone,
                memoryClone,
                readHeadsClone,
                writeHeadClone,
                controllerOutputClone,
                Time, ntmId);
        } 
        #endregion

        #region Nothing to do here as public - TODO remove
        //TODO remove
        public ReadHead GetReadHead(int i)
        {
            return _readHeads[i];
        }

        public WriteHead GetWriteHead(int i)
        {
            return _writeHeads[i];
        }
        
        #endregion
    }
}
