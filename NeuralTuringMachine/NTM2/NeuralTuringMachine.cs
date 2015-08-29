using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;
using NTM2.Controller;
using NTM2.Learning;
using NTM2.Memory;
using NTM2.Memory.Addressing;

namespace NTM2
{
    [DataContract]
    public sealed class NeuralTuringMachine : INeuralTuringMachine
    {
        #region Fiels
        [DataMember(Name = "Controller")]
        private readonly FeedForwardController _controller;
        [DataMember(Name = "Memory")]
        private readonly NTMMemory _memory;
        [DataMember(Name = "OldMemoryState")]
        private MemoryState _oldMemoryState;
        [DataMember(Name = "NewMemoryState")]
        private MemoryState _newMemoryState;

        private double[] _lastInput; 
        #endregion
        
        #region Ctors

        internal NeuralTuringMachine(NeuralTuringMachine oldMachine)
        {
            _controller = oldMachine._controller.Clone();
            _memory = oldMachine._memory;
            _newMemoryState = oldMachine._newMemoryState;
            _oldMemoryState = oldMachine._oldMemoryState;
        }

        public NeuralTuringMachine(int inputSize, int outputSize, int controllerSize, int headCount, int memoryColumnsN, int memoryRowsM, IWeightUpdater initializer)
        {
            _memory = new NTMMemory(memoryColumnsN, memoryRowsM, headCount);
            _controller = new FeedForwardController(controllerSize, inputSize, outputSize, headCount, memoryRowsM);
            UpdateWeights(initializer);
        }
        
        #endregion

        #region Public Methods

        public void Process(double[] input)
        {
            _lastInput = input;
            _oldMemoryState = _newMemoryState;

            _controller.Process(input, _oldMemoryState.ReadData);
            _newMemoryState = _oldMemoryState.Process(GetHeads());
        }
        
        public double[] GetOutput()
        {
            return _controller.GetOutput();
        }

        public void Save(Stream stream)
        {
            DataContractJsonSerializer serializer = new DataContractJsonSerializer(typeof(NeuralTuringMachine));
            serializer.WriteObject(stream, this);
        }

        public void Save(string path)
        {
            FileStream stream = File.Create(path);
            Save(stream);
            stream.Close();
        }

        public static NeuralTuringMachine Load(string path)
        {
            FileStream stream = File.OpenRead(path);
            NeuralTuringMachine machine = Load(stream);
            stream.Close();
            return machine;
        }

        public static NeuralTuringMachine Load(Stream stream)
        {
            DataContractJsonSerializer serializer = new DataContractJsonSerializer(typeof(NeuralTuringMachine));
            object machine = serializer.ReadObject(stream);
            return (NeuralTuringMachine)machine;
        }

        #endregion
        
        #region Internal Methods
        
        internal Head[] GetHeads()
        {
            return _controller.OutputLayer.HeadsNeurons;
        }

        internal void InitializeMemoryState()
        {
            _newMemoryState = new MemoryState(_memory);
            _newMemoryState.DoInitialReading();
            _oldMemoryState = null;
        }

        internal void BackwardErrorPropagation(double[] knownOutput)
        {
            _newMemoryState.BackwardErrorPropagation();
            _controller.BackwardErrorPropagation(knownOutput, _lastInput, _oldMemoryState.ReadData);
        }

        internal void BackwardErrorPropagation()
        {
            _newMemoryState.BackwardErrorPropagation2();
        }

        public void UpdateWeights(IWeightUpdater weightUpdater)
        {
            _memory.UpdateWeights(weightUpdater);
            _controller.UpdateWeights(weightUpdater);
        } 

        #endregion

        public double[] GetHeadAdressings()
        {
            return _newMemoryState.GetHeadAdressings();
        }

    }
}
