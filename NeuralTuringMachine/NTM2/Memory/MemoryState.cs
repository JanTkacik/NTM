using System;
using NTM2.Controller;
using NTM2.Memory.Addressing;

namespace NTM2.Memory
{
    internal class MemoryState
    {
        private readonly NTMMemory _memory;
        private HeadSetting[] _headSettings;
        private ReadData[] _reads;
        private ContentAddressing[] _contentAddressings;

        internal MemoryState(NTMMemory memory)
        {
            _memory = memory;
        }

        internal MemoryState(NTMMemory memory, HeadSetting[] headSettings, ReadData[] readDatas)
        {
            _memory = memory;
            _headSettings = headSettings;
            _reads = readDatas;
        }

        internal void DoInitialReading()
        {
            _contentAddressings = _memory.GetContentAddressing();
            _headSettings = HeadSetting.GetVector(_memory.HeadCount, i => new Tuple<int, ContentAddressing>(_memory.MemoryColumnsN, _contentAddressings[i]));
            _reads = Memory.ReadData.GetVector(_memory.HeadCount, i => new Tuple<HeadSetting, NTMMemory>(_headSettings[i], _memory));
        }

        public ReadData[] ReadData
        {
            get { return _reads; }
        }
        
        public void BackwardErrorPropagation()
        {
            foreach (ReadData readData in _reads)
            {
                readData.BackwardErrorPropagation();
            }
            _memory.BackwardErrorPropagation();

            foreach (HeadSetting headSetting in _memory.HeadSettings)
            {
                headSetting.BackwardErrorPropagation();
                headSetting.ShiftedAddressing.BackwardErrorPropagation();
                headSetting.ShiftedAddressing.GatedAddressing.BackwardErrorPropagation();
                headSetting.ShiftedAddressing.GatedAddressing.ContentAddressing.BackwardErrorPropagation();
                foreach (BetaSimilarity similarity in headSetting.ShiftedAddressing.GatedAddressing.ContentAddressing.BetaSimilarities)
                {
                    similarity.BackwardErrorPropagation();
                    similarity.CosineSimilarity.BackwardErrorPropagation();
                }
            }
        }

        public void BackwardErrorPropagation2()
        {
            for (int i = 0; i < _reads.Length; i++)
            {
                _reads[i].BackwardErrorPropagation();
                for (int j = 0; j < _reads[i].HeadSetting.Data.Length; j++)
                {
                    _contentAddressings[i].Data[j].Gradient += _reads[i].HeadSetting.Data[j].Gradient;
                }
                _contentAddressings[i].BackwardErrorPropagation();
            }
        }

        internal MemoryState Process(Head[] heads)
        {
            ReadData[] newReadDatas = new ReadData[heads.Length];
            HeadSetting[] newHeadSettings = new HeadSetting[heads.Length];
            for (int i = 0; i < heads.Length; i++)
            {
                Head head = heads[i];
                BetaSimilarity[] similarities = new BetaSimilarity[_memory.MemoryColumnsN];
                for (int j = 0; j < _memory.Data.Length; j++)
                {
                    Unit[] memoryColumn = _memory.Data[j];
                    CosineSimilarity cosineSimilarity = new CosineSimilarity(head.KeyVector, memoryColumn);
                    similarities[j] = new BetaSimilarity(head.Beta, cosineSimilarity);
                }
                ContentAddressing ca = new ContentAddressing(similarities);
                GatedAddressing ga = new GatedAddressing(head.Gate, ca, _headSettings[i]);
                ShiftedAddressing sa = new ShiftedAddressing(head.Shift, ga);

                newHeadSettings[i] = new HeadSetting(head.Gamma, sa);
                newReadDatas[i] = new ReadData(newHeadSettings[i], _memory);
            }

            NTMMemory newMemory = new NTMMemory(newHeadSettings, heads, _memory);

            return new MemoryState(newMemory, newHeadSettings, newReadDatas);
        }
    }
}
