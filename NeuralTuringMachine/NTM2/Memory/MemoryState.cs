using System;
using NTM2.Controller;
using NTM2.Memory.Addressing;

namespace NTM2.Memory
{
    public class MemoryState
    {
        private readonly NTMMemory _memory;
        private readonly HeadSetting[] _headSettings;
        private readonly ReadData[] _reads;
        private readonly ContentAddressing[] _contentAddressings;

        public MemoryState(NTMMemory memory)
        {
            _memory = memory;
            //TODO refactor
            _contentAddressings = _memory.GetContentAddressing();
            _headSettings = HeadSetting.GetVector(_memory.HeadCount, i => new Tuple<int, ContentAddressing>(_memory.MemoryColumnsN, _contentAddressings[i]));
            _reads = NTM2.Memory.ReadData.GetVector(_memory.HeadCount, i => new Tuple<HeadSetting, NTMMemory>(_headSettings[i], _memory));
        }

        public MemoryState(Head[] heads, NTMMemory memory)
        {
            _reads = new ReadData[heads.Length];
            _headSettings = new HeadSetting[heads.Length];
            for (int i = 0; i < heads.Length; i++)
            {
                Head head = heads[i];
                BetaSimilarity[] similarities = new BetaSimilarity[memory.MemoryColumnsN];
                for (int j = 0; j < memory.Data.Length; j++)
                {
                    Unit[] memoryColumn = memory.Data[j];
                    CosineSimilarity cosineSimilarity = new CosineSimilarity(head.KeyVector, memoryColumn);
                    similarities[j] = new BetaSimilarity(head.Beta, cosineSimilarity);
                }
                ContentAddressing ca = new ContentAddressing(similarities);
                GatedAddressing ga = new GatedAddressing(head.Gate, ca, head.OldHeadSettings);
                ShiftedAddressing sa = new ShiftedAddressing(head.Shift, ga);

                _headSettings[i] = new HeadSetting(head.Gamma, sa);
                _reads[i] = new ReadData(_headSettings[i], memory);
            }

            _memory = new NTMMemory(_headSettings, heads, memory);
        }

        public ReadData[] ReadData
        {
            get { return _reads; }
        }

        public HeadSetting[] HeadSettings
        {
            get { return _headSettings; }
        }

        public NTMMemory Memory
        {
            get { return _memory; }
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
    }
}
