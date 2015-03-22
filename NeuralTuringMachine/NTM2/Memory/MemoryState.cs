using System;
using System.Runtime.Serialization;
using NTM2.Controller;
using NTM2.Memory.Addressing;
using NTM2.Memory.Addressing.Content;

namespace NTM2.Memory
{
    [DataContract]
    internal class MemoryState
    {
        [DataMember]
        private readonly NTMMemory _memory;
        [DataMember]
        private HeadSetting[] _headSettings;
        [DataMember]
        internal ReadData[] ReadData;
        [DataMember]
        private ContentAddressing[] _contentAddressings;

        internal MemoryState(NTMMemory memory)
        {
            _memory = memory;
        }

        internal MemoryState(NTMMemory memory, HeadSetting[] headSettings, ReadData[] readDatas)
        {
            _memory = memory;
            _headSettings = headSettings;
            ReadData = readDatas;
        }

        internal void DoInitialReading()
        {
            _contentAddressings = _memory.GetContentAddressing();
            _headSettings = HeadSetting.GetVector(_memory.HeadCount, i => new Tuple<int, ContentAddressing>(_memory.CellCountN, _contentAddressings[i]));
            ReadData = Memory.ReadData.GetVector(_memory.HeadCount, i => new Tuple<HeadSetting, NTMMemory>(_headSettings[i], _memory));
        }
        
        internal void BackwardErrorPropagation()
        {
            foreach (ReadData readData in ReadData)
            {
                readData.BackwardErrorPropagation();
            }
            _memory.BackwardErrorPropagation();

            foreach (HeadSetting headSetting in _memory.HeadSettings)
            {
                headSetting.BackwardErrorPropagation();
                headSetting.ShiftedVector.BackwardErrorPropagation();
                headSetting.ShiftedVector.GatedAddressing.BackwardErrorPropagation();
                headSetting.ShiftedVector.GatedAddressing.ContentVector.BackwardErrorPropagation();
                foreach (BetaSimilarity similarity in headSetting.ShiftedVector.GatedAddressing.ContentVector.BetaSimilarities)
                {
                    similarity.BackwardErrorPropagation();
                    similarity.Similarity.BackwardErrorPropagation();
                }
            }
        }

        internal void BackwardErrorPropagation2()
        {
            for (int i = 0; i < ReadData.Length; i++)
            {
                ReadData[i].BackwardErrorPropagation();
                for (int j = 0; j < ReadData[i].HeadSetting.AddressingVector.Length; j++)
                {
                    _contentAddressings[i].ContentVector[j].Gradient += ReadData[i].HeadSetting.AddressingVector[j].Gradient;
                }
                _contentAddressings[i].BackwardErrorPropagation();
            }
        }

        internal MemoryState Process(Head[] heads)
        {
            int headCount = heads.Length;
            int memoryColumnsN = _memory.CellCountN;

            ReadData[] newReadDatas = new ReadData[headCount];
            HeadSetting[] newHeadSettings = new HeadSetting[headCount];
            for (int i = 0; i < headCount; i++)
            {
                Head head = heads[i];
                BetaSimilarity[] similarities = new BetaSimilarity[_memory.CellCountN];
                
                for (int j = 0; j < memoryColumnsN; j++)
                {
                    Unit[] memoryColumn = _memory.Data[j];
                    SimilarityMeasure similarity = new SimilarityMeasure(new CosineSimilarityFunction(), head.KeyVector, memoryColumn);
                    similarities[j] = new BetaSimilarity(head.Beta, similarity);
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
