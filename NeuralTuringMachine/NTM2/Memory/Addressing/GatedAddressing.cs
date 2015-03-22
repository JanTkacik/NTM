using System.Runtime.Serialization;
using NTM2.Controller;
using NTM2.Memory.Addressing.Content;

namespace NTM2.Memory.Addressing
{
    [DataContract]
    internal class GatedAddressing
    {
        [DataMember]
        private readonly Unit _gate;
        [DataMember]
        private readonly HeadSetting _oldHeadSettings;
        [DataMember]
        internal readonly ContentAddressing ContentVector;
        [DataMember]
        internal readonly Unit[] GatedVector;
        [DataMember]
        private readonly int _memoryCellCount;
        //Interpolation gate
        [DataMember]
        private readonly double _gt;
        [DataMember]
        private readonly double _oneminusgt;

        internal GatedAddressing(Unit gate, ContentAddressing contentAddressing, HeadSetting oldHeadSettings)
        {
            _gate = gate;
            ContentVector = contentAddressing;
            _oldHeadSettings = oldHeadSettings;
            Unit[] contentVector = ContentVector.ContentVector;
            _memoryCellCount = contentVector.Length;
            GatedVector = UnitFactory.GetVector(_memoryCellCount);

            //Implementation of focusing by location - page 8 part 3.3.2. Focusing by Location
            _gt = Sigmoid.GetValue(_gate.Value);
            _oneminusgt = (1 - _gt);

            for (int i = 0; i < _memoryCellCount; i++)
            {
                GatedVector[i].Value = (_gt * contentVector[i].Value) + (_oneminusgt * _oldHeadSettings.AddressingVector[i].Value);
            }
        }

        private GatedAddressing()
        {
            
        }

        internal void BackwardErrorPropagation()
        {
            Unit[] contentVector = ContentVector.ContentVector;

            double gradient = 0;
            for (int i = 0; i < _memoryCellCount; i++)
            {
                Unit oldHeadSetting = _oldHeadSettings.AddressingVector[i];
                Unit contentVectorItem = contentVector[i];
                Unit gatedVectorItem = GatedVector[i];

                gradient += (contentVectorItem.Value - oldHeadSetting.Value) * gatedVectorItem.Gradient;
                contentVectorItem.Gradient += _gt * gatedVectorItem.Gradient;
                oldHeadSetting.Gradient += _oneminusgt * gatedVectorItem.Gradient;
            }

            _gate.Gradient += gradient * _gt * _oneminusgt;
        }
    }
}
