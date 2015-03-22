using NTM2.Controller;
using NTM2.Memory.Addressing.Content;

namespace NTM2.Memory.Addressing
{
    internal class GatedAddressing
    {
        private readonly Unit _gate;
        private readonly HeadSetting _oldHeadSettings;
        internal readonly ContentAddressing ContentVector;
        internal readonly Unit[] GatedVector;
        private readonly int _memoryCellCount;
        //Interpolation gate
        private readonly double _gt;
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
