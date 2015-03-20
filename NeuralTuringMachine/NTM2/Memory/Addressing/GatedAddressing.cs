using NTM2.Controller;

namespace NTM2.Memory.Addressing
{
    public class GatedAddressing
    {
        private readonly Unit _gate;
        private readonly ContentAddressing _contentAddressing;
        private readonly HeadSetting _oldHeadSettings;
        private readonly Unit[] _data;

        public GatedAddressing(Unit gate, ContentAddressing contentAddressing, HeadSetting oldHeadSettings)
        {
            _gate = gate;
            _contentAddressing = contentAddressing;
            _oldHeadSettings = oldHeadSettings;
            _data = UnitFactory.GetVector(_contentAddressing.Data.Length);

            //Implementation of focusing by location - page 8 part 3.3.2. Focusing by Location
            double g = Sigmoid.GetValue(_gate.Value);
            for (int i = 0; i < _data.Length; i++)
            {
                _data[i].Value = (g * _contentAddressing.Data[i].Value) + ((1 - g) * _oldHeadSettings.Data[i].Value);
            }
        }

        public Unit[] Data
        {
            get { return _data; }
        }

        public ContentAddressing ContentAddressing
        {
            get { return _contentAddressing; }
        }

        public void BackwardErrorPropagation()
        {
            double gt = Sigmoid.GetValue(_gate.Value);

            double gradient = 0;
            for (int i = 0; i < _data.Length; i++)
            {
                gradient += (_contentAddressing.Data[i].Value - _oldHeadSettings.Data[i].Value)*_data[i].Gradient;
            }
            _gate.Gradient += gradient*gt*(1 - gt);

            for (int i = 0; i < _contentAddressing.Data.Length; i++)
            {
                _contentAddressing.Data[i].Gradient += gt*_data[i].Gradient;
            }

            for (int i = 0; i < _oldHeadSettings.Data.Length; i++)
            {
                _oldHeadSettings.Data[i].Gradient += (1 - gt)*_data[i].Gradient;
            }
        }
    }
}
