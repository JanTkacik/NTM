using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NTM2.Controller;

namespace NTM2.Memory.Addressing
{
    public class ShiftedAddressing
    {
        private readonly Unit _shift;
        private readonly double _convolution;
        private readonly GatedAddressing _gatedAddressing;
        private readonly Unit[] _data;

        //IMPLEMENTATION OF SHIFT - page 9
        public ShiftedAddressing(Unit shift, GatedAddressing gatedAddressing)
        {
            _shift = shift;
            _gatedAddressing = gatedAddressing;
            _data = Unit.GetVector(_gatedAddressing.Data.Length);

            //WTF
            double length = (double)_gatedAddressing.Data.Length;
            double maxShift = ((2*Sigmoid.GetValue(_shift.Value)) - 1);
            _convolution = (maxShift + length) % length;

            double simj = 1 - (_convolution - Math.Floor(_convolution));
            
            int n = _gatedAddressing.Data.Length;
            for (int i = 0; i < _data.Length; i++)
            {
                int imj = (i + (int) _convolution) % n;
                _data[i].Value = (_gatedAddressing.Data[imj].Value*simj) +
                                 (_gatedAddressing.Data[(imj + 1)%n].Value*(1 - simj));
                if (_data[i].Value < 0 || double.IsNaN(_data[i].Value))
                {
                    throw new Exception("Error - weight should not be smaller than zero or nan");
                }
            }
        }

        public Unit[] Data
        {
            get { return _data; }
        }

        public GatedAddressing GatedAddressing
        {
            get { return _gatedAddressing; }
        }

        public void BackwardErrorPropagation()
        {
            double gradient = 0;
            int n = _gatedAddressing.Data.Length;
            for (int i = 0; i < _data.Length; i++)
            {
                int imj = (i + ((int)_convolution)) % n;
                gradient += ((-_gatedAddressing.Data[imj].Value) + _gatedAddressing.Data[(imj + 1)%n].Value) * _data[i].Gradient;
            }
            double sig = Sigmoid.GetValue(_shift.Value);
            gradient = gradient * 2 * sig * (1 - sig);
            _shift.Gradient += gradient;

            double simj = 1 - (_convolution - Math.Floor(_convolution));
            for (int i = 0; i < _gatedAddressing.Data.Length; i++)
            {
                int j = (i - ((int) _convolution) + n) % n;
                _gatedAddressing.Data[i].Gradient += (_data[i].Gradient * simj) + (_data[(j-1+n)%n].Gradient * (1-simj));
            }
        }
    }
}
