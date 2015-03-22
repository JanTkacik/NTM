using System;
using NTM2.Controller;

namespace NTM2.Memory.Addressing
{
    internal class ShiftedAddressing
    {
        private readonly Unit _shift;
        private readonly double _convolution;
        private readonly int _cellCount;
        private readonly double _simj;
        private readonly double _oneMinusSimj;
        
        internal readonly GatedAddressing GatedAddressing;
        internal readonly Unit[] ShiftedVector;

        //IMPLEMENTATION OF SHIFT - page 9
        internal ShiftedAddressing(Unit shift, GatedAddressing gatedAddressing)
        {
            _shift = shift;
            GatedAddressing = gatedAddressing;
            _cellCount = GatedAddressing.GatedVector.Length;
            
            ShiftedVector = UnitFactory.GetVector(_cellCount);
            double cellCountDbl = _cellCount;

            //Max shift is from range -1 to 1
            double maxShift = ((2 * Sigmoid.GetValue(_shift.Value)) - 1);
            _convolution = (maxShift + cellCountDbl) % cellCountDbl;

            _simj = 1 - (_convolution - Math.Floor(_convolution));
            _oneMinusSimj = (1 - _simj);
            
            int convolution = (int)_convolution;
            
            for (int i = 0; i < _cellCount; i++)
            {
                int imj = (i + convolution) % _cellCount;
                
                ShiftedVector[i].Value = (GatedAddressing.GatedVector[imj].Value * _simj) +
                                 (GatedAddressing.GatedVector[(imj + 1) % _cellCount].Value * _oneMinusSimj);
                if (ShiftedVector[i].Value < 0 || double.IsNaN(ShiftedVector[i].Value))
                {
                    throw new Exception("Error - weight should not be smaller than zero or nan");
                }
            }
        }

        internal void BackwardErrorPropagation()
        {
            double gradient = 0;
            for (int i = 0; i < _cellCount; i++)
            {
                int imj = (i + ((int)_convolution)) % _cellCount;
                gradient += ((-GatedAddressing.GatedVector[imj].Value) + GatedAddressing.GatedVector[(imj + 1) % _cellCount].Value) * ShiftedVector[i].Gradient;
                int j = (i - ((int)_convolution) + _cellCount) % _cellCount;
                GatedAddressing.GatedVector[i].Gradient += (ShiftedVector[i].Gradient * _simj) + (ShiftedVector[(j - 1 + _cellCount) % _cellCount].Gradient * _oneMinusSimj);
            }
            
            double sig = Sigmoid.GetValue(_shift.Value);
            gradient = gradient * 2 * sig * (1 - sig);
            _shift.Gradient += gradient;
        }
    }
}
