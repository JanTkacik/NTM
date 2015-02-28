using System;
using System.Linq;
using NTM2.Controller;

namespace NTM2.Memory.Addressing
{
    public class ContentAddressing
    {
        private readonly BetaSimilarity[] _units;
        private readonly Unit[] _data;

        //Implementation of focusing by content (Page 8, Unit 3.3.1 Focusing by Content)
        public ContentAddressing(BetaSimilarity[] units, UnitFactory unitFactory)
        {
            _units = units;
            _data = unitFactory.GetVector(units.Length);

            //Subtracting max increase numerical stability
            double max = _units.Max(similarity => similarity.Data.Value);
            double sum = 0;

            for (int i = 0; i < _units.Length; i++)
            {
                BetaSimilarity unit = _units[i];
                double weight = Math.Exp(unit.Data.Value - max);
                _data[i].Value = weight;
                sum += weight;
            }
            
            foreach (Unit unit in _data)
            {
                unit.Value = unit.Value/sum;
            }
        }

        public Unit[] Data
        {
            get { return _data; }
        }

        public BetaSimilarity[] BetaSimilarities
        {
            get { return _units; }
        }

        public static ContentAddressing[] GetVector(int x, Func<int,BetaSimilarity[]> paramGetter, UnitFactory unitFactory)
        {
            ContentAddressing[] vector = new ContentAddressing[x];
            for (int i = 0; i < x; i++)
            {
                vector[i] = new ContentAddressing(paramGetter(i), unitFactory);
            }
            return vector;
        }

        public void BackwardErrorPropagation()
        {
            double gradient = 0;
            foreach (Unit unit in _data)
            {
                gradient += unit.Gradient*unit.Value;
            }

            for (int i = 0; i < _data.Length; i++)
            {
                _units[i].Data.Gradient += (_data[i].Gradient - gradient)*_data[i].Value;
            }
        }
    }
}
