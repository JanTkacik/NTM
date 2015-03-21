using System;
using NTM2.Controller;

namespace NTM2.Memory.Addressing
{
    internal class BetaSimilarity
    {
        private readonly Unit _beta;
        private readonly CosineSimilarity _similarity;
        private readonly Unit _data;
        private readonly double _b;

        public BetaSimilarity(Unit beta, CosineSimilarity cosineSimilarity)
        {
            _beta = beta;
            _similarity = cosineSimilarity;
            _b = Math.Exp(_beta.Value);
            _data = new Unit(_b * _similarity.Data.Value);
        }

        public BetaSimilarity()
        {
            _beta = new Unit();
            _data = new Unit();
        }
        
        public Unit Data
        {
            get { return _data; }
        }

        public CosineSimilarity CosineSimilarity
        {
            get { return _similarity; }
        }

        public static BetaSimilarity[][] GetTensor2(int x, int y)
        {
            BetaSimilarity[][] tensor = new BetaSimilarity[x][];
            for (int i = 0; i < x; i++)
            {
                tensor[i] = GetVector(y);
            }
            return tensor;
        }

        public static BetaSimilarity[] GetVector(int x)
        {
            BetaSimilarity[] vector = new BetaSimilarity[x];
            for (int i = 0; i < x; i++)
            {
                vector[i] = new BetaSimilarity();
            }
            return vector;
        }

        public void BackwardErrorPropagation()
        {
            _beta.Gradient += _similarity.Data.Value*_b*_data.Gradient;
            _similarity.Data.Gradient += _b*_data.Gradient;
        }
    }
}
