using System;
using NTM2.Controller;

namespace NTM2.Memory.Addressing
{
    public class CosineSimilarity
    {
        private readonly Unit[] _u;
        private readonly Unit[] _v;
        private readonly Unit _data;
        private readonly double _uv;
        private readonly double _normalizedU;
        private readonly double _normalizedV;

        //Implementation of cosine similarity (Page 8, Unit 3.3.1 Focusing by Content)
        public CosineSimilarity(Unit[] u, Unit[] v)
        {
            _u = u;
            _v = v;
            
            for (int i = 0; i < u.Length; i++)
            {
                _uv += u[i].Value*v[i].Value;
                _normalizedU += u[i].Value * u[i].Value;
                _normalizedV += v[i].Value * v[i].Value;
            }

            _normalizedU = Math.Sqrt(_normalizedU);
            _normalizedV = Math.Sqrt(_normalizedV);

            _data = new Unit(_uv / (_normalizedU * _normalizedV));
            if (double.IsNaN(_data.Value))
            {
                throw new Exception("Cosine similarity is nan -> error");
            }
        }

        public CosineSimilarity()
        {
        }

        public Unit Data
        {
            get { return _data; }
        }

        public static CosineSimilarity[] GetVector(int x)
        {
            CosineSimilarity[] vector = new CosineSimilarity[x];
            for (int i = 0; i < x; i++)
            {
                vector[i] = new CosineSimilarity();
            }
            return vector;
        }

        public static CosineSimilarity[][] GetTensor2(int x, int y)
        {
            CosineSimilarity[][] tensor = new CosineSimilarity[x][];
            for (int i = 0; i < x; i++)
            {
                tensor[i] = GetVector(y);
            }
            return tensor;
        }

        public void BackwardErrorPropagation()
        {
            double uvuu = _uv/(_normalizedU*_normalizedU);
            double uvvv = _uv/(_normalizedV*_normalizedV);
            double uvg = _data.Gradient/(_normalizedU*_normalizedV);
            for (int i = 0; i < _u.Length; i++)
            {
                double u = _u[i].Value;
                double v = _v[i].Value;

                _u[i].Gradient += (v - (u*uvuu))*uvg;
                _v[i].Gradient += (u - (v*uvvv))*uvg;
            }
        }
    }
}
