using System;
using NTM2.Controller;

namespace NTM2.Memory.Addressing
{
    internal class CosineSimilarity
    {
        private readonly Unit[] _u;
        private readonly Unit[] _v;
        private readonly Unit _data;
        private readonly double _uv;
        private readonly double _normalizedU;
        private readonly double _normalizedV;
        
        private CosineSimilarity(Unit[] u, Unit[] v, double uv, double normalizedU, double normalizedV, Unit data)
        {
            _u = u;
            _v = v;
            _uv = uv;
            _data = data;
            _normalizedU = normalizedU;
            _normalizedV = normalizedV;
        }

        internal Unit Data
        {
            get { return _data; }
        }
        
        internal void BackwardErrorPropagation()
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

        //Implementation of cosine similarity (Page 8, Unit 3.3.1 Focusing by Content)
        public static CosineSimilarity Calculate(Unit[] u, Unit[] v)
        {
            double normalizedU = 0;
            double normalizedV = 0;
            double uv = 0;

            for (int i = 0; i < u.Length; i++)
            {
                uv += u[i].Value * v[i].Value;
                normalizedU += u[i].Value * u[i].Value;
                normalizedV += v[i].Value * v[i].Value;
            }

            normalizedU = Math.Sqrt(normalizedU);
            normalizedV = Math.Sqrt(normalizedV);

            Unit data = new Unit(uv / (normalizedU * normalizedV));
            if (double.IsNaN(data.Value))
            {
                throw new Exception("Cosine similarity is nan -> error");
            }

            return new CosineSimilarity(u, v, uv, normalizedU, normalizedV, data);
        }
    }
}
