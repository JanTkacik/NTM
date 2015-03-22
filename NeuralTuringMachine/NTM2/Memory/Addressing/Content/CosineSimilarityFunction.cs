using System;
using NTM2.Controller;

namespace NTM2.Memory.Addressing.Content
{
    class CosineSimilarityFunction : ISimilarityFunction
    {
        private double _uv;
        private double _normalizedU;
        private double _normalizedV;

        public Unit Calculate(Unit[] u, Unit[] v)
        {
            for (int i = 0; i < u.Length; i++)
            {
                _uv += u[i].Value * v[i].Value;
                _normalizedU += u[i].Value * u[i].Value;
                _normalizedV += v[i].Value * v[i].Value;
            }

            _normalizedU = Math.Sqrt(_normalizedU);
            _normalizedV = Math.Sqrt(_normalizedV);

            Unit data = new Unit(_uv / (_normalizedU * _normalizedV));
            if (double.IsNaN(data.Value))
            {
                throw new Exception("Cosine similarity is nan -> error");
            }

            return data;
        }

        public void Differentiate(Unit similarity, Unit[] uVector, Unit[] vVector)
        {
            double uvuu = _uv / (_normalizedU * _normalizedU);
            double uvvv = _uv / (_normalizedV * _normalizedV);
            double uvg = similarity.Gradient / (_normalizedU * _normalizedV);
            for (int i = 0; i < uVector.Length; i++)
            {
                double u = uVector[i].Value;
                double v = vVector[i].Value;

                uVector[i].Gradient += (v - (u * uvuu)) * uvg;
                vVector[i].Gradient += (u - (v * uvvv)) * uvg;
            }
        }
    }
}
