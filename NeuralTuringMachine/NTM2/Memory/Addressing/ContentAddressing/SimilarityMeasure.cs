using NTM2.Controller;

namespace NTM2.Memory.Addressing.ContentAddressing
{
    internal class SimilarityMeasure
    {
        private readonly ISimilarityFunction _similarityFunction;
        private readonly Unit[] _u;
        private readonly Unit[] _v;
        internal readonly Unit Data;

        internal SimilarityMeasure(ISimilarityFunction similarityFunction, Unit[] u, Unit[] v)
        {
            _similarityFunction = similarityFunction;
            _u = u;
            _v = v;
            Data = similarityFunction.Calculate(u, v);
        }

        internal void BackwardErrorPropagation()
        {
            _similarityFunction.Differentiate(Data, _u, _v);
        }
    }
}
