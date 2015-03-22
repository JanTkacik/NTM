using NTM2.Controller;

namespace NTM2.Memory.Addressing.Content
{
    internal class SimilarityMeasure
    {
        private readonly ISimilarityFunction _similarityFunction;
        private readonly Unit[] _u;
        private readonly Unit[] _v;
        internal readonly Unit Similarity;

        internal SimilarityMeasure(ISimilarityFunction similarityFunction, Unit[] u, Unit[] v)
        {
            _similarityFunction = similarityFunction;
            _u = u;
            _v = v;
            Similarity = similarityFunction.Calculate(u, v);
        }

        internal void BackwardErrorPropagation()
        {
            _similarityFunction.Differentiate(Similarity, _u, _v);
        }
    }
}
