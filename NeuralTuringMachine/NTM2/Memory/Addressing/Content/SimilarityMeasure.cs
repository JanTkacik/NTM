using System.Runtime.Serialization;
using NTM2.Controller;

namespace NTM2.Memory.Addressing.Content
{
    [KnownType(typeof(CosineSimilarityFunction))]
    [DataContract]
    internal class SimilarityMeasure
    {
        [DataMember]
        private readonly ISimilarityFunction _similarityFunction;
        [DataMember]
        private readonly Unit[] _u;
        [DataMember]
        private readonly Unit[] _v;
        [DataMember]
        internal readonly Unit Similarity;

        internal SimilarityMeasure(ISimilarityFunction similarityFunction, Unit[] u, Unit[] v)
        {
            _similarityFunction = similarityFunction;
            _u = u;
            _v = v;
            Similarity = similarityFunction.Calculate(u, v);
        }

        private SimilarityMeasure()
        {
            
        }

        internal void BackwardErrorPropagation()
        {
            _similarityFunction.Differentiate(Similarity, _u, _v);
        }
    }
}
