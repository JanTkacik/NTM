using System;
using System.Collections.Generic;

namespace ObjectPool
{
    public class ObjectArrayPool<T> where T : PoolableObjectBase
    {
        private readonly ObjectPool<T> _basicItemPool; 
        private readonly Dictionary<int, ObjectPool<PoolableObjectArray<T>>> _objectPools;

        public ObjectArrayPool(Func<T> factory)
        {
            _basicItemPool = new ObjectPool<T>(factory);
            _objectPools = new Dictionary<int, ObjectPool<PoolableObjectArray<T>>>();
        }

        public T[] GetObjectArray(int length)
        {
            if (!_objectPools.ContainsKey(length))
            {
                _objectPools.Add(length, new ObjectPool<PoolableObjectArray<T>>(() => new PoolableObjectArray<T>(length, _basicItemPool)));
            }
            return _objectPools[length].GetObject().Array;
        }
    }

    internal class PoolableObjectArray<T> : PoolableObjectBase  where T : PoolableObjectBase
    {
        private readonly T[] _array;

        public PoolableObjectArray(int length, ObjectPool<T> basicObjectPool)
        {
            _array = new T[length];
            for (int i = 0; i < length; i++)
            {
                _array[i] = basicObjectPool.GetObject();
            }
        }
        
        public T[] Array
        {
            get { return _array; }
        }

        protected override void OnResetState()
        {
            foreach (T item in _array)
            {
                item.ResetState();
            }
        }

        protected override void OnReleaseResources()
        {
            foreach (T item in _array)
            {
                item.ReleaseResources();
            }
        }
    }
}
