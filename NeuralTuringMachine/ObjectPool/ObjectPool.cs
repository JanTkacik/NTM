using System;
using System.Collections.Concurrent;

namespace ObjectPool
{
    public class ObjectPool<T> where T : PoolableObjectBase
    {
        private readonly ConcurrentBag<T> _pooledObject;
        private readonly Func<T> _factoryMethod;

        public ObjectPool(Func<T> factoryMethod)
        {
            _factoryMethod = factoryMethod;
            _pooledObject = new ConcurrentBag<T>();
        }

        private void ReturnToPoolAction(PoolableObjectBase pooledObject, bool reRegisterForFinalization)
        {
            T item = (T)pooledObject;

            //Reset object state and if reset fails destroy object
            if (!item.ResetState())
            {
                DestroyPooledObject(item);
                return;
            }

            //Object resurrection - called from finalize method
            if (reRegisterForFinalization)
            {
                GC.ReRegisterForFinalize(item);
            }

            _pooledObject.Add(item);
        }

        private void DestroyPooledObject(PoolableObjectBase objectToDestroy)
        {
            if (!objectToDestroy.Disposed)
            {
                objectToDestroy.ReleaseResources();
                objectToDestroy.Disposed = true;
            }

            GC.SuppressFinalize(objectToDestroy);
        }

        public T GetObject()
        {
            T objectFromPool;

            bool objectAvailible = _pooledObject.TryTake(out objectFromPool);
            if (objectAvailible)
            {
                return objectFromPool;
            }
            T newObject = _factoryMethod();
            newObject.ReturnToPoolAction = ReturnToPoolAction;
            return newObject;
        }
        
        ~ObjectPool()
        {
            foreach (T item in _pooledObject)
            {
                DestroyPooledObject(item);
            }
        }
    }
}
