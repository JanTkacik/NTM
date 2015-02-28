using System;

namespace ObjectPool
{
    public abstract class PoolableObjectBase : IDisposable
    {
        internal Action<PoolableObjectBase, bool> ReturnToPoolAction;
        internal bool Disposed { get; set; }
        
        internal void ReleaseResources()
        {
            try
            {
                OnReleaseResources();
            }
            // ReSharper disable EmptyGeneralCatchClause - destroying object we do not care about a result of release operation
            catch (Exception)
            // ReSharper restore EmptyGeneralCatchClause
            {
                
            }
        }

        internal bool ResetState()
        {
            bool success = true;

            try
            {
                OnResetState();
            }
            catch (Exception)
            {
                success = false;
            }

            return success;
        }

        protected abstract void OnResetState();

        protected abstract void OnReleaseResources();

        public void Dispose()
        {
            AddToPool(false);
        }

        ~PoolableObjectBase()
        {
            AddToPool(true);
        }

        private void AddToPool(bool reRegisterForFinalization)
        {
            if (!Disposed)
            {
                try
                {
                    ReturnToPoolAction(this, reRegisterForFinalization);
                }
                catch (Exception)
                {
                    Disposed = true;
                    ReleaseResources();
                }
            }
        }
    }
}
