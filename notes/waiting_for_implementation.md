# Implement free()

For M1.3, because we're only handling one request, free() can simply clear the reference/state.

For example, you can eventually design request-specific allocation, but don't prematurely build the M2/M3 allocator.

For now we implemented in kv_cache.py only:


```python
def free(self, request_id):
    pass
```