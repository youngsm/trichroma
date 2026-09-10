// Keep asynchronous WebGPU allocation errors inside the operation that owns them.
export async function requestGpuDevice({forceFallback = false, storageBufferBytes = 0} = {}) {
  if (!navigator.gpu) throw Error('This browser does not expose WebGPU. Use a compatible browser over HTTPS or localhost.');
  const failures=[];
  for (const fallback of forceFallback ? [true] : [false,true]) {
    try {
      const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance',forceFallbackAdapter:fallback});
      if (!adapter) continue;
      // Larger history buffers are opt-in; other viewers keep the default limits.
      const requiredLimits={};
      if(storageBufferBytes){
        const bytes=Math.min(storageBufferBytes,adapter.limits.maxStorageBufferBindingSize,adapter.limits.maxBufferSize);
        requiredLimits.maxStorageBufferBindingSize=bytes;requiredLimits.maxBufferSize=bytes;
      }
      const device=await adapter.requestDevice({requiredLimits});
      const info=adapter.info;
      const software=!!info.isFallbackAdapter || /swiftshader|llvmpipe|software/i.test(`${info.architecture} ${info.description}`);
      return {adapter,device,software};
    } catch (error) { failures.push(error.message || String(error)); }
  }
  throw Error('No usable WebGPU adapter: hardware is unavailable and this browser did not provide a CPU adapter. Enable hardware acceleration and reload.'+(failures.length?' '+failures.join('; '):''));
}

export function isGpuMemoryError(error) {
  return error?.name === 'GpuMemoryError' ||
    /out.of.(device.|host.)?memory|VK_ERROR_OUT_OF_|memory allocation failed/i.test(String(error?.message || error || ''));
}

export async function checkedGpuWork(device, operation) {
  device.pushErrorScope('out-of-memory');
  device.pushErrorScope('validation');
  let result, failure;
  try { result = await operation(); }
  catch (error) { failure = error; }
  const validation = await device.popErrorScope();
  const memory = await device.popErrorScope();
  if (memory) {
    const error = new Error(memory.message || 'GPU memory allocation failed.');
    error.name = 'GpuMemoryError';
    throw error;
  }
  if (failure) throw failure;
  if (validation) throw new Error(validation.message);
  return result;
}

export async function allocateGpu(device, build) {
  const owned = [];
  const arena = {
    buffer(data, usage) {
      const size = typeof data === 'number' ? data : data.byteLength;
      if (size > device.limits.maxBufferSize ||
          ((usage & GPUBufferUsage.STORAGE) && size > device.limits.maxStorageBufferBindingSize)) {
        const error = new Error(`GPU buffer needs ${size} bytes, exceeding this adapter's limit.`);
        error.name = 'GpuMemoryError';
        throw error;
      }
      const buffer = device.createBuffer({size: Math.max(4, size), usage});
      owned.push(buffer);
      if (typeof data !== 'number') device.queue.writeBuffer(buffer, 0, data);
      return buffer;
    },
    texture(descriptor) {
      const texture = device.createTexture(descriptor);
      owned.push(texture);
      return texture;
    }
  };
  try { return await checkedGpuWork(device, () => build(arena)); }
  catch (error) {
    for (const resource of owned) resource.destroy();
    throw error;
  }
}
