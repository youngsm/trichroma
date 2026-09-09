// Bound individual GPU submissions and leave idle time for the rest of the system.
// Batching changes scheduling only; callers retain global photon/pixel indices.
export class GpuBudget {
  generation = 0;
  lastRun = null;
  submissions = 0;

  get mode() { return document.getElementById('gpu_mode')?.value || 'balanced'; }
  cancel() { this.generation++; }
  async checkpoint(token, delay = 0) {
    if (delay) await new Promise(resolve => setTimeout(resolve, delay));
    while (document.hidden && token === this.generation)
      await new Promise(resolve => setTimeout(resolve, 100));
    if (token !== this.generation) throw new DOMException('Stopped.', 'AbortError');
  }

  async run(device, total, encode, {initial = 8, maximum = 512, progress = null} = {}) {
    const token = this.generation, start = performance.now();
    let offset = 0, batch = initial, compute_ms = 0, batches = 0, max_batch_ms = 0;
    let queueLatency = 0;
    while (offset < total) {
      await this.checkpoint(token);
      const count = Math.min(batch, maximum, total - offset);
      const begin = performance.now();
      device.queue.submit([encode(offset, count).finish()]);
      this.submissions++;
      await device.queue.onSubmittedWorkDone();
      const elapsed = performance.now() - begin;
      this.active = {completed: offset + count, total, batch: count, elapsed_ms: elapsed};
      compute_ms += elapsed;
      max_batch_ms = Math.max(max_batch_ms, elapsed);
      offset += count;
      batches++;
      if (token !== this.generation) throw new DOMException('Stopped.', 'AbortError');
      if (progress) progress(offset / total);
      // Tiny batches primarily measure browser/driver round-trip latency. Account
      // for that fixed cost so a 5 ms round trip cannot defeat a 4 ms GPU budget.
      if (batches === 1 || count === 1) queueLatency = Math.min(20, elapsed);
      const mode = this.mode, target = queueLatency + (mode === 'eco' ? 4 : mode === 'fast' ? 12 : 8);
      // Round growth upward so driver round-trip latency cannot trap us forever
      // at a single workgroup after a cold shader compilation.
      batch = Math.max(1, Math.min(maximum, count * 2, Math.ceil(count * target / Math.max(.25, elapsed))));
      // Full speed still uses bounded submissions and pauses in background tabs.
      const pause = mode === 'fast' ? 0 : Math.max(mode === 'eco' ? 16 : 8, elapsed * (mode === 'eco' ? 3 : 1));
      await this.checkpoint(token, offset < total ? pause : 0);
    }
    return this.lastRun = {compute_ms, wall_ms: performance.now() - start, batches, max_batch_ms};
  }
}

export const gpuBudget = new GpuBudget();
document.getElementById('stop')?.addEventListener('click', () => {
  gpuBudget.cancel();
  document.dispatchEvent(new Event('gpu-stop'));
});
