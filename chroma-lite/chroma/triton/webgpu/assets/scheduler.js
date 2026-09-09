// Bound individual GPU submissions; wait for each before queuing more work.
// Batching changes scheduling only; callers retain global photon/pixel indices.
export class GpuBudget {
  generation = 0;
  lastRun = null;
  submissions = 0;
  software = false;

  get mode() { return document.getElementById('gpu_mode')?.value || 'balanced'; }
  cancel() { this.generation++; }
  async checkpoint(token, delay = 0) {
    if (delay) await new Promise(resolve => setTimeout(resolve, delay));
    while (document.hidden && token === this.generation)
      await new Promise(resolve => setTimeout(resolve, 100));
    if (token !== this.generation) throw new DOMException('Stopped.', 'AbortError');
  }

  async run(device, total, encode, {initial = 256, maximum = 2048, progress = null, interactive = false} = {}) {
    const token = this.generation, start = performance.now();
    let offset = 0, batch = this.software ? Math.min(initial,8) : this.mode === 'eco' ? Math.min(initial,32) : initial;
    let compute_ms = 0, batches = 0, max_batch_ms = 0;
    while (offset < total) {
      await this.checkpoint(token);
      const mode = this.mode;
      const limit = Math.min(maximum, this.software ? 8 : interactive ? 256 : mode === 'eco' ? 128 : mode === 'fast' ? 2048 : 1024);
      const count = Math.min(batch, limit, total - offset);
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
      // A handful of workgroups badly underfills a GPU, especially when one ray
      // has a long path. Start with useful parallelism, then adapt toward a short
      // submission rather than imposing an 8 ms target that shrinks to one group.
      // These are scheduling targets, not hard driver execution deadlines.
      const target = mode === 'eco' && !interactive ? 24 : mode === 'fast' ? 250 : 200;
      batch = Math.max(1, Math.min(maximum, count * 2, Math.ceil(count * target / Math.max(.25, elapsed))));
      // Balanced uses the GPU freely. Explicit Low mode adds idle time, while
      // short interaction previews take priority in every mode.
      const pause = mode === 'eco' && !interactive ? Math.max(16,elapsed) : 0;
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
