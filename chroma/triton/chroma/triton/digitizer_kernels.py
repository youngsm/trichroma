"""FP64 GPU pulse superposition with the same interpolation as the CPU model."""
import numpy as np
import torch
import triton
import triton.language as tl

from .optical_response import Waveforms, digitizer_parameters


@triton.jit
def pulse_kernel(pe_times, charges, channels, event_rows, pulse_x, pulse_y, metadata, samples,
                 nchannel, nsample,
                 NK: tl.constexpr, BLOCK: tl.constexpr):
    pe = tl.program_id(0)
    start, period = tl.load(metadata), tl.load(metadata+1)
    first_x, last_x = tl.load(metadata+2), tl.load(metadata+3)
    t = tl.load(pe_times+pe).to(tl.float64)
    q = tl.load(charges+pe).to(tl.float64)
    ch = tl.load(channels+pe).to(tl.int64)
    ev = tl.load(event_rows+pe).to(tl.int64)
    lo = tl.maximum(0, tl.ceil((t+first_x-start)/period).to(tl.int64))
    hi = tl.minimum(nsample, tl.floor((t+last_x-start)/period).to(tl.int64)+1)
    sample = lo+tl.program_id(1)*BLOCK+tl.arange(0, BLOCK)
    valid = sample < hi
    x = start+sample.to(tl.float64)*period-t
    left = tl.full((BLOCK,), 0, tl.int32)
    right = tl.full((BLOCK,), NK-1, tl.int32)
    while tl.sum((right-left > 1).to(tl.int32), 0) > 0:
        mid = (left+right)//2
        xm = tl.load(pulse_x+mid)
        choose_right = xm <= x
        width = right-left
        left = tl.where(choose_right & (width > 1), mid, left)
        right = tl.where(~choose_right & (width > 1), mid, right)
    xa, xb = tl.load(pulse_x+left), tl.load(pulse_x+right)
    ya, yb = tl.load(pulse_y+left), tl.load(pulse_y+right)
    value = ya+(x-xa)*((yb-ya)/(xb-xa))
    tl.atomic_add(samples+(ev*nchannel+ch)*nsample+sample, q*value, valid, sem="relaxed")


def digitize_gpu(photoelectrons, *, event_indices, channel_count, start_ns, sample_period_ns,
                 sample_count, pulse_times_ns, pulse_adc_per_pe, baseline=0., noise_rms=0.,
                 adc_bits=None, seed=1, device="cuda"):
    events, px, py, times = digitizer_parameters(event_indices=event_indices, channel_count=channel_count,
        start_ns=start_ns, sample_period_ns=sample_period_ns, sample_count=sample_count,
        pulse_times_ns=pulse_times_ns, pulse_adc_per_pe=pulse_adc_per_pe, baseline=baseline,
        noise_rms=noise_rms, adc_bits=adc_bits)
    pe = photoelectrons
    if np.any(pe.channels < 0) or np.any(pe.channels >= channel_count) or not np.isin(pe.event_indices, events).all():
        raise ValueError("photoelectron refers to an unspecified event or channel")
    samples = torch.full((len(events), channel_count, sample_count), float(baseline), dtype=torch.float64, device=device)
    if len(pe.times):
        order = np.argsort(events)
        rows = order[np.searchsorted(events[order], pe.event_indices)]
        arrays = [torch.from_numpy(np.array(a, copy=True)).to(device) for a in
                  (pe.times, pe.charges, pe.channels, rows, px, py,
                   np.asarray([start_ns, sample_period_ns, px[0], px[-1]], np.float64))]
        width = int(np.ceil((px[-1]-px[0])/sample_period_ns))+2
        block = min(128, triton.next_power_of_2(width))
        pulse_kernel[(len(pe.times), triton.cdiv(width, block))](
            *arrays, samples, channel_count, sample_count, NK=len(px), BLOCK=block,
            num_warps=4, enable_fp_fusion=False)
    host = samples.cpu().numpy()
    if noise_rms:
        for row, event in enumerate(events):
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(event), 2052]))
            host[row] += rng.normal(0, noise_rms, host[row].shape)
    if adc_bits is not None:
        host = np.rint(np.clip(host, 0, 2**adc_bits-1)).astype(np.uint32)
    return Waveforms(events.copy(), times, host)
