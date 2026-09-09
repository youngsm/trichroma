// -*-c++-*-
#include <curand_kernel.h>

#include "sorting.h"

extern "C"
{

__global__ void
bin_hits(int nchannels, float *channel_q, float *channel_time,
	 unsigned int *hitcount, int tbins, float tmin, float tmax, int qbins,
	 float qmin, float qmax, unsigned int *pdf)
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;

    if (id >= nchannels)
	return;

    unsigned int q = channel_q[id];
    float t = channel_time[id];
	
    if (t < 1e8 && t >= tmin && t < tmax && q >= qmin && q < qmax) {
	hitcount[id] += 1;
		
	int tbin = (t - tmin) / (tmax - tmin) * tbins;
	int qbin = (q - qmin) / (qmax - qmin) * qbins;
	    
	// row major order (channel, t, q)
	int bin = id * (tbins * qbins) + tbin * qbins + qbin;
	pdf[bin] += 1;
    }
}

__global__ void
accumulate_bincount(int nchannels,
		    int ndaq,
		    unsigned int *event_hit,
		    float *event_time, 
		    float *mc_time,
		    unsigned int *hitcount, unsigned int *bincount,
		    float min_twidth, float tmin, float tmax,
		    int min_bin_content,
		    unsigned int *map_channel_id_to_hit_offset,
		    unsigned int *work_queues)
{
    int channel_id = threadIdx.x + blockDim.x * blockIdx.x;
	
    if (channel_id >= nchannels)
	return;

    float channel_hitcount = hitcount[channel_id];
    float channel_bincount = bincount[channel_id];
    float channel_event_time = event_time[channel_id];
    int channel_event_hit = event_hit[channel_id];
    
    unsigned int *work_queue = work_queues + map_channel_id_to_hit_offset[channel_id] * (ndaq + 1);
    unsigned int next_slot = work_queue[0];
	
    for (int i=0; i < ndaq; i++) {
	int read_offset = nchannels * i + channel_id;

	// Was this channel hit in the Monte Carlo?
	float channel_mc_time = mc_time[read_offset];
	if (channel_mc_time >= 1e8)
	    continue; // Nothing else to do
	
	// Is this channel inside the range of the PDF?
	float distance;
	if (channel_mc_time < tmin || channel_mc_time > tmax)
	    continue;  // Nothing else to do
		
	channel_hitcount += 1;

	// Was this channel hit in the event-of-interest?
	if (!channel_event_hit)
	    continue; // No need to update PDF value for unhit channel
    
	// Are we inside the minimum size bin?
	distance = fabsf(channel_mc_time - channel_event_time);
	if (distance < min_twidth/2.0) {
	    channel_bincount += 1;
	}

	// Add this hit to the work queue if we also need to sort it into the 
	// nearest_mc_list
	if (channel_bincount < min_bin_content) {
	    work_queue[next_slot] = read_offset;
	    next_slot++;
	}
    }

    hitcount[channel_id] = channel_hitcount;
    bincount[channel_id] = channel_bincount;
    if (channel_event_hit)
	work_queue[0] = next_slot;
}
	
__global__ void
accumulate_nearest_neighbor(int nhit,
			    int ndaq,
			    unsigned int *map_hit_offset_to_channel_id,
                            unsigned int *work_queues,
			    float *event_time,
			    float *mc_time,
			    float *nearest_mc, int min_bin_content)
{
    int hit_id = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (hit_id >= nhit)
	return;

    unsigned int *work_queue = work_queues + hit_id * (ndaq + 1);
    int queue_items = work_queue[0] - 1;

    int channel_id = map_hit_offset_to_channel_id[hit_id];
    float channel_event_time = event_time[channel_id];
    
    float distance_table[1000];
    int distance_table_len = 0;

    // Load existing distance table
    int offset = min_bin_content * hit_id;    
    for (int i=0; i < min_bin_content; i++) {
      float d = nearest_mc[offset + i];
      if (d > 1e8)
	break;

      distance_table[distance_table_len] = d;
      distance_table_len++;
    }
    
    // append new entries
    for (int i=0; i < queue_items; i++) {
	unsigned int read_offset = work_queue[i+1];
	float channel_mc_time = mc_time[read_offset];
	float distance = fabsf(channel_mc_time - channel_event_time);

	distance_table[distance_table_len] = distance;
	distance_table_len++;
    }

    // Sort table
    piksrt(distance_table_len, distance_table);
    
    // Copy first section of table back out to global memory
    distance_table_len = min(distance_table_len, min_bin_content);
    for (int i=0; i < distance_table_len; i++) {
      nearest_mc[offset + i] = distance_table[i];
    }
}

__global__ void
accumulate_nearest_neighbor_block(int nhit,
			    int ndaq,
			    unsigned int *map_hit_offset_to_channel_id,
                            unsigned int *work_queues,
			    float *event_time,
			    float *mc_time,
			    float *nearest_mc, int min_bin_content)
{
    int hit_id = blockIdx.x;
    
    __shared__ float distance_table[1000];
    __shared__ unsigned int *work_queue;
    __shared__ int queue_items;
    __shared__ int channel_id;
    __shared__ float channel_event_time;
    __shared__ int distance_table_len;
    __shared__ int offset;

    if (threadIdx.x == 0) {
      work_queue = work_queues + hit_id * (ndaq + 1);
      queue_items = work_queue[0] - 1;

      channel_id = map_hit_offset_to_channel_id[hit_id];
      channel_event_time = event_time[channel_id];
      distance_table_len = min_bin_content;
      offset = min_bin_content * hit_id;    
    }

    __syncthreads();

    // Load existing distance table
    for (int i=threadIdx.x; i < min_bin_content; i += blockDim.x) {
      float d = nearest_mc[offset + i];
      if (d > 1e8) {
	atomicMin(&distance_table_len, i);
	break;
      }
      distance_table[i] = d;
    }
    
    __syncthreads();

    // append new entries
    for (int i=threadIdx.x; i < queue_items; i += blockDim.x) {
	unsigned int read_offset = work_queue[i+1];
	float channel_mc_time = mc_time[read_offset];
	float distance = fabsf(channel_mc_time - channel_event_time);

	distance_table[distance_table_len + i] = distance;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
      distance_table_len += queue_items;
      // Sort table
      piksrt(distance_table_len, distance_table);
      // Copy first section of table back out to global memory
      distance_table_len = min(distance_table_len, min_bin_content);
    }
    
    __syncthreads();

    for (int i=threadIdx.x; i < distance_table_len; i += blockDim.x) {
      nearest_mc[offset + i] = distance_table[i];
    }
}



__global__ void
accumulate_moments(int time_only, int nchannels,
		   float *mc_time,
		   float *mc_charge,
		   float tmin, float tmax,
		   float qmin, float qmax,
		   unsigned int *mom0,
		   float *t_mom1,
		   float *t_mom2,
		   float *q_mom1,
		   float *q_mom2)

{
    int id = threadIdx.x + blockDim.x * blockIdx.x;
	
    if (id >= nchannels)
	return;
	
    // Was this channel hit in the Monte Carlo?
    float channel_mc_time = mc_time[id];

    // Is this channel inside the range of the PDF?
    if (time_only) {
	if (channel_mc_time < tmin || channel_mc_time > tmax)
	    return;  // Nothing else to do
	
	mom0[id] += 1;
	t_mom1[id] += channel_mc_time;
	t_mom2[id] += channel_mc_time*channel_mc_time;
    }
    else { // time and charge PDF
	float channel_mc_charge = mc_charge[id]; // int->float conversion because DAQ just returns an integer
	
	if (channel_mc_time < tmin || channel_mc_time > tmax ||
	    channel_mc_charge < qmin || channel_mc_charge > qmax)
	    return;  // Nothing else to do

	mom0[id] += 1;
	t_mom1[id] += channel_mc_time;
	t_mom2[id] += channel_mc_time*channel_mc_time;
	q_mom1[id] += channel_mc_charge;
	q_mom2[id] += channel_mc_charge*channel_mc_charge;
    }
}

static const float invroot2 = 0.70710678118654746f; // 1/sqrt(2)
static const float rootPiBy2 = 1.2533141373155001f; // sqrt(M_PI/2)

__global__ void
accumulate_kernel_eval(int time_only, int nchannels, unsigned int *event_hit,
		       float *event_time, float *event_charge, float *mc_time,
		       float *mc_charge,
		       float tmin, float tmax,
		       float qmin, float qmax,
		       float *inv_time_bandwidths,
		       float *inv_charge_bandwidths,
		       unsigned int *hitcount,
		       float *time_pdf_values,
		       float *charge_pdf_values)
		       
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;
	
    if (id >= nchannels)
	return;
	
    // Was this channel hit in the Monte Carlo?
    float channel_mc_time = mc_time[id];
	
    // Is this channel inside the range of the PDF?
    if (time_only) {
	if (channel_mc_time < tmin || channel_mc_time > tmax)
	    return;  // Nothing else to do
		
	// This MC information is contained in the PDF
	hitcount[id] += 1;
	  
	// Was this channel hit in the event-of-interest?
	int channel_event_hit = event_hit[id];
	if (!channel_event_hit)
	    return; // No need to update PDF value for unhit channel

	// Kernel argument
	float channel_event_time = event_time[id];
	float inv_bandwidth = inv_time_bandwidths[id];
	float arg = (channel_mc_time - channel_event_time) * inv_bandwidth;

	// evaluate 1D Gaussian normalized within time window
	float term = expf(-0.5f * arg * arg) * inv_bandwidth;

	float norm = tmax - tmin;
	if (inv_bandwidth > 0.0f) {
	  float loarg = (tmin - channel_mc_time)*inv_bandwidth*invroot2;
	  float hiarg = (tmax - channel_mc_time)*inv_bandwidth*invroot2;
	  norm = (erff(hiarg) - erff(loarg)) * rootPiBy2;
	}
	time_pdf_values[id] += term / norm;
    }
    else { // time and charge PDF
	float channel_mc_charge = mc_charge[id]; // int->float conversion because DAQ just returns an integer
	
	if (channel_mc_time < tmin || channel_mc_time > tmax ||
	    channel_mc_charge < qmin || channel_mc_charge > qmax)
	    return;  // Nothing else to do
		
	// This MC information is contained in the PDF
	hitcount[id] += 1;
	  
	// Was this channel hit in the event-of-interest?
	int channel_event_hit = event_hit[id];
	if (!channel_event_hit)
	    return; // No need to update PDF value for unhit channel


	// Kernel argument: time dim
	float channel_event_obs = event_time[id];
	float inv_bandwidth = inv_time_bandwidths[id];
	float arg = (channel_mc_time - channel_event_obs) * inv_bandwidth;

	float norm = tmax - tmin;
	if (inv_bandwidth > 0.0f) {
	  float loarg = (tmin - channel_mc_time)*inv_bandwidth*invroot2;
	  float hiarg = (tmax - channel_mc_time)*inv_bandwidth*invroot2;
	  norm = (erff(hiarg) - erff(loarg)) * rootPiBy2;
	}
	float term = expf(-0.5f * arg * arg);

	time_pdf_values[id] += term / norm;

	// Kernel argument: charge dim
	channel_event_obs = event_charge[id];
	inv_bandwidth = inv_charge_bandwidths[id];
	arg = (channel_mc_charge - channel_event_obs) * inv_bandwidth;
	
	norm = qmax - qmin;
	if (inv_bandwidth > 0.0f) {
	  float loarg = (qmin - channel_mc_charge)*inv_bandwidth*invroot2;
	  float hiarg = (qmax - channel_mc_charge)*inv_bandwidth*invroot2;
	  norm = (erff(hiarg) - erff(loarg)) * rootPiBy2;
	}

	term = expf(-0.5f * arg * arg);

	charge_pdf_values[id] += term / norm;
    }
}


} // extern "C"
