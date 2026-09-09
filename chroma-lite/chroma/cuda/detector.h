#ifndef __DETECTOR_H__
#define __DETECTOR_H__

struct Detector
{
    // Order in decreasing size to avoid alignment problems
    int *solid_id_to_channel_index;

    float *time_cdf_x;
    float *time_cdf_y;

    float *charge_cdf_x;
    float *charge_cdf_y;

    int nchannels;
    int time_cdf_len;
    int charge_cdf_len;
    float charge_unit; 
    // Convert charges to/from quantized integers with
    // q_int = (int) roundf(q / charge_unit )
    // q = q_int * charge_unit
};


#endif // __DETECTOR_H__
