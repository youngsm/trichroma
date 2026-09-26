// PTX probe for fill_state's FP64 wire-frame re-orthonormalization.

extern "C"
__global__ void
wire_frame_probe(int count,
                 const float3 *__restrict__ raw_u,
                 const float3 *__restrict__ raw_v,
                 double *__restrict__ output)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    const double ux = (double)raw_u[index].x;
    const double uy = (double)raw_u[index].y;
    const double uz = (double)raw_u[index].z;
    const double vx0 = (double)raw_v[index].x;
    const double vy0 = (double)raw_v[index].y;
    const double vz0 = (double)raw_v[index].z;
    const double un = 1.0 / sqrt(ux*ux + uy*uy + uz*uz);
    const double ux1 = ux*un, uy1 = uy*un, uz1 = uz*un;
    const double vdotu = vx0*ux1 + vy0*uy1 + vz0*uz1;
    const double vx1 = vx0 - vdotu*ux1;
    const double vy1 = vy0 - vdotu*uy1;
    const double vz1 = vz0 - vdotu*uz1;
    const double vn = 1.0 / sqrt(vx1*vx1 + vy1*vy1 + vz1*vz1);
    const double vx = vx1*vn, vy = vy1*vn, vz = vz1*vn;
    const double nx = uy1*vz - uz1*vy;
    const double ny = uz1*vx - ux1*vz;
    const double nz = ux1*vy - uy1*vx;
    double *row = output + 9 * index;
    row[0] = ux1; row[1] = uy1; row[2] = uz1;
    row[3] = vx; row[4] = vy; row[5] = vz;
    row[6] = nx; row[7] = ny; row[8] = nz;
}
