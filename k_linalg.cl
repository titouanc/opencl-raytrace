#include "linalg.h"

// Kernelification of internal functions for unit testing
// To simplify things, since these function only takes (vector of) floats
// as inputs, they are passed as a flat buffer.

#define F2(buf, pos) (float2)(buf[2*pos], buf[2*pos+1])
#define F3(buf, pos) (float3)(buf[3*pos], buf[3*pos+1], buf[3*pos+2])

__kernel void k_line2seg2d(__global float *res, __global float *args)
{
    float2 origin = F2(args, 0);
    float2 direction = F2(args, 1);
    float2 p0 = F2(args, 2);
    float2 p1 = F2(args, 3);

    *res = line2seg2d(origin, direction, p0, p1);
}


__kernel void k_line2tri3d(__global float *res, __global float *args)
{
    float3 origin = F3(args, 0);
    float3 direction = F3(args, 1);
    float3 p0 = F3(args, 2);
    float3 p1 = F3(args, 3);
    float3 p2 = F3(args, 4);

    *res = line2tri3d(origin, direction, p0, p1, p2);
}
