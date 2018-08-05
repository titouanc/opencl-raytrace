#ifndef ZOIDBERG_LINALG
#define ZOIDBERG_LINALG

#define between(lower, x, upper) ((lower) <= (x) && (x) <= (upper))


/**
 * @brief      Find the intersection between a semi-line and a line segment in 2D
 * @param[in]  origin     The origin of the semi-line
 * @param[in]  direction  The direction of the semi-line
 * @param[in]  p0         The first point of the segment
 * @param[in]  p1         The second point of the segment
 * @return     k (>=0) such that origin+k*direction lies between p0 and p1,
 *             or NaN if such a point does not exist
 *
 * @details    Use the following relation:
 *             origin + k*direction = p0 + l*(p1-p0)
 *             <=>
 *             k*direction + l*(p0-p1) = p0 - origin
 *             <=>
 *             let A = (direction, (p0-p1))
 *                 B = p0 - origin
 *             in (k, l)*A = B <=> (k, l) = A⁻¹B
 */
float line2seg2d(float2 origin, float2 direction, float2 p0, float2 p1)
{
    //                   A0 A1
    // Column matrix A: |x  x|
    //                  |y  y|
    float2 A0 = direction;
    float2 A1 = p0 - p1;
    float det = A0.x*A1.y - A0.y*A1.x;
    if (det == 0){
        return NAN;
    }

    // Row matrix A⁻¹: |x  y| Ai0
    //                 |x  y| Ai1
    float2 Ai0 = (float2)(A1.y, -A1.x) / det;
    float2 Ai1 = (float2)(-A0.y, A0.x) / det;
    float2 B = p0 - origin;
    float2 res = (float2)(dot(Ai0, B), dot(Ai1, B));
    return (0.0f <= res.s0 && between(0.0f, res.s1, 1.0f)) ? res.s0 : NAN;
}


/**
 * @brief      Find the intersection between a semi-line and a triangle in 3D
 * @param[in]  origin     The origin of the semi-line
 * @param[in]  direction  The direction of the semi-line
 * @param[in]  p0         The first point of the triangle
 * @param[in]  p1         The second point of the triangle
 * @param[in]  p2         The third point of the triangle
 * @return     k (>=0) such that origin+k*direction lies between p0, p1 and p2,
 *             or NaN if such a point does not exist
 *             
 * @details    Use the following relation:
 *             origin + k*direction = p0 + l*(p1-p0) + m*(p2-p0)
 *             <=>
 *             k*direction + l*(p0-p1) + m*(p0-p2) = p0 - origin
 *             <=>
 *             let A = (direction, (p0-p1), (p0-p2))
 *                 B = p0 - origin
 *             in (k, l, m)*A = B <=> (k, l, m) = A⁻¹B
 */
float line2tri3d(float3 origin, float3 direction, float3 p0, float3 p1, float3 p2)
{
    //                   A0 A1 A2
    // Column matrix A: |x  x  x|
    //                  |y  y  y|
    //                  |z  z  z|
    float3 A0 = direction;
    float3 A1 = (p0 - p1);
    float3 A2 = (p0 - p2);

    float det = A0.x * (A1.y*A2.z - A1.z*A2.y)
              - A0.y * (A1.x*A2.z - A1.z*A2.x)
              + A0.z * (A1.x*A2.y - A1.y*A2.x);

    if (det == 0){
        return NAN;
    }

    float3 B = p0 - origin;

    // Row matrix A⁻¹: |x  y  z| Ai0
    //                 |x  y  z| Ai1
    //                 |x  y  z| Ai2
    float3 Ai0 = (float3)( 1.0f * (A1.y*A2.z - A1.z*A2.y),
                          -1.0f * (A1.x*A2.z - A1.z*A2.x),
                           1.0f * (A1.x*A2.y - A1.y*A2.x)) / det;
    float3 Ai1 = (float3)(-1.0f * (A0.y*A2.z - A0.z*A2.y),
                           1.0f * (A0.x*A2.z - A0.z*A2.x),
                          -1.0f * (A0.x*A2.y - A0.y*A2.x)) / det;
    float3 Ai2 = (float3)( 1.0f * (A0.y*A1.z - A0.z*A1.y),
                          -1.0f * (A0.x*A1.z - A0.z*A1.x),
                           1.0f * (A0.x*A1.y - A0.y*A1.x)) / det;

    float3 res = (float3)(dot(Ai0, B), dot(Ai1, B), dot(Ai2, B));
    bool in_triangle = between(0.0f, res.s1, 1.0f) &&
                       between(0.0f, res.s2, 1.0f) &&
                       between(0.0f, res.s1+res.s2, 1.0f);
    return (0.0f <= res.s0 && in_triangle) ? res.s0 : NAN;
}

#endif
