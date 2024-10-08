#include <optixu/optixu_math_namespace.h>
#include "redflash.h"
#include "random.h"
#include <optix_world.h>

using namespace optix;

#define TAU 6.28318530718

rtDeclareVariable(float, time, , );

rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(uint, raymarching_iteration, , );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

rtDeclareVariable(float3, center, , );
rtDeclareVariable(float3, aabb_min, , );
rtDeclareVariable(float3, aabb_max, , );
rtDeclareVariable(float3, texcoord, attribute texcoord, );

// プライマリレイのDepthを利用した高速化用
rtDeclareVariable(PerRayData_pathtrace, current_prd, rtPayload, );

static __forceinline__ __device__ float3 abs_float3(float3 v)
{
    return make_float3(abs(v.x), abs(v.y), abs(v.z));
}

static __forceinline__ __device__ float3 max_float3(float3 v, float a)
{
    return make_float3(max(v.x, a), max(v.y, a), max(v.z, a));
}

void rot(float2& p, float a)
{
    // 行列バージョン（動かない）
    // p = mul(make_float2x2(cos(a), sin(a), -sin(a), cos(a)), p);

    p = cos(a) * p + sin(a) * make_float2(p.y, -p.x);
}

float sdBox(float3 p, float3 b)
{
    float3 q = abs_float3(p) - b;
    return length(max_float3(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float dBox(float3 p, float3 b)
{
    float3 q = abs_float3(p) - b;
    return length(max_float3(q, 0.0));
}

static __forceinline__ __device__ float dMenger(float3 z0, float3 offset, float scale) {
    float3 z = z0;
    float w = 1.0;
    float scale_minus_one = scale - 1.0;

    for (int n = 0; n < 3; n++) {
        z = abs_float3(z);

        // if (z.x < z.y) z.xy = z.yx;
        if (z.x < z.y)
        {
            float x = z.x;
            z.x = z.y;
            z.y = x;
        }

        // if (z.x < z.z) z.xz = z.zx;
        if (z.x < z.z)
        {
            float x = z.x;
            z.x = z.z;
            z.z = x;
        }

        // if (z.y < z.z) z.yz = z.zy;
        if (z.y < z.z)
        {
            float y = z.y;
            z.y = z.z;
            z.z = y;
        }

        z *= scale;
        w *= scale;

        z -= offset * scale_minus_one;

        float tmp = offset.z * scale_minus_one;
        if (z.z < -0.5 * tmp) z.z += tmp;
    }
    return (length(max_float3(abs_float3(z) - make_float3(1.0), 0.0)) - 0.01) / w;
}

float3 get_xyz(float4 p)
{
    return make_float3(p.x, p.y, p.z);
}

// not work...
void set_xyz(float4& a, float3 b)
{
    a.x = b.x;
    a.y = b.y;
    a.x = b.z;
}

float dMandelFast(float3 p, float scale, int n) {
    float4 q0 = make_float4(p, 1.);
    float4 q = q0;

    for (int i = 0; i < n; i++) {
        // q.xyz = clamp(q.xyz, -1.0, 1.0) * 2.0 - q.xyz;
        // set_xyz(q, clamp(get_xyz(q), -1.0, 1.0) * 2.0 - get_xyz(q));
        float4 tmp = clamp(q, -1.0, 1.0) * 2.0 - q;
        q.x = tmp.x;
        q.y = tmp.y;
        q.z = tmp.z;

        // q = q * scale / clamp( dot( q.xyz, q.xyz ), 0.3, 1.0 ) + q0;
        float3 q_xyz = get_xyz(q);
        q = q * scale / clamp(dot(q_xyz, q_xyz), 0.3, 1.0) + q0;
    }

    // return length( q.xyz ) / abs( q.w );
    return length(get_xyz(q)) / abs(q.w);
}

float fracf(float x)
{
    return x - floor(x);
}

float mod(float a, float b)
{
    return fracf(abs(a / b)) * abs(b);
}

float opRep(float p, float interval)
{
    return mod(p, interval) - interval * 0.5;
}

float3 opRep(float3 p, float3 interval)
{
    p.x = opRep(p.x, interval.x);
    p.y = opRep(p.y, interval.y);
    p.z = opRep(p.z, interval.z);
    return p;
}

float3 opRepXZ(float3 p, float2 interval)
{
    p.x = opRep(p.x, interval.x);
    p.z = opRep(p.z, interval.y);
    return p;
}

void opUnion(float4& m1, float4& m2)
{
    if (m2.x < m1.x) m1 = m2;
}

float4 dMengerDouble(float3 z0, float3 offset, float scale, float iteration, float width, float idOffset)
{
    float3 z = z0;
    float w = 1.0;
    float scale_minus_one = scale - 1.0;

    for (int n = 0; n < iteration; n++)
    {
        z = abs_float3(z);

        // if (z.x < z.y) z.xy = z.yx;
        if (z.x < z.y)
        {
            float x = z.x;
            z.x = z.y;
            z.y = x;
        }

        // if (z.x < z.z) z.xz = z.zx;
        if (z.x < z.z)
        {
            float x = z.x;
            z.x = z.z;
            z.z = x;
        }

        // if (z.y < z.z) z.yz = z.zy;
        if (z.y < z.z)
        {
            float y = z.y;
            z.y = z.z;
            z.z = y;
        }

        z *= scale;
        w *= scale;

        z -= offset * scale_minus_one;

        float tmp = offset.z * scale_minus_one;
        if (z.z < -0.5 * tmp) z.z += tmp;
    }

    float e = 0.05;
    float d0 = (dBox(z, make_float3(1, 1, 1)) - e) / w;
    float d1 = (dBox(z, make_float3(1 + width, width, 1 + width)) - e) / w;
    float4 m0 = make_float4(d0, 0 + idOffset, 0, 0);
    float4 m1 = make_float4(d1, 1 + idOffset, 0, 0);
    opUnion(m0, m1);

    return m0;
}

float4 inverseStereographic(float3 p)
{
    float k = 2.0 / (1.0 + dot(p, p));
    return make_float4(k * p, k - 1.0);
}
float3 stereographic(float4 p4)
{
    float k = 1.0 / (1.0 + p4.w);
    return k * make_float3(p4.x, p4.y, p4.z);
}

#define _INVERSION4D_ON 1

float4 map_id_rtcamp9(float3 pos, int scene_id)
{
    if (scene_id == 1)
    {
        return make_float4(length(pos) - 2, 1, 0, 0);
    }

    #if _INVERSION4D_ON
        float f = length(pos);
        float4 p4d = inverseStereographic(pos);

        // rot(p4d.zw, time / 5 * TAU);
        float2 p4d_zw = make_float2(p4d.z, p4d.w);
        rot(p4d_zw, (-time + 0.01) / 5 * TAU);
        p4d.z = p4d_zw.x;
        p4d.w = p4d_zw.y;

        float3 p = stereographic(p4d);
        // p = pos;
    #else
        float3 p = pos;
    #endif

    float _MengerUniformScale0 = 1;
    float3 _MengerOffset0 = make_float3(0.82, 1.17, 0.46);
    float _MengerScale0 = 2.37;
    float _MengerIteration0 = 4;

    float _MengerUniformScale1 = 0.7;
    float3 _MengerOffset1 = make_float3(0.88 + 0.1 * sin(time * TAU / 10), 1.52, 0.13);
    float _MengerScale1 = 2.37;
    float _MengerIteration1 = 2;

    float4 m0 = dMengerDouble(p / _MengerUniformScale0, _MengerOffset0, _MengerScale0, _MengerIteration0, 0.2, 0);
    m0.x *= _MengerUniformScale0;

    float4 m1 = dMengerDouble(p / _MengerUniformScale1, _MengerOffset1, _MengerScale1, _MengerIteration1, 0.1, 2);
    m1.x *= _MengerUniformScale1;

    opUnion(m0, m1);
    m0.z = p.y;
    m0.w = p.z;

    #if _INVERSION4D_ON
        float e = length(p);
        m0.x *= min(1.0, 0.7 / e) * max(1.0, f);
    #endif

    return m0;
}

#define M_Floor 0
#define M_IFS_Base 1
#define M_IFS_Emissive 2
#define M_S2_Box 3

#define phase(x) (floor(x) + .5 + .5 * cos(TAU * .5 * exp(-5. * mod(x, 1))))

float sdTowers(float3 pos)
{
    float s = 1;
    float d;

    // p.xz *= rotate2D(t * .2);
    float2 rep = make_float2(3, 3);
    float3 p = opRepXZ(pos, rep);

    float2 grid = floor(make_float2(pos.x, pos.z) / rep);
    float height = 3.2;

    p.y -= height;

    for (int j = 0; j < 9; j++) {
        s /= d = min(dot(p, p), 0.42) + 0.1;
        p = abs_float3(p) / d - 0.2;
        p.y -= height;
    }

    d = length(p) / s - height / s;

    // 下半分だけ切り取る
    d = max(d, pos.y - height);

    return d;
}

float4 map_id(float3 pos, int scene_id)
{
    float4 m0 = make_float4(100, 0, 0, 0);

    float beatPhase = phase(time);

    if (scene_id == 0)
    {
        // Floor
        // float4 m_floor = make_float4(pos.y, M_Floor, 0, 0);
        // opUnion(m0, m_floor);

        int _IFS_Iteration = 3;
        float3 _IFS_Rot = make_float3(0.8, 0.6, 0.7);
        float3 _IFS_Offset = make_float3(0.89, 2.21, 0.53);
        float3 _opRep = make_float3(20, 10, 20);

        float3 p1 = opRep(pos, _opRep);
        p1 -= _IFS_Offset;

        for (int i = 0; i < _IFS_Iteration; i++)
        {
            p1 = abs_float3(p1 + _IFS_Offset) - _IFS_Offset;

            float2 p1_xz = make_float2(p1.x, p1.z);
            rot(p1_xz, TAU * _IFS_Rot.x);
            p1.x = p1_xz.x;
            p1.z = p1_xz.y;

            float2 p1_zy = make_float2(p1.z, p1.y);
            rot(p1_zy, TAU * _IFS_Rot.y);
            p1.z = p1_zy.x;
            p1.y = p1_zy.y;

            float2 p1_xy = make_float2(p1.x, p1.y);
            rot(p1_xy, TAU * _IFS_Rot.z + beatPhase / 2.3);
            p1.x = p1_xy.x;
            p1.y = p1_xy.y;
        }

        float4 m_base = make_float4(dBox(p1, make_float3(1, 0.5, 0.5)), M_IFS_Base, 0, 0);
        float4 m_emissive = make_float4(dBox(p1, make_float3(1.1, 0.6, 0.1)), M_IFS_Emissive, 0, 0);

        opUnion(m0, m_base);
        opUnion(m0, m_emissive);
    }
    else if (scene_id == 1)
    {
        float d = sdTowers(pos - make_float3(0, 2, 0));
        m0 = make_float4(d, M_S2_Box, 0, 0);
    }

    return m0;
}

float map(float3 pos, int scene_id)
{
    return map_id(pos, scene_id).x;
}

#define calcNormal(p, dFunc, eps, scene_id) normalize(\
    make_float3( eps, -eps, -eps) * dFunc(p + make_float3( eps, -eps, -eps), scene_id) + \
    make_float3(-eps, -eps,  eps) * dFunc(p + make_float3(-eps, -eps,  eps), scene_id) + \
    make_float3(-eps,  eps, -eps) * dFunc(p + make_float3(-eps,  eps, -eps), scene_id) + \
    make_float3( eps,  eps,  eps) * dFunc(p + make_float3( eps,  eps,  eps), scene_id))

// https://www.shadertoy.com/view/lttGDn
float calcEdge(float3 p, float width, int scene_id)
{
    float edge = 0.0;
    float2 e = make_float2(width, 0.0f);

    // Take some distance function measurements from either side of the hit point on all three axes.
    float d1 = map(p + make_float3(width, 0.0f, 0.0f), scene_id), d2 = map(p - make_float3(width, 0.0f, 0.0f), scene_id);
    float d3 = map(p + make_float3(0.0f, width, 0.0f), scene_id), d4 = map(p - make_float3(0.0f, width, 0.0f), scene_id);
    float d5 = map(p + make_float3(0.0f, 0.0f, width), scene_id), d6 = map(p - make_float3(0.0f, 0.0f, width), scene_id);
    float d = map(p, scene_id) * 2.;	// The hit point itself - Doubled to cut down on calculations. See below.

    // Edges - Take a geometry measurement from either side of the hit point. Average them, then see how
    // much the value differs from the hit point itself. Do this for X, Y and Z directions. Here, the sum
    // is used for the overall difference, but there are other ways. Note that it's mainly sharp surface
    // curves that register a discernible difference.
    edge = abs(d1 + d2 - d) + abs(d3 + d4 - d) + abs(d5 + d6 - d);
    //edge = max(max(abs(d1 + d2 - d), abs(d3 + d4 - d)), abs(d5 + d6 - d)); // Etc.

    // Once you have an edge value, it needs to normalized, and smoothed if possible. How you
    // do that is up to you. This is what I came up with for now, but I might tweak it later.
    edge = smoothstep(0., 1., sqrt(edge / e.x * 2.));

    // Return the normal.
    // Standard, normalized gradient mearsurement.
    return edge;
}

RT_CALLABLE_PROGRAM void materialAnimation_Nop(MaterialParameter& mat, State& state, int scene_id)
{
    // nop
}

RT_CALLABLE_PROGRAM void materialAnimation_Raymarching(MaterialParameter& mat, State& state, int scene_id)
{
    float3 p = state.hitpoint;
    float4 m = map_id(p, scene_id);
    uint id = uint(m.y);

    if (id == M_Floor)
    {
        mat.albedo = make_float3(1.0, 1.0, 1.0);
        mat.roughness = 0.15;
        mat.metallic = 0.9;
    }
    else if (id == M_IFS_Base)
    {
        mat.albedo = make_float3(0.2, 0.2, 0.2);
        mat.roughness = 0.05;
        mat.metallic = 0.8;
    }
    else if (id == M_IFS_Emissive)
    {
        mat.albedo = make_float3(1.0, 1.0, 1.0);
        mat.roughness = 0.6;
        mat.metallic = 0.2;

        float a = saturate(cos((-p.z / 256 + time) * TAU));
        mat.emission += make_float3(0.2, 0.2, 4) * a;
    }
    else if (id == M_S2_Box)
    {
        mat.albedo = make_float3(0.7, 0.7, 0.7);
        mat.roughness = 0.7;
        mat.metallic = 0.1;
    }
}

RT_PROGRAM void intersect(int primIdx)
{
    float eps;
    float t = ray.tmin, d = 0.0;
    float3 p = ray.origin;

    if (current_prd.depth == 0)
    {
        t = max(current_prd.distance, t);
    }

    for (int i = 0; i < raymarching_iteration; i++)
    {
        p = ray.origin + t * ray.direction;
        d = map(p, current_prd.scene_id);
        t += d;
        eps = scene_epsilon * t;
        if (abs(d) < eps || t > ray.tmax)
        {
            break;
        }
    }

    // if (t < ray.tmax && rtPotentialIntersection(t))
    if (abs(d) < eps && rtPotentialIntersection(t))
    {
        shading_normal = geometric_normal = calcNormal(p, map, scene_epsilon, current_prd.scene_id);
        texcoord = make_float3(p.x, p.y, 0);
        rtReportIntersection(0);
    }
}

float calcSlope(float t0, float t1, float r0, float r1)
{
    return (r1 - r0) / max(t1 - t0, 1e-5);
}

RT_PROGRAM void intersect_AutoRelaxation(int primIdx)
{
    float t = ray.tmin;

    if (current_prd.depth == 0)
    {
        t = max(current_prd.distance, t);
    }

    float eps = scene_epsilon * t;
    float r = map(ray.origin + t * ray.direction, current_prd.scene_id);
    int i = 1;
    float z = r;
    float m = -1;
    float stepRelaxation = 0.2;

    while (t + r < ray.tmax          // miss
        && r > eps    // hit
        && i < raymarching_iteration)  // didn't converge
    {
        float T = t + z;
        float R = map(ray.origin + T * ray.direction, current_prd.scene_id);
        bool doBackStep = z > abs(R) + r;
        //bool doBackStep = t + abs(r) < T - abs(R);
        float M = calcSlope(t, T, r, R);
        m = doBackStep ? -1 : lerp(m, M, stepRelaxation);
        t = doBackStep ? t : T;
        r = doBackStep ? r : R;
        float omega = max(1.0, 2.0 / (1.0 - m));
        eps = scene_epsilon * t;
        z = max(eps, r * omega);
        ++i;
#ifdef ENABLE_DEBUG_UTILS
        // backStep += doBackStep ? 1 : 0;
#endif
    }

#ifdef ENABLE_DEBUG_UTILS
    // stepCount = i;
#endif

    float retT = t + r;
    //retT = min(retT, ray.tmax);

    // if (retT < ray.tmax && rtPotentialIntersection(retT))
    if (r <= eps && rtPotentialIntersection(retT))
    {
        float3 p = ray.origin + retT * ray.direction;
        shading_normal = geometric_normal = calcNormal(p, map, scene_epsilon, current_prd.scene_id);
        texcoord = make_float3(p.x, p.y, 0);
        rtReportIntersection(0);
    }
}

RT_PROGRAM void bounds(int, float result[6])
{
    optix::Aabb* aabb = (optix::Aabb*)result;
    aabb->m_min = aabb_min;
    aabb->m_max = aabb_max;
}