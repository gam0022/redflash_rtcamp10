#include <optixu/optixu_math_namespace.h>
#include "redflash.h"
#include "random.h"

using namespace optix;

// Scene wide variables
rtDeclareVariable(float, scene_epsilon, , );

RT_CALLABLE_PROGRAM void Pdf(MaterialParameter &mat, State &state, PerRayData_pathtrace &prd)
{
    prd.pdf = 1;
}

RT_FUNCTION float fresnel(float cos_theta_i, float cos_theta_t, float eta)
{
    const float rs = (cos_theta_i - cos_theta_t * eta) /
                     (cos_theta_i + eta * cos_theta_t);
    const float rp = (cos_theta_i * eta - cos_theta_t) /
                     (cos_theta_i * eta + cos_theta_t);

    return 0.5f * (rs * rs + rp * rp);
}

RT_CALLABLE_PROGRAM void Sample(MaterialParameter &mat, State &state, PerRayData_pathtrace &prd)
{
    const float3 w_out = prd.wo;
    float3 normal = state.ffnormal;
    float cos_theta_i = dot(w_out, normal);

    float eta = mat.eta;

    if (cos_theta_i > 0.0f)
    {
        // レイが空気からガラスに入る場合
    }
    else
    {
        // レイがガラスから空気に出る場合
        eta = 1.0 / eta;
        cos_theta_i = -cos_theta_i;
        normal = -normal;
    }

    float3 w_t;
    const bool tir = !refract(w_t, -w_out, normal, eta);
    const float cos_theta_t = -dot(normal, w_t);
    const float R = tir ? 1.0f : fresnel(cos_theta_i, cos_theta_t, eta);
    const float z = rnd(prd.seed);

    if (z <= R)
    {
        // Reflect
        prd.origin = state.hitpoint + normal * scene_epsilon * 10.0;
        prd.direction = reflect(-w_out, normal);
    }
    else
    {
        // Refract
        prd.origin = state.hitpoint - normal * scene_epsilon * 10.0;
        prd.direction = w_t;
    }

    /*
    float3 dir = -prd.wo;

    // 板ポリを突き抜けるようなレイを生成
    prd.origin = state.hitpoint + dir * 0.01f;

    // レイの方向は維持する
    prd.direction = dir;
    */
}

RT_CALLABLE_PROGRAM float3 Eval(MaterialParameter &mat, State &state, PerRayData_pathtrace &prd)
{
    float3 out = mat.albedo;

    return out;
}