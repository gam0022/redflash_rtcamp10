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

RT_CALLABLE_PROGRAM void Sample(MaterialParameter &mat, State &state, PerRayData_pathtrace &prd)
{
    float3 dir = -prd.wo;

    // 板ポリを突き抜けるようなレイを生成
    prd.origin = state.hitpoint + dir * 0.01f;

    // レイの方向は維持する
    prd.direction = dir;

    // scene_idを反転
    prd.scene_id = 1 - prd.scene_id;
}

RT_CALLABLE_PROGRAM float3 Eval(MaterialParameter &mat, State &state, PerRayData_pathtrace &prd)
{
    float3 out = mat.albedo;

    return out;
}