//-----------------------------------------------------------------------------
//
// redflash: Raymarching x Pathtracer
//
//-----------------------------------------------------------------------------

#ifdef __APPLE__
#  include <GLUT/glut.h>
#else
#  include <GL/glew.h>
#  if defined( _WIN32 )
#    include <GL/wglew.h>
#    include <GL/freeglut.h>
#  else
#    include <GL/glut.h>
#  endif
#endif

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

#include "redflash.h"
#include <sutil.h>
#include <Arcball.h>
#include <OptiXMesh.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdint.h>
#include <filesystem>

#include <string>
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <cstdlib>
#include <iomanip>
#include <thread>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace fs = std::filesystem;

using namespace optix;

const char* const SAMPLE_NAME = "redflash";

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

Context context = 0;

// NEE処理のコストも無視できないのでライトは消す
bool useLight = false;

// 起動オプションで設定するパラメーター
int width = 1920 / 2;
int height = 1080 / 2;
bool use_pbo = true;
bool flag_debug = false;

// 動画モード時の初回フレーム（ベンチマーク時）のsample_per_launch
int init_sample_per_launch = 3;

// 静止画モード用のパラメーター
bool auto_set_sample_per_launch = false;
double auto_set_sample_per_launch_scale = 0.95;
double last_frame_scale = 1.7;

// ランタイムに変化するパラメーター
bool flag_debug_render = false;

// 時間
double launch_time;
double animate_begin_time;
double animate_time = 0.0f;

// sampling
int max_depth = 5;
int rr_begin_depth = 1;// ロシアンルーレット開始のdepth（未使用）
int sample_per_launch = 1;
int frame_number = 1;
int total_sample = 0;

// Intersect Programs
Program pgram_intersection = 0;
Program pgram_bounding_box = 0;
Program pgram_intersection_raymarching_local = 0;
Program pgram_intersection_raymarching_world = 0;
Program pgram_bounding_box_raymarching = 0;
Program pgram_intersection_sphere = 0;
Program pgram_bounding_box_sphere = 0;

// Common Material
Program common_closest_hit = 0;
Program common_any_hit = 0;
Material common_material = 0;

int materialCount = 0;
optix::Buffer m_bufferMaterialParameters;
std::vector<MaterialParameter> materialParameters;

// Light Material
Program light_closest_hit = 0;
Material light_material = 0;
optix::Buffer m_bufferLightParameters;

// Post-processing
CommandList commandListWithDenoiser;
CommandList commandListWithoutDenoiser;
PostprocessingStage tonemapStage;
PostprocessingStage denoiserStage;
Buffer denoisedBuffer;
Buffer emptyBuffer;
Buffer trainingDataBuffer;

// Rendering
float tonemap_exposure = 2.5f;

// PostprocessingのTonemapを有効にするかどうか
bool use_post_tonemap = false;

bool denoiser_perf_mode = false;
int denoiser_perf_iter = 1;

// number of frames that show the original image before switching on denoising
int numNonDenoisedFrames = 0;

// Defines the amount of the original image that is blended with the denoised result
// ranging from 0.0 to 1.0
float denoiseBlend = 0.f;

// Defines which buffer to show.
// 0 - denoised 1 - original, 2 - tonemapped, 3 - albedo, 4 - normal
int showBuffer = 0;

// The denoiser mode.
// 0 - RGB only, 1 - RGB + albedo, 2 - RGB + albedo + normals
int denoiseMode = 2;

// The path to the training data file set with -t or empty
std::string training_file;

// The path to the second training data file set with -t2 or empty
std::string training_file_2;

// Toggles between using custom training data (if set) or the built in training data.
bool useCustomTrainingData = true;

// Toggles the custom data between the one specified with -t1 and -t2, if available.
bool useFirstTrainingDataPath = true;

// Contains info for the currently shown buffer
std::string bufferInfo;


// Camera state
float3         camera_up;
float3         camera_lookat;
float3         camera_eye;
float          camera_fov;
Matrix4x4      camera_rotate;
bool           camera_changed = true;
bool           postprocessing_needs_init = true;
sutil::Arcball arcball;

Matrix4x4 frame;
Matrix4x4 frame_inv;

// Mouse state
int2           mouse_prev_pos;
int            mouse_button;

// Scene
Group top_object_scene0;
Group top_object_scene1;

// Lightを動的にアップデートするための参照
std::vector<LightParameter> light_parameters;
std::vector<GeometryInstance> light_gis;
GeometryGroup light_group;

// 動的にシーンをアップデートするための参照
GeometryGroup raymarching_common_gg;

Group dynamic_scene0_group;
std::vector<Transform> dynamic_scene0_transforms;
std::vector<GeometryInstance > dynamic_scene0_gis;

// Ball
float3 ball_center;
Geometry ball_g;

float3 mandelBox_center;

// WASD移動
bool is_key_W_pressed = false;
bool is_key_A_pressed = false;
bool is_key_S_pressed = false;
bool is_key_D_pressed = false;
bool is_key_Q_pressed = false;
bool is_key_E_pressed = false;


//------------------------------------------------------------------------------
//
// Forward decls
//
//------------------------------------------------------------------------------

Buffer getOutputBuffer();
void destroyContext();
void registerExitHandler();
void createContext();
void loadGeometry();
void setupCamera();
void updateCamera();
void glutInitialize(int* argc, char** argv);
void glutRun();

void glutDisplay();
void glutKeyboardPress(unsigned char k, int x, int y);
void glutKeyboardPressUp(unsigned char k, int x, int y);
void glutMousePress(int button, int state, int x, int y);
void glutMouseMotion(int x, int y);
void glutResize(int w, int h);


//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

std::string resolveDataPath(const char* filename)
{
    std::vector<std::string> source_locations;

    std::string base_dir = std::string(sutil::samplesDir());

    // Potential source locations (in priority order)
    source_locations.push_back(fs::current_path().string() + "/" + filename);
    source_locations.push_back(fs::current_path().string() + "/data/" + filename);

    // 絶対パス指定は提出時のアセット漏れの原因になるのでコメントアウト
    // source_locations.push_back(base_dir + "/data/" + filename);

    // Visual Studioで実行したときの相対パス
    source_locations.push_back(fs::current_path().string() + "/../../data/" + filename);

    for (auto it = source_locations.cbegin(); it != source_locations.end(); ++it) {
        std::cout << "[info] resolvePath source_location: " + *it << std::endl;

        // Try to get source code from file
        if (fs::exists(*it))
        {
            return *it;
        }
    }

    // Wasn't able to find or open the requested file
    throw Exception("Couldn't open source file " + std::string(filename));
}

void loadTrainingFile(const std::string& path)
{
    if (path.length() == 0)
    {
        trainingDataBuffer->setSize(0);
        return;
    }

    using namespace std;
    ifstream fin(path.c_str(), ios::in | ios::ate | ios::binary);
    if (fin.fail())
    {
        fprintf(stderr, "Failed to load training file %s\n", path.c_str());
        return;
    }
    size_t size = static_cast<size_t>(fin.tellg());

    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp)
    {
        fprintf(stderr, "Failed to load training file %s\n", path.c_str());
        return;
    }

    trainingDataBuffer->setSize(size);

    char* data = reinterpret_cast<char*>(trainingDataBuffer->map());

    const bool ok = fread(data, 1, size, fp) == size;
    fclose(fp);

    trainingDataBuffer->unmap();

    if (!ok)
    {
        fprintf(stderr, "Failed to load training file %s\n", path.c_str());
        trainingDataBuffer->setSize(0);
    }
}

Buffer getOutputBuffer()
{
    return context["output_buffer"]->getBuffer();
}

Buffer getLinerBuffer()
{
    return context["liner_buffer"]->getBuffer();
}

Buffer getTonemappedBuffer()
{
    return context["tonemapped_buffer"]->getBuffer();
}

Buffer getAlbedoBuffer()
{
    return context["input_albedo_buffer"]->getBuffer();
}

Buffer getNormalBuffer()
{
    return context["input_normal_buffer"]->getBuffer();
}

Buffer getLinerDepthBuffer()
{
    return context["liner_depth_buffer"]->getBuffer();
}


void destroyContext()
{
    if (context)
    {
        context->destroy();
        context = 0;
    }
}


void registerExitHandler()
{
    // register shutdown handler
#ifdef _WIN32
    glutCloseFunc(destroyContext);  // this function is freeglut-only
#else
    atexit(destroyContext);
#endif
}

GeometryInstance createRaymrachingObject(const float3& center, const float3& bounds_size, RaymarchingMapType mapType)
{
    Geometry raymarching = context->createGeometry();
    raymarching->setPrimitiveCount(1u);

    if (mapType == dTower || mapType == dOcean)
    {
        raymarching->setIntersectionProgram(pgram_intersection_raymarching_world);
    }
    else
    {
        raymarching->setIntersectionProgram(pgram_intersection_raymarching_local);
    }

    raymarching->setBoundingBoxProgram(pgram_bounding_box_raymarching);

    raymarching["center"]->setFloat(center);
    raymarching["aabb_min"]->setFloat(center - bounds_size * 0.5f);
    raymarching["aabb_max"]->setFloat(center + bounds_size * 0.5f);
    raymarching["map_id"]->setInt(mapType);

    if (mapType == dBall)
    {
        ball_g = raymarching;
        context["ball_center"]->setFloat(center);
    }

    GeometryInstance gi = context->createGeometryInstance();
    gi->setGeometry(raymarching);
    return gi;
}

GeometryInstance createSphereObject(const float3& center, const float radius)
{
    Geometry sphere = context->createGeometry();
    sphere->setPrimitiveCount(1u);
    sphere->setIntersectionProgram(pgram_intersection_sphere);
    sphere->setBoundingBoxProgram(pgram_bounding_box_sphere);

    sphere["center"]->setFloat(center);
    sphere["radius"]->setFloat(radius);
    sphere["aabb_min"]->setFloat(center - radius);
    sphere["aabb_max"]->setFloat(center + radius);

    GeometryInstance gi = context->createGeometryInstance();
    gi->setGeometry(sphere);
    return gi;
}

Matrix4x4 createMatrix(
    const float3& center,
    const float3& scale,
    const float3& axis = make_float3(0.0f, 1.0f, 0.0f),
    const float radians = 0.0f
)
{
    // NOTE: OptiXでは行優先っぽいので、右から順番に適用される
    Matrix4x4 mat = Matrix4x4::translate(center) * Matrix4x4::rotate(radians, axis) * Matrix4x4::scale(scale);

    return mat;
}

GeometryInstance createMesh(
    const std::string& filename,
    const float3& center,
    const float3& scale,
    const float3& axis = make_float3(0.0f, 1.0f, 0.0f),
    const float radians = 0.0f)
{
    OptiXMesh mesh;
    mesh.context = context;
    mesh.use_tri_api = true;
    mesh.ignore_mats = false;

    // NOTE: registerMaterial で上書きするので、この指定は意味がない
    mesh.material = common_material;

    mesh.closest_hit = common_closest_hit;
    mesh.any_hit = common_any_hit;

    Matrix4x4 mat = createMatrix(center, scale, axis, radians);

    loadMesh(filename, mesh, mat);
    return mesh.geom_instance;
}

Transform createDynamicMesh(
    const std::string& filename,
    const float3& center,
    const float3& scale,
    const float3& axis = make_float3(0.0f, 1.0f, 0.0f),
    const float radians = 0.0f)
{
    OptiXMesh mesh;
    mesh.context = context;
    mesh.use_tri_api = true;
    mesh.ignore_mats = false;

    // NOTE: registerMaterial で上書きするので、この指定は意味がない
    mesh.material = common_material;

    mesh.closest_hit = common_closest_hit;
    mesh.any_hit = common_any_hit;

    loadMesh(filename, mesh);

    GeometryGroup geometry_group = context->createGeometryGroup();
    geometry_group->setAcceleration(context->createAcceleration("Trbvh"));
    geometry_group->addChild(mesh.geom_instance);

    Matrix4x4 mat = createMatrix(center, scale, axis, radians);

    Transform transform = context->createTransform();
    transform->setMatrix(false, mat.getData(), 0);
    transform->setChild(geometry_group);

    // FIXME: 外から指定できるようにしたい
    dynamic_scene0_gis.push_back(mesh.geom_instance);
    dynamic_scene0_transforms.push_back(transform);

    return transform;
}

void setupBSDF(std::vector<std::string>& bsdf_paths)
{
    const int bsdf_type_count = bsdf_paths.size();

    std::vector<std::string> ptxs;
    for (int i = 0; i < bsdf_type_count; ++i)
    {
        ptxs.push_back(sutil::getPtxString(SAMPLE_NAME, bsdf_paths[i].c_str()));
    }

    std::string var_prefix = "prgs_BSDF_";
    std::vector<std::string> bsdf_prg_names = { "Sample", "Eval", "Pdf" };

    for (auto it = bsdf_prg_names.cbegin(); it != bsdf_prg_names.end(); ++it) {
        optix::Buffer buffer_BSDF_prgs = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, bsdf_type_count);
        int* BSDF_prgs = (int*)buffer_BSDF_prgs->map(0, RT_BUFFER_MAP_WRITE_DISCARD);

        for (int i = 0; i < bsdf_type_count; ++i)
        {
            Program prg = context->createProgramFromPTXString(ptxs[i], *it);
            BSDF_prgs[i] = prg->getId();
        }

        buffer_BSDF_prgs->unmap();
        context[var_prefix + *it]->setBuffer(buffer_BSDF_prgs);
    }
}

void setupRaymarchingMapProgram(const char* ptx)
{
    std::string prefix = "RaymarchingMap_";
    std::vector<std::string> prg_names = { "Ball", "Tower", "Ocean", "MandelBox"};
    int prg_count = prg_names.size();

    optix::Buffer buffer_RaymarchingMap_prgs = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, prg_count);
    int* RaymarchingMap_prgs = (int*)buffer_RaymarchingMap_prgs->map(0, RT_BUFFER_MAP_WRITE_DISCARD);

    for (int i = 0; i < prg_count; ++i)
    {
        Program prg = context->createProgramFromPTXString(ptx, prefix + prg_names[i]);
        RaymarchingMap_prgs[i] = prg->getId();
    }

    buffer_RaymarchingMap_prgs->unmap();
    context["prgs_RaymarchingMap"]->setBuffer(buffer_RaymarchingMap_prgs);
}

void setupMaterialAnimationProgram(const char* ptx)
{
    std::string prefix = "materialAnimation_";
    std::vector<std::string> prg_names = { "Nop", "Laser", "Ocean", "Tower", "MandelBox"};
    int prg_count = prg_names.size();

    optix::Buffer buffer_MaterialAnimation_prgs = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, prg_count);
    int* MaterialAnimation_prgs = (int*)buffer_MaterialAnimation_prgs->map(0, RT_BUFFER_MAP_WRITE_DISCARD);

    for (int i = 0; i < prg_count; ++i)
    {
        Program prg = context->createProgramFromPTXString(ptx, prefix + prg_names[i]);
        MaterialAnimation_prgs[i] = prg->getId();
    }

    buffer_MaterialAnimation_prgs->unmap();
    context["prgs_MaterialAnimation"]->setBuffer(buffer_MaterialAnimation_prgs);
}

void createContext()
{
    context = Context::create();
    context->setRayTypeCount(2);
    context->setEntryPointCount(1);
    context->setStackSize(1800);
    context->setMaxTraceDepth(2);

    context["scene_id_init"]->setUint(0);
    context["scene_epsilon"]->setFloat(0.001f);
    context["raymarching_iteration"]->setUint(200);
    context["useLight"]->setUint(useLight ? 1 : 0);
    // context["rr_begin_depth"]->setUint( rr_begin_depth );
    context["max_depth"]->setUint(max_depth);
    context["sample_per_launch"]->setUint(sample_per_launch);
    context["total_sample"]->setUint(total_sample);
    context["usePostTonemap"]->setUint(use_post_tonemap);
    context["tonemap_exposure"]->setFloat(tonemap_exposure);

    Buffer output_buffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
    context["output_buffer"]->set(output_buffer);

    Buffer liner_buffer = sutil::createInputOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
    context["liner_buffer"]->set(liner_buffer);

    Buffer liner_depth_buffer = sutil::createInputOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
    context["liner_depth_buffer"]->set(liner_depth_buffer);

    Buffer tonemappedBuffer = sutil::createInputOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
    context["tonemapped_buffer"]->set(tonemappedBuffer);

    Buffer albedoBuffer = sutil::createInputOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
    context["input_albedo_buffer"]->set(albedoBuffer);

    // The normal buffer use float4 for performance reasons, the fourth channel will be ignored.
    Buffer normalBuffer = sutil::createInputOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
    context["input_normal_buffer"]->set(normalBuffer);

    denoisedBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
    emptyBuffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, 0, 0);
    trainingDataBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE, 0);

    // Setup programs
    const char* ptx = sutil::getPtxString(SAMPLE_NAME, "redflash.cu");

    context->setEntryPointCount(2);
    context->setRayGenerationProgram(0, context->createProgramFromPTXString(ptx, "pathtrace_camera"));
    context->setRayGenerationProgram(1, context->createProgramFromPTXString(ptx, "debug_camera"));

    context->setExceptionProgram(0, context->createProgramFromPTXString(ptx, "exception"));
    context->setMissProgram(0, context->createProgramFromPTXString(ptx, "envmap_miss"));
    context["bad_color"]->setFloat(1000000.0f, 0.0f, 1000000.0f); // Super magenta to make sure it doesn't get averaged out in the progressive rendering.

    // Common Materials
    common_material = context->createMaterial();
    common_closest_hit = context->createProgramFromPTXString(ptx, "closest_hit");
    common_any_hit = context->createProgramFromPTXString(ptx, "shadow");
    common_material->setClosestHitProgram(0, common_closest_hit);
    common_material->setAnyHitProgram(1, common_any_hit);

    // Light Materials
    light_material = context->createMaterial();
    light_closest_hit = context->createProgramFromPTXString(ptx, "light_closest_hit");
    light_material->setClosestHitProgram(0, light_closest_hit);

    // Raymarching programs
    ptx = sutil::getPtxString(SAMPLE_NAME, "intersect_raymarching.cu");
    pgram_bounding_box_raymarching = context->createProgramFromPTXString(ptx, "bounds");
    pgram_intersection_raymarching_local = context->createProgramFromPTXString(ptx, "intersectLocal");
    pgram_intersection_raymarching_world = context->createProgramFromPTXString(ptx, "intersectWorld");
    setupRaymarchingMapProgram(ptx);

    // Material Custom Program
    setupMaterialAnimationProgram(ptx);

    // Sphere programs
    ptx = sutil::getPtxString(SAMPLE_NAME, "intersect_sphere.cu");
    pgram_bounding_box_sphere = context->createProgramFromPTXString(ptx, "bounds");
    pgram_intersection_sphere = context->createProgramFromPTXString(ptx, "sphere_intersect");

    // BSDF
    std::vector<std::string> bsdf_paths{ "bsdf_diffuse.cu", "bsdf_disney.cu", "bsdf_glass.cu", "bsdf_portal.cu" };
    setupBSDF(bsdf_paths);
}

void setupPostprocessing()
{
    if (!tonemapStage)
    {
        // create stages only once: they will be reused in several command lists without being re-created
        denoiserStage = context->createBuiltinPostProcessingStage("DLDenoiser");

        if (trainingDataBuffer)
        {
            Variable trainingBuff = denoiserStage->declareVariable("training_data_buffer");
            trainingBuff->set(trainingDataBuffer);
        }

        if (use_post_tonemap)
        {
            tonemapStage = context->createBuiltinPostProcessingStage("TonemapperSimple");
            tonemapStage->declareVariable("input_buffer")->set(getOutputBuffer());
            tonemapStage->declareVariable("output_buffer")->set(getTonemappedBuffer());
            tonemapStage->declareVariable("exposure")->setFloat(tonemap_exposure);
            tonemapStage->declareVariable("gamma")->setFloat(2.2f);
        }

        denoiserStage->declareVariable("input_buffer")->set(use_post_tonemap ? getTonemappedBuffer() : getOutputBuffer());
        denoiserStage->declareVariable("output_buffer")->set(denoisedBuffer);
        denoiserStage->declareVariable("blend")->setFloat(denoiseBlend);
        denoiserStage->declareVariable("input_albedo_buffer");
        denoiserStage->declareVariable("input_normal_buffer");
    }

    if (commandListWithDenoiser)
    {
        commandListWithDenoiser->destroy();
        commandListWithoutDenoiser->destroy();
    }

    // Create two command lists with two postprocessing topologies we want:
    // One with the denoiser stage, one without. Note that both share the same
    // tonemap stage.

    commandListWithDenoiser = context->createCommandList();
    commandListWithDenoiser->appendLaunch(0, width, height);
    if (use_post_tonemap)
        commandListWithDenoiser->appendPostprocessingStage(tonemapStage, width, height);
    commandListWithDenoiser->appendPostprocessingStage(denoiserStage, width, height);
    commandListWithDenoiser->finalize();

    commandListWithoutDenoiser = context->createCommandList();
    commandListWithoutDenoiser->appendLaunch(0, width, height);
    if (use_post_tonemap)
        commandListWithoutDenoiser->appendPostprocessingStage(tonemapStage, width, height);
    commandListWithoutDenoiser->finalize();

    postprocessing_needs_init = false;
}

void registerMaterial(GeometryInstance& gi, MaterialParameter& mat,
    MaterialAnimationProgramType material_animation_program_id = MaterialAnimationProgramType::Nop, bool isLight = false, bool isReuse = false)
{
    if (!isReuse)
    {
        materialParameters.push_back(mat);
    }

    gi->setMaterialCount(1);
    gi->setMaterial(0, isLight ? light_material : common_material);
    gi["material_id"]->setInt(isReuse ? materialCount - 1 : materialCount++);
    gi["bsdf_id"]->setInt(mat.bsdf);
    gi["material_animation_program_id"]->setInt(material_animation_program_id);
}

void updateMaterialParameters()
{
    MaterialParameter* dst = static_cast<MaterialParameter*>(m_bufferMaterialParameters->map(0, RT_BUFFER_MAP_WRITE_DISCARD));
    for (size_t i = 0; i < materialParameters.size(); ++i, ++dst) {
        MaterialParameter mat = materialParameters[i];

        dst->albedo = mat.albedo;
        dst->emission = mat.emission;
        dst->metallic = mat.metallic;
        dst->subsurface = mat.subsurface;
        dst->specular = mat.specular;
        dst->specularTint = mat.specularTint;
        dst->roughness = mat.roughness;
        dst->anisotropic = mat.anisotropic;
        dst->sheen = mat.sheen;
        dst->sheenTint = mat.sheenTint;
        dst->clearcoat = mat.clearcoat;
        dst->clearcoatGloss = mat.clearcoatGloss;
        dst->eta = mat.eta;
        dst->bsdf = mat.bsdf;
        dst->albedoID = mat.albedoID;
    }
    m_bufferMaterialParameters->unmap();
}

void updateLightParameters(const std::vector<LightParameter>& lightParameters)
{
    LightParameter* dst = static_cast<LightParameter*>(m_bufferLightParameters->map(0, RT_BUFFER_MAP_WRITE_DISCARD));
    for (size_t i = 0; i < lightParameters.size(); ++i, ++dst) {
        LightParameter mat = lightParameters[i];

        dst->position = mat.position;
        dst->emission = mat.emission;
        dst->radius = mat.radius;
        dst->area = mat.area;
        dst->u = mat.u;
        dst->v = mat.v;
        dst->normal = mat.normal;
        dst->lightType = mat.lightType;
    }
    m_bufferLightParameters->unmap();
}

GeometryGroup createStaticGeometryCommon()
{
    MaterialParameter mat;
    std::vector<GeometryInstance> gis;

    // Mesh portal
    std::string mesh_file = resolveDataPath("mesh/portal.obj");
    gis.push_back(createMesh(mesh_file, make_float3(0.0f), make_float3(1.0f)));
    mat.bsdf = PORTAL;
    mat.albedo = make_float3(1.0f, 1.0f, 1.0f);
    mat.metallic = 0.8f;
    mat.roughness = 0.05f;
    registerMaterial(gis.back(), mat);

    GeometryGroup gg = context->createGeometryGroup(gis.begin(), gis.end());
    gg->setAcceleration(context->createAcceleration("Trbvh"));
    return gg;
}

GeometryGroup createStaticGeometryScene0()
{
    MaterialParameter mat;
    std::vector<GeometryInstance> gis;

    // Mesh Room
    std::string mesh_file = resolveDataPath("mesh/room.obj");
    gis.push_back(createMesh(mesh_file, make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f)));
    mat.albedo = make_float3(1.0f, 1.0f, 1.0f);
    mat.metallic = 0.8f;
    mat.roughness = 0.05f;
    registerMaterial(gis.back(), mat);

    // Mesh Monkey
    mesh_file = resolveDataPath("mesh/monkey.obj");
    gis.push_back(createMesh(mesh_file, make_float3(0.0f, 1.0f, 5.0f), make_float3(1.0f), make_float3(0, 1, 0), TAU * 0.5));
    mat.albedo = make_float3(236.0, 176.0, 0.0) / 255.0;
    mat.metallic = 0.8f;
    mat.roughness = 0.0f;
    registerMaterial(gis.back(), mat);

    GeometryGroup gg = context->createGeometryGroup(gis.begin(), gis.end());
    gg->setAcceleration(context->createAcceleration("Trbvh"));
    return gg;
}

GeometryGroup createStaticGeometryScene1()
{
    MaterialParameter mat;
    std::vector<GeometryInstance> gis;

    // Mesh Ocean
    std::string mesh_file = resolveDataPath("mesh/plane.obj");
    gis.push_back(createMesh(mesh_file, make_float3(0.0f, -8.0f, 0.0f), make_float3(10000.0f, 1.0f, 10000.0f)));
    mat.bsdf = GLASS;
    mat.albedo = make_float3(10, 100, 252) / 255.;
    mat.metallic = 1.0;
    mat.roughness = 0.0;
    mat.eta = 1.1f;;
    registerMaterial(gis.back(), mat, Ocean);

    GeometryGroup gg = context->createGeometryGroup(gis.begin(), gis.end());
    gg->setAcceleration(context->createAcceleration("Trbvh"));
    return gg;
}

Group createDynamicGeometryScene0()
{
    MaterialParameter mat;

    Group group = context->createGroup();
    group->setAcceleration(context->createAcceleration("Trbvh"));

    // Mesh door_base
    std::string mesh_file = resolveDataPath("mesh/door_base.obj");
    Transform door_base = createDynamicMesh(mesh_file, make_float3(-0.42f, 0.0f, -0.4f), make_float3(1.0f), make_float3(0.0f, 1.0f, 0.0f), TAU * -0.3f);
    mat.albedo = make_float3(1.0f, 1.0f, 1.0f);
    mat.metallic = 0.05f;
    mat.roughness = 0.95f;
    registerMaterial(dynamic_scene0_gis.back(), mat);
    group->addChild(door_base);

    // Mesh door_glass
    mesh_file = resolveDataPath("mesh/door_glass.obj");
    Transform door_glass = createDynamicMesh(mesh_file, make_float3(-0.42f, 0.0f, -0.4f), make_float3(1.0f), make_float3(0.0f, 1.0f, 0.0f), TAU * -0.3f);
    mat.bsdf = GLASS;
    mat.albedo = make_float3(0.5f, 1.0f, 0.5f);
    mat.eta = 1.45;
    registerMaterial(dynamic_scene0_gis.back(), mat);
    group->addChild(door_glass);

    // Mesh door_handle
    mesh_file = resolveDataPath("mesh/door_handle.obj");
    Transform door_handle = createDynamicMesh(mesh_file, make_float3(0.84f, 0.96f, 0.022f), make_float3(1.0f), make_float3(0.0f, 1.0f, 0.0f), TAU * -0.3f);
    mat.bsdf = DISNEY;
    mat.albedo = make_float3(236.0, 176.0, 0.0) / 255.0;
    mat.metallic = 0.95f;
    mat.roughness = 0.01f;
    registerMaterial(dynamic_scene0_gis.back(), mat);
    group->addChild(door_handle);

    bool isReuseMaterial = false;

    // レーザー
    mesh_file = resolveDataPath("mesh/quad_bottom.obj");

    for (int i = 0; i < 20; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            Transform transform = createDynamicMesh(mesh_file, make_float3(-3.0 + 6 * j, 0.0f, -10 + i + 0.5), make_float3(0.1f, 8.0, 0.1), make_float3(0.0f, 0.0f, 1.0f), TAU * 0.1f * i);
            mat.emission = make_float3(0.1, 0.1, 1.0);
            registerMaterial(dynamic_scene0_gis.back(), mat, Laser, false, isReuseMaterial);
            group->addChild(transform);
            isReuseMaterial = true;
        }
    }

    // うまくいかない
    for (auto transform = dynamic_scene0_transforms.begin(); transform != dynamic_scene0_transforms.end(); ++transform)
    {
        // group->addChild(transform);
    }

    return group;
}

GeometryGroup createRaymarchingGeometryCommon()
{
    MaterialParameter mat;

    std::vector<GeometryInstance> gis;

    // Raymarcing
    gis.push_back(createRaymrachingObject(
        make_float3(0.0f, 1.0, -4),
        make_float3(0.58f),
        dBall));
    mat.albedo = make_float3(1.0);
    mat.metallic = 1.0;
    mat.roughness = 0.0;
    registerMaterial(gis.back(), mat);

    GeometryGroup gg = context->createGeometryGroup(gis.begin(), gis.end());
    gg->setAcceleration(context->createAcceleration("Trbvh"));
    return gg;
}

GeometryGroup createRaymarchingGeometryScene1()
{
    MaterialParameter mat;

    std::vector<GeometryInstance> gis;

    // Raymarcing Tower
    gis.push_back(createRaymrachingObject(
        make_float3(0.0f, -2.0f, 0.0f),
        make_float3(10000.0f, 36.0f, 10000.0f),
        dTower));
    mat.albedo = make_float3(107 / 255.0, 142 / 255.0, 35 / 255.0);
    mat.metallic = 0.8f;
    mat.roughness = 0.05f;
    registerMaterial(gis.back(), mat, Tower);

    // Large MandelBox
    mandelBox_center = make_float3(0.0f, 40.0f, 40.0f);
    gis.push_back(createRaymrachingObject(
        mandelBox_center,
        make_float3(40.5),
        dMandelBox));
    mat.albedo = make_float3(0.6);
    mat.metallic = 0.8f;
    mat.roughness = 0.05f;
    registerMaterial(gis.back(), mat, MandelBox);
    context["mandelBox_center"]->setFloat(mandelBox_center);

    // Raymarcing Ocean
    /*
    gis.push_back(createRaymrachingObject(
        make_float3(0.0f, -8.0f, 0.0f),
        make_float3(10.0f, 0.1f, 10.0f),
        Ocean));
    mat.albedo = make_float3(0.2, 0.2, 0.7);
    mat.metallic = 1.0;
    mat.roughness = 0.0;
    registerMaterial(gis.back(), mat);
    */

    GeometryGroup gg = context->createGeometryGroup(gis.begin(), gis.end());
    gg->setAcceleration(context->createAcceleration("Trbvh"));
    return gg;
}

GeometryGroup createGeometryLight()
{
    {
        LightParameter light;
        light.lightType = SPHERE;
        light.position = make_float3(0.0f, 0.0f, 0.0f);
        light.radius = 0.05f;
        light.emission = make_float3(0.7f, 0.7f, 20.0f);
        if (useLight) light_parameters.push_back(light);
    }

    {
        LightParameter light;
        light.lightType = SPHERE;
        light.position = make_float3(3.8f, 161.4f, 200.65f);
        light.radius = 1.0f;
        light.emission = make_float3(20.0f, 10.00f, 5.00f) * 2;
        if (useLight) light_parameters.push_back(light);
    }

    int index = 0;
    for (auto light = light_parameters.begin(); light != light_parameters.end(); ++light)
    {
        light->area = 4.0f * M_PIf * light->radius * light->radius;
        light->normal = optix::normalize(light->normal);

        light_gis.push_back(createSphereObject(light->position, light->radius));
        light_gis.back()["lightMaterialId"]->setInt(index);

        MaterialParameter mat;
        mat.emission = light->emission;
        registerMaterial(light_gis.back(), mat, MaterialAnimationProgramType::Nop, true);

        ++index;
    }

    // Create geometry group
    GeometryGroup light_group = context->createGeometryGroup(light_gis.begin(), light_gis.end());
    light_group->setAcceleration(context->createAcceleration("Trbvh"));

    // Create sysLightParameters
    m_bufferLightParameters = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
    m_bufferLightParameters->setElementSize(sizeof(LightParameter));
    m_bufferLightParameters->setSize(light_parameters.size());
    updateLightParameters(light_parameters);
    context["sysNumberOfLights"]->setInt(light_parameters.size());
    context["sysLightParameters"]->setBuffer(m_bufferLightParameters);

    return light_group;
}

float sinFbm(float time)
{
    return sin(time) + 0.5 * sin(2.0 * time) + 0.25 * sin(4.0 * time);
}

float3 sinFbm3(float time)
{
    float t = time * TAU;
    return make_float3(sinFbm(t), sinFbm(t + 2), sinFbm(t + 3));
}

void updateGeometryLight(float time)
{
    int index = 0;
    for (auto light = light_parameters.begin(); light != light_parameters.end(); ++light)
    {
        if (light->lightType == LightType::SPHERE)
        {
            auto sphere = light_gis[index]->getGeometry();
            auto center = light->position;
            auto radius = light->radius;

            sphere["center"]->setFloat(center);
            sphere["radius"]->setFloat(radius);
            sphere["aabb_min"]->setFloat(center - radius);
            sphere["aabb_max"]->setFloat(center + radius);
        }

        ++index;
    }

    updateLightParameters(light_parameters);

    light_group->getAcceleration()->markDirty();
    light_group->getContext()->launch(0, 0, 0);
}

void setupScene()
{
    GeometryGroup static_common_gg = createStaticGeometryCommon();
    light_group = createGeometryLight();
    raymarching_common_gg = createRaymarchingGeometryCommon();

    // Scene0
    GeometryGroup static_scene0_gg = createStaticGeometryScene0();
    dynamic_scene0_group = createDynamicGeometryScene0();
    
    top_object_scene0 = context->createGroup();
    top_object_scene0->setAcceleration(context->createAcceleration("Trbvh"));
    top_object_scene0->addChild(raymarching_common_gg);
    top_object_scene0->addChild(static_common_gg);
    top_object_scene0->addChild(static_scene0_gg);
    top_object_scene0->addChild(dynamic_scene0_group);
    top_object_scene0->addChild(light_group);
    context["top_object_scene0"]->set(top_object_scene0);

    // Scene1
    GeometryGroup raymarching_scene1_gg = createRaymarchingGeometryScene1();
    GeometryGroup static_scene1_gg = createStaticGeometryScene1();

    top_object_scene1 = context->createGroup();
    top_object_scene1->setAcceleration(context->createAcceleration("Trbvh"));
    top_object_scene1->addChild(static_common_gg);
    top_object_scene1->addChild(static_scene1_gg);
    top_object_scene1->addChild(raymarching_common_gg);
    top_object_scene1->addChild(raymarching_scene1_gg);
    top_object_scene1->addChild(light_group);
    context["top_object_scene1"]->set(top_object_scene1);

    // Envmap
    const float3 default_color = make_float3(1.0f, 1.0f, 1.0f);
    const std::string texpath0 = resolveDataPath("polyhaven/fireplace_4k.hdr");
    context["envmap0"]->setTextureSampler(sutil::loadTexture(context, texpath0, default_color));

    const std::string texpath1 = resolveDataPath("polyhaven/table_mountain_1_puresky_4k.hdr");
    context["envmap1"]->setTextureSampler(sutil::loadTexture(context, texpath1, default_color));

    // Material Parameters
    m_bufferMaterialParameters = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
    m_bufferMaterialParameters->setElementSize(sizeof(MaterialParameter));
    m_bufferMaterialParameters->setSize(materialParameters.size());
    updateMaterialParameters();
    context["sysMaterialParameters"]->setBuffer(m_bufferMaterialParameters);
}

void setupCamera()
{
    camera_fov = 35;
    camera_up = make_float3(0.0f, 1.0f, 0.0f);

    // 中心
    camera_eye = make_float3(0, 2.0f, -5.0f);
    camera_lookat = make_float3(0, 1.0f, 0.0f);

    camera_rotate = Matrix4x4::identity();
}

float saturate(float x)
{
    return clamp(x, 0.0, 1.0);
}

// アニメーションの実装
void updateFrame(float time)
{
    // NOTE: falseにすれば自由カメラになる
    bool update_camera = true;

    bool update_dynamic = true;
    float vignetteIntensity = 0.9;

    float3 eye_shake = 0.05f * sinFbm3((time - 1) / 10.0);
    ball_center = make_float3(0.0, 1.0, -4.5 + time);

    uint scene_id_init = 0;

    if (update_camera)
    {
        float t;
        camera_up = make_float3(0.0f, 1.0f, 0.0f);
        camera_fov = 90.0f;
        camera_lookat = make_float3(0.0f, 1.0f, 0.0f);

        camera_eye = make_float3(2.0f, 1.8, -5.0f) * eye_shake;

        if (time < 2)
        {
            // 部屋全体のカメラ
            camera_eye = make_float3(0.0 + time, 2.0, -9.0f + time * 2.0) + eye_shake;
        }
        else if (time < 4)
        {
            // 扉が開き、横から見るカメラ
            camera_eye = make_float3(2.0f, 1.8, -2.5 + time * 0.3) + eye_shake;
            // camera_lookat = ball_center;
        }
        else if (time < 6)
        {
            t = time - 4;
            // ターゲットを追従し、ターゲットと一緒に次の世界に移動
            eye_shake.z = 0;
            camera_eye = make_float3(0.0f, 1.1, -1 + t) + eye_shake;
            camera_lookat = ball_center;
        }
        else if (time < 7)
        {
            // 外の世界からターゲットを追尾するカメラ
            camera_eye = make_float3(0.5f, 1.2, 2.0) + eye_shake;
            camera_lookat = ball_center;
        }
        else if (time < 9)
        {
            t = time - 7;
            float e = easeInOutCubic(t / 2);
            // 外の世界をを見せるカメラ
            // 海の下からの上昇、MandelBoxを見上げる
            camera_eye = make_float3(24.0f, -10.0 + 5 * e, 24.0) + eye_shake;
            camera_fov = lerp(30, 90, e);
            float3 t0 = make_float3(0, 1.0f, 0);
            camera_lookat = lerp(t0, mandelBox_center, e);
        }
        else if (time < 10 + 2.)
        {
            t = saturate(time - 9);
            // MandelBox俯瞰視点
            float e = easeInOutCubic(t);
            camera_eye = mandelBox_center + make_float3(0, e * -40, -22 -e * 40) + eye_shake;
            camera_lookat = mandelBox_center + make_float3(0, e * -10, 0);
            camera_fov = lerp(30, 90, e);
        }

        if (time >= 5)
        {
            scene_id_init = 1;
        }
    }

    if (update_dynamic)
    {
        // ドアの開閉
        // float rad = TAU * easeInOutCubic(clamp(time - 2, 0.0, 2.0) / 2.0f) * 0.4;
        float rad = smoothstep(2, 4, time) * TAU * 0.35;
        Matrix4x4 mat = createMatrix(make_float3(-0.42, 0.0, -0.01), make_float3(1.0f), make_float3(0.0f, 1.0f, 0.0f), rad);
        dynamic_scene0_transforms[0]->setMatrix(false, mat.getData(), false);

        Matrix4x4 mat_glass = createMatrix(make_float3(0.0f, 0.0f, -0.04f), make_float3(1.0f));
        mat_glass = mat * mat_glass;
        dynamic_scene0_transforms[1]->setMatrix(false, mat_glass.getData(), false);

        rad = smoothstep(1.5, 2, time) * TAU * 0.25;
        Matrix4x4 mat_handle = createMatrix(make_float3(0.84f, 0.96f, 0.022f), make_float3(1.0f), make_float3(0.0f, 0.0f, 1.0f), rad);
        mat_handle = mat * mat_handle;
        dynamic_scene0_transforms[2]->setMatrix(false, mat_handle.getData(), false);

        // レーザー
        for (int i = 0; i < 20; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                rad = 0.7 * sin(TAU * (0.5 * time + i * 0.05));
                if (j == 0) rad = -rad;
                mat = createMatrix(make_float3(-4.0 + 8 * j, -0.1f, -10 + i + 0.5), make_float3(0.1f, 8.0, 0.1), make_float3(0.0f, 0.0f, 1.0f), rad);
                dynamic_scene0_transforms[3 + i * 2 + j]->setMatrix(false, mat.getData(), false);
            }
        }


        dynamic_scene0_group->getAcceleration()->markDirty();
        dynamic_scene0_group->getContext()->launch(0, 0, 0);

        context["ball_center"]->setFloat(ball_center);
        ball_g["center"]->setFloat(ball_center);
        ball_g["aabb_min"]->setFloat(ball_center - 0.5f);
        ball_g["aabb_max"]->setFloat(ball_center + 0.5f);

        raymarching_common_gg->getAcceleration()->markDirty();
        raymarching_common_gg->getContext()->launch(0, 0, 0);
    }

    if (useLight) updateGeometryLight(time);

    if (useLight || update_dynamic)
    {
        top_object_scene0->getAcceleration()->markDirty();
        top_object_scene0->getContext()->launch(0, 0, 0);

        top_object_scene1->getAcceleration()->markDirty();
        top_object_scene1->getContext()->launch(0, 0, 0);
    }

    camera_changed = true;
    context["scene_id_init"]->setUint(scene_id_init);
    context["time"]->setFloat(time);
    context["vignetteIntensity"]->setFloat(vignetteIntensity);
}


void updateCamera()
{
    const float aspect_ratio = static_cast<float>(width) / static_cast<float>(height);

    float3 camera_u, camera_v, camera_w;
    sutil::calculateCameraVariables(
        camera_eye, camera_lookat, camera_up, camera_fov, aspect_ratio,
        camera_u, camera_v, camera_w, /*fov_is_vertical*/ true);

    // 左手系Y-UPにしたいので反転
    camera_u = -camera_u;

    frame = Matrix4x4::fromBasis(
        normalize(camera_u),
        normalize(camera_v),
        normalize(-camera_w),
        camera_lookat);
    frame_inv = frame.inverse();
    // Apply camera rotation twice to match old SDK behavior
    const Matrix4x4 trans = frame * camera_rotate * camera_rotate * frame_inv;

    camera_eye = make_float3(trans * make_float4(camera_eye, 1.0f));
    camera_lookat = make_float3(trans * make_float4(camera_lookat, 1.0f));
    // camera_up     = make_float3( trans*make_float4( camera_up,     0.0f ) );

    sutil::calculateCameraVariables(
        camera_eye, camera_lookat, camera_up, camera_fov, aspect_ratio,
        camera_u, camera_v, camera_w, true);

    // 左手系Y-UPにしたいので反転
    camera_u = -camera_u;

    camera_rotate = Matrix4x4::identity();

    frame_number++;
    total_sample += sample_per_launch;

    if (camera_changed) // reset accumulation
    {
        frame_number = 1;
        total_sample = 0;
    }

    camera_changed = false;

    context["frame_number"]->setUint(frame_number);
    context["total_sample"]->setUint(total_sample);

    context["eye"]->setFloat(camera_eye);
    context["U"]->setFloat(camera_u);
    context["V"]->setFloat(camera_v);
    context["W"]->setFloat(camera_w);

    const Matrix4x4 current_frame_inv = Matrix4x4::fromBasis(
        normalize(camera_u),
        normalize(camera_v),
        normalize(-camera_w),
        camera_lookat).inverse();
    Matrix3x3 normal_matrix = make_matrix3x3(current_frame_inv);
    context["normal_matrix"]->setMatrix3x3fv(false, normal_matrix.getData());
}


void glutInitialize(int* argc, char** argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutInitWindowPosition(100, 100);
    glutCreateWindow(SAMPLE_NAME);
    glutHideWindow();
}


void glutRun()
{
    // Initialize GL state
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glViewport(0, 0, width, height);

    glutShowWindow();
    glutReshapeWindow(width, height);

    // register glut callbacks
    glutDisplayFunc(glutDisplay);
    glutIdleFunc(glutDisplay);
    glutReshapeFunc(glutResize);
    glutKeyboardFunc(glutKeyboardPress);
    glutKeyboardUpFunc(glutKeyboardPressUp);
    glutMouseFunc(glutMousePress);
    glutMotionFunc(glutMouseMotion);

    registerExitHandler();

    glutMainLoop();
}


//------------------------------------------------------------------------------
//
//  GLUT callbacks
//
//------------------------------------------------------------------------------

void fpsCameraMove(float3& camera_local_offset, float speed)
{
    float4 offset = make_float4(camera_local_offset, 0.0f);
    offset = frame * offset;
    float3 offset_v3 = { offset.x, offset.y, offset.z };
    offset_v3 *= speed;
    camera_eye += offset_v3;
    camera_lookat += offset_v3;
    camera_changed = true;
}

void glutDisplay()
{
    // 10秒でループ
    if (animate_time > 10.0f)
    {
        animate_begin_time = sutil::currentTime();
    }

    animate_time = sutil::currentTime() - animate_begin_time;

    // NOTE: デバッグ用に開始時間を調整。提出時にはコメントアウトする
    // animate_time += 7;

    // FPSカメラ移動
    {
        float speed = 5;
        if (is_key_W_pressed) fpsCameraMove(make_float3(0, 0, -1), speed);
        if (is_key_A_pressed) fpsCameraMove(make_float3(-1, 0, 0), speed);
        if (is_key_S_pressed) fpsCameraMove(make_float3(0, 0, 1), speed);
        if (is_key_D_pressed) fpsCameraMove(make_float3(1, 0, 0), speed);
        if (is_key_Q_pressed) fpsCameraMove(make_float3(0, -1, 0), speed);
        if (is_key_E_pressed) fpsCameraMove(make_float3(0, 1, 0), speed);
    }

    updateFrame(animate_time);

    updateCamera();

    if (postprocessing_needs_init)
    {
        setupPostprocessing();
    }

    Variable(denoiserStage->queryVariable("blend"))->setFloat(denoiseBlend);

    bool isEarlyFrame = (frame_number <= numNonDenoisedFrames);

    if (flag_debug_render)
    {
        context->launch(1, width, height);
    }
    else
    {
        if (isEarlyFrame)
        {
            // NOTE: commandList を使わない場合
            // context->launch( 0, width, height );

            commandListWithoutDenoiser->execute();
        }
        else
        {
            commandListWithDenoiser->execute();
        }
    }

    switch (showBuffer)
    {
    case 1:
    {
        bufferInfo = "Original";
        sutil::displayBufferGL(use_post_tonemap ? getOutputBuffer() : getLinerBuffer());
        break;
    }
    case 2:
    {
        bufferInfo = "Tonemapped";
        // gamma correction already applied by tone mapper, avoid doing it twice
        sutil::displayBufferGL(use_post_tonemap ? getTonemappedBuffer() : getOutputBuffer(), BUFFER_PIXEL_FORMAT_DEFAULT, true);
        break;
    }
    case 3:
    {
        bufferInfo = "Albedo";
        sutil::displayBufferGL(getAlbedoBuffer());
        break;
    }
    case 4:
    {
        bufferInfo = "Normals";
        Buffer normalBuffer = getNormalBuffer();
        sutil::displayBufferGL(normalBuffer);
        break;
    }
    default:
        switch (denoiseMode)
        {
        case 0:
        {
            bufferInfo = "Denoised";
            break;
        }
        case 1:
        {
            bufferInfo = "Denoised (albedo)";
            break;
        }
        case 2:
        {
            bufferInfo = "Denoised (albedo+normals)";
            break;
        }
        }
        if (isEarlyFrame || flag_debug_render)
        {
            bufferInfo = "Tonemapped (early frame non-denoised)";
            // gamma correction already applied by tone mapper, avoid doing it twice
            if (use_post_tonemap)
            {
                sutil::displayBufferGL(getTonemappedBuffer(), BUFFER_PIXEL_FORMAT_DEFAULT, true);
            }
            else
            {
                sutil::displayBufferGL(getOutputBuffer(), BUFFER_PIXEL_FORMAT_DEFAULT, true);
            }
        }
        else
        {
            RTsize trainingSize = 0;
            trainingDataBuffer->getSize(trainingSize);
            if (useCustomTrainingData && trainingSize > 0)
            {
                if (useFirstTrainingDataPath)
                    bufferInfo += " Custom data";
                else
                    bufferInfo += " Custom data 2";
            }

            // gamma correction already applied by tone mapper, avoid doing it twice
            sutil::displayBufferGL(denoisedBuffer, BUFFER_PIXEL_FORMAT_DEFAULT, true);
        }
    }

    {
        sutil::displayText(bufferInfo.c_str(), 140, 10);
        char str[64];
        sprintf(str, "#%d", frame_number);
        sutil::displayText(str, (float)width - 60, (float)height - 20);
    }

    {
        static unsigned frame_count = 0;
        sutil::displayFps(frame_count++);
    }

    {
        static char animate_time_text[32];
        sprintf(animate_time_text, "animate_time:   %7.2f", animate_time);
        sutil::displayText(animate_time_text, 10, 80);
    }

    {
        static char camera_eye_text[32];
        sprintf(camera_eye_text, "camera_eye:    %7.2f, %7.2f, %7.2f", camera_eye.x, camera_eye.y, camera_eye.z);
        sutil::displayText(camera_eye_text, 10, 60);
    }

    {
        static char camera_lookat_text[32];
        sprintf(camera_lookat_text, "camera_lookat: %7.2f, %7.2f, %7.2f", camera_lookat.x, camera_lookat.y, camera_lookat.z);
        sutil::displayText(camera_lookat_text, 10, 40);
    }

    glutSwapBuffers();
}


void glutKeyboardPress(unsigned char k, int x, int y)
{

    switch (k)
    {
    case(27): // ESC
    {
        destroyContext();
        exit(0);
    }
    case('w'):
    {
        is_key_W_pressed = true;
        break;
    }
    case('a'):
    {
        is_key_A_pressed = true;
        break;
    }
    case('s'):
    {
        is_key_S_pressed = true;
        break;
    }
    case('d'):
    {
        is_key_D_pressed = true;
        break;
    }
    case('q'):
    {
        is_key_Q_pressed = true;
        break;
    }
    case('e'):
    {
        is_key_E_pressed = true;
        break;
    }
    case('p'):
    {
        Buffer buff;
        bool disableSrgbConversion = true;
        switch (showBuffer)
        {
        case 0:
        {
            buff = denoisedBuffer;
            break;
        }
        case 1:
        {
            disableSrgbConversion = false;
            buff = getOutputBuffer();
            break;
        }
        case 2:
        {
            buff = getTonemappedBuffer();
            break;
        }
        case 3:
        {
            disableSrgbConversion = false;
            buff = getAlbedoBuffer();
            break;
        }
        case 4:
        {
            disableSrgbConversion = false;
            buff = getNormalBuffer();
            break;
        }
        }

        const std::string outputImage = std::string(SAMPLE_NAME) + ".png";
        std::cerr << "Saving current frame to '" << outputImage << "'\n";
        sutil::displayBufferPNG(outputImage.c_str(), getOutputBuffer(), false);
        break;
    }
    case('b'):
    {
        showBuffer++;
        if (showBuffer > 5) showBuffer = 1;
        break;
    }
    case('r'):
    {
        flag_debug_render = !flag_debug_render;
        break;
    }
    case('t'):
    {
        animate_begin_time = sutil::currentTime();
        break;
    }
    case('m'):
    {
        ++denoiseMode;
        if (denoiseMode > 2) denoiseMode = 0;
        switch (denoiseMode)
        {
        case 0:
        {
            Variable albedoBuffer = denoiserStage->queryVariable("input_albedo_buffer");
            albedoBuffer->set(emptyBuffer);
            Variable normalBuffer = denoiserStage->queryVariable("input_normal_buffer");
            normalBuffer->set(emptyBuffer);
            break;
        }
        case 1:
        {
            Variable albedoBuffer = denoiserStage->queryVariable("input_albedo_buffer");
            albedoBuffer->set(getAlbedoBuffer());
            break;
        }
        case 2:
        {
            Variable normalBuffer = denoiserStage->queryVariable("input_normal_buffer");
            normalBuffer->set(getNormalBuffer());
            break;
        }
        }
        break;
    }
    case('0'):
    {
        denoiseBlend = 0.f;
        break;
    }
    case('1'):
    {
        denoiseBlend = 0.1f;
        break;
    }
    case('2'):
    {
        denoiseBlend = 0.2f;
        break;
    }
    case('3'):
    {
        denoiseBlend = 0.3f;
        break;
    }
    case('4'):
    {
        denoiseBlend = 0.4f;
        break;
    }
    case('5'):
    {
        denoiseBlend = 0.5f;
        break;
    }
    case('6'):
    {
        denoiseBlend = 0.6f;
        break;
    }
    case('7'):
    {
        denoiseBlend = 0.7f;
        break;
    }
    case('8'):
    {
        denoiseBlend = 0.8f;
        break;
    }
    case('9'):
    {
        denoiseBlend = 0.9f;
        break;
    }
    case('c'):
    {
        useCustomTrainingData = !useCustomTrainingData;
        Variable trainingBuff = denoiserStage->queryVariable("training_data_buffer");
        if (trainingBuff)
        {
            if (useCustomTrainingData)
                trainingBuff->setBuffer(trainingDataBuffer);
            else
                trainingBuff->setBuffer(emptyBuffer);
        }
        break;
    }
    case('z'):
    {
        useFirstTrainingDataPath = !useFirstTrainingDataPath;
        if (useFirstTrainingDataPath)
        {
            if (training_file.length() == 0)
                useFirstTrainingDataPath = false;
            else
                loadTrainingFile(training_file);
        }
        else
        {
            if (training_file_2.length() == 0)
                useFirstTrainingDataPath = true;
            else
                loadTrainingFile(training_file_2);
        }
    }
    }
}

void glutKeyboardPressUp(unsigned char k, int x, int y)
{

    switch (k)
    {
    case('w'):
    {
        is_key_W_pressed = false;
        break;
    }
    case('a'):
    {
        is_key_A_pressed = false;
        break;
    }
    case('s'):
    {
        is_key_S_pressed = false;
        break;
    }
    case('d'):
    {
        is_key_D_pressed = false;
        break;
    }
    case('q'):
    {
        is_key_Q_pressed = false;
        break;
    }
    case('e'):
    {
        is_key_E_pressed = false;
        break;
    }
    }
}


void glutMousePress(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouse_button = button;
        mouse_prev_pos = make_int2(x, y);
    }
    else
    {
        // nothing
    }
}


void glutMouseMotion(int x, int y)
{
    if (mouse_button == GLUT_RIGHT_BUTTON)
    {
        const float dx = static_cast<float>(x - mouse_prev_pos.x) /
            static_cast<float>(width);
        const float dy = static_cast<float>(y - mouse_prev_pos.y) /
            static_cast<float>(height);
        const float dmax = fabsf(dx) > fabs(dy) ? dx : dy;
        const float scale = std::min<float>(dmax, 0.9f);
        camera_eye = camera_eye + (camera_lookat - camera_eye) * scale;
        camera_changed = true;
    }
    else if (mouse_button == GLUT_LEFT_BUTTON)
    {
        const float2 from = { static_cast<float>(mouse_prev_pos.x),
                              static_cast<float>(mouse_prev_pos.y) };
        const float2 to = { static_cast<float>(x),
                              static_cast<float>(y) };

        const float2 a = { from.x / width, from.y / height };
        const float2 b = { to.x / width, to.y / height };

        camera_rotate = arcball.rotate(b, a);
        camera_changed = true;
    }
    else if (mouse_button == GLUT_MIDDLE_BUTTON)
    {
        const float dx = static_cast<float>(x - mouse_prev_pos.x) /
            static_cast<float>(width);
        const float dy = static_cast<float>(y - mouse_prev_pos.y) /
            static_cast<float>(height);
        float4 offset = { -dx, dy, 0, 0 };
        offset = frame * offset;
        float3 offset_v3 = { offset.x, offset.y, offset.z };
        offset_v3 *= 200;
        camera_eye += offset_v3;
        camera_lookat += offset_v3;
        camera_changed = true;
    }

    mouse_prev_pos = make_int2(x, y);
}


void glutResize(int w, int h)
{
    if (w == (int)width && h == (int)height) return;

    camera_changed = true;

    width = w;
    height = h;
    sutil::ensureMinimumSize(width, height);

    sutil::resizeBuffer(getOutputBuffer(), width, height);
    sutil::resizeBuffer(getLinerBuffer(), width, height);
    sutil::resizeBuffer(getTonemappedBuffer(), width, height);
    sutil::resizeBuffer(getAlbedoBuffer(), width, height);
    sutil::resizeBuffer(getNormalBuffer(), width, height);
    sutil::resizeBuffer(getLinerDepthBuffer(), width, height);
    sutil::resizeBuffer(denoisedBuffer, width, height);
    postprocessing_needs_init = true;

    glViewport(0, 0, width, height);

    glutPostRedisplay();
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

void printUsageAndExit(const std::string& argv0)
{
    std::cerr << "\nUsage: " << argv0 << " [options]\n";
    std::cerr <<
        "App Options:\n"
        "  -h | --help               Print this usage message and exit.\n"
        "  -f | --file               Save single frame to file and exit.\n"
        "  -n | --nopbo              Disable GL interop for display buffer.\n"
        "  -s | --sample             Sample number.\n"
        "  -t | --time_limit         Time limit(ssc).\n"
        "  --movie_time              Output Movie time length(ssc).\n"
        "  --fps                     Frame per Second.\n"
        "  --init_sample_per_launch  For movie mode\n"
        "App Keystrokes:\n"
        "  q  Quit\n"
        "  s  Save image to '" << SAMPLE_NAME << ".png'\n"
        << std::endl;

    exit(1);
}

void SavePNG(const unsigned char* Pix, const char* fname, int wid, int hgt, int chan)
{
    if (Pix == NULL || wid < 1 || hgt < 1)
        throw Exception("Image is ill-formed. Not saving");

    if (chan != 1 && chan != 3 && chan != 4)
        throw Exception("Attempting to save image with channel count != 1, 3, or 4.");

    int bpp = chan;
    int ret = stbi_write_png(fname, wid, hgt, bpp, Pix, wid * bpp);
    if (!ret)
        throw Exception("Failed to SavePNG");
}

std::thread displayBufferPNG(const char* filename, Buffer& buffer)
{
    double begin = sutil::currentTime();
    std::vector<unsigned char> pix(width * height * 3);

    sutil::getRawImageBuffer(filename, buffer, &pix[0], true);
    std::thread thd{ SavePNG, &pix[0], filename, width, height, 3 };
    thd.join();

    double end = sutil::currentTime();
    std::cout << "[info] save_png: " << filename << "\t" << (end - begin) << " sec." << std::endl;

    return thd;
}

std::thread displayBufferPNG_task(const char* filename, Buffer& buffer, unsigned char* pix)
{
    double begin = sutil::currentTime();

    sutil::getRawImageBuffer(filename, buffer, pix, true);
    std::thread thd{ SavePNG, pix, filename, width, height, 3 };

    double end = sutil::currentTime();
    std::cout << "[info] save_png: " << filename << "\t" << (end - begin) << " sec." << std::endl;

    return thd;
}

int main(int argc, char** argv)
{
    launch_time = sutil::currentTime();

    std::string out_file;
    int sampleMax = 20;
    double time_limit = 60 * 60;// 1 hour
    bool use_time_limit = false;

    // 動画の連番画像用
    float movie_time_start = -1.0f;
    float movie_time_end = -1.0f;
    float movie_fps = -1.0f;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);

        if (arg == "-h" || arg == "--help")
        {
            printUsageAndExit(argv[0]);
        }
        else if (arg == "-f" || arg == "--file")
        {
            if (i == argc - 1)
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsageAndExit(argv[0]);
            }
            out_file = argv[++i];
            use_pbo = false;
        }
        else if (arg == "-n" || arg == "--nopbo")
        {
            use_pbo = false;
        }
        else if (arg == "-s" || arg == "--sample")
        {
            if (i == argc - 1)
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsageAndExit(argv[0]);
            }
            sampleMax = atoi(argv[++i]);
        }
        else if (arg == "-t" || arg == "--time_limit")
        {
            if (i == argc - 1)
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsageAndExit(argv[0]);
            }
            time_limit = atof(argv[++i]);
            use_time_limit = true;
        }
        else if (arg == "-W" || arg == "--width")
        {
            if (i == argc - 1)
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsageAndExit(argv[0]);
            }
            width = atoi(argv[++i]);
        }
        else if (arg == "-H" || arg == "--height")
        {
            if (i == argc - 1)
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsageAndExit(argv[0]);
            }
            height = atoi(argv[++i]);
        }
        else if (arg == "-S" || arg == "--sample_per_launch")
        {
            if (i == argc - 1)
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsageAndExit(argv[0]);
            }
            sample_per_launch = atoi(argv[++i]);
        }
        else if (arg == "-A" || arg == "--auto_sample_per_launch")
        {
            if (i == argc - 1)
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsageAndExit(argv[0]);
            }
            auto_set_sample_per_launch = true;
            auto_set_sample_per_launch_scale = atof(argv[++i]);
        }
        else if (arg == "--last_frame_scale")
        {
            if (i == argc - 1)
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsageAndExit(argv[0]);
            }
            last_frame_scale = atof(argv[++i]);
        }
        else if (arg == "--tonemap_exposure")
        {
            if (i == argc - 1)
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsageAndExit(argv[0]);
            }
            tonemap_exposure = atof(argv[++i]);
        }
        else if (arg.find("-d") == 0 || arg.find("--dim") == 0)
        {
            size_t index = arg.find_first_of('=');
            if (index == std::string::npos)
            {
                std::cerr << "Option '" << arg << " is malformed. Please use the syntax -d | --dim=<width>x<height>.\n";
                printUsageAndExit(argv[0]);
            }
            std::string dim = arg.substr(index + 1);
            try
            {
                sutil::parseDimensions(dim.c_str(), width, height);
            }
            catch (Exception e)
            {
                std::cerr << "Option '" << arg << " is malformed. Please use the syntax -d | --dim=<width>x<height>.\n";
                printUsageAndExit(argv[0]);
            }
        }
        else if (arg == "--blend")
        {
            if (i == argc - 1)
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsageAndExit(argv[0]);
            }
            int denoiseBlendPercent = atoi(argv[++i]);
            if (denoiseBlendPercent < 0) denoiseBlendPercent = 0;
            if (denoiseBlendPercent > 100) denoiseBlendPercent = 100;
            denoiseBlend = denoiseBlendPercent / 100.f;
        }
        else if (arg == "--denoise_mode")
        {
            if (i == argc - 1)
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsageAndExit(argv[0]);
            }
            denoiseMode = atoi(argv[++i]);
            if (denoiseMode < 0 || denoiseMode > 2)
            {
                std::cerr << "Option '" << arg << "' must be 0, 1, or 2.\n";
                printUsageAndExit(argv[0]);
            }
        }
        else if (arg == "--perf")
        {
            if (i == argc - 1)
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsageAndExit(argv[0]);
            }
            denoiser_perf_mode = true;
            denoiser_perf_iter = atoi(argv[++i]);
        }
        else if (arg == "--training_file")
        {
            if (i == argc - 1)
            {
                std::cerr << "Option '" << argv[i] << "' requires additional argument.\n";
                printUsageAndExit(argv[0]);
            }
            training_file = argv[++i];
        }
        else if (arg == "--training_file2")
        {
            if (i == argc - 1)
            {
                std::cerr << "Option '" << argv[i] << "' requires additional argument.\n";
                printUsageAndExit(argv[0]);
            }
            training_file_2 = argv[++i];
        }
        else if (arg == "--debug")
        {
            flag_debug = true;
        }
        else if (arg == "--movie_time_range")
        {
            if (i == argc - 2)
            {
                std::cerr << "Option '" << argv[i] << "' requires additional 2 arguments.\n";
                printUsageAndExit(argv[0]);
            }
            movie_time_start = atof(argv[++i]);
            movie_time_end = atof(argv[++i]);
        }
        else if (arg == "--movie_fps")
        {
            if (i == argc - 1)
            {
                std::cerr << "Option '" << argv[i] << "' requires additional argument.\n";
                printUsageAndExit(argv[0]);
            }
            movie_fps = atof(argv[++i]);
        }
        else if (arg == "--init_sample_per_launch")
        {
            if (i == argc - 1)
            {
                std::cerr << "Option '" << argv[i] << "' requires additional argument.\n";
                printUsageAndExit(argv[0]);
            }
            init_sample_per_launch = atoi(argv[++i]);
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit(argv[0]);
        }
    }

    try
    {
        if (use_pbo && out_file.empty()) {
            glutInitialize(&argc, argv);

#ifndef __APPLE__
            glewInit();
#endif
        }

        createContext();

        if (training_file.length() == 0 && training_file_2.length() != 0)
            useFirstTrainingDataPath = false;

        if (useFirstTrainingDataPath)
            loadTrainingFile(training_file);
        else
            loadTrainingFile(training_file_2);

        setupCamera();
        setupScene();

        context->validate();

        // 動画の画像連番書き出しモード
        if (movie_fps > 0.0f && movie_time_start >= 0.0f && movie_time_end >= 0.0f)
        {
            std::ofstream render_time_tsv("render_time.tsv");
            render_time_tsv << "frame\tframe_time\trender_time" << std::endl;

            setupPostprocessing();
            Variable(denoiserStage->queryVariable("blend"))->setFloat(denoiseBlend);
            int all_frame_total_sample = 0;

            if (denoiseMode > 0)
            {
                Variable albedoBuffer = denoiserStage->queryVariable("input_albedo_buffer");
                albedoBuffer->set(getAlbedoBuffer());
            }

            if (denoiseMode > 1)
            {
                Variable normalBuffer = denoiserStage->queryVariable("input_normal_buffer");
                normalBuffer->set(getNormalBuffer());
            }

            int frame_start = movie_time_start * movie_fps;
            int frame_count = movie_fps * (movie_time_end - movie_time_start);

            // print config
            std::cout << "[info] time_range: " << "start: " << movie_time_start << "\tend: " << movie_time_end << std::endl;
            std::cout << "[info] frame_count: " << frame_count << " (fps: " << movie_fps << " x time length: " << (movie_time_end - movie_time_start) << " sec.)" << std::endl;
            std::cout << "[info] resolution: " << width << "x" << height << " px" << std::endl;
            std::cout << "[info] time_limit: " << time_limit << " sec." << std::endl;
            std::cout << "[info] init_sample_per_launch: " << init_sample_per_launch << std::endl;
            std::cout << "[info] tonemap_exposure: " << tonemap_exposure << std::endl;

            // 画像の非同期保存のためのスレッド
            std::vector<std::thread> threads;

            // 画像の非同期保存のためのバッファ
            std::vector<unsigned char> pix(width * height * 3);

            for (int frame = frame_start; frame < frame_start + frame_count; ++frame)
            {
                float movie_time = frame / movie_fps;

                double frame_start_time = sutil::currentTime();
                double last_time = frame_start_time;
                bool finalFrame = false;
                total_sample = 0;

                // 1回目のサンプリング数を初期値にリセット
                sample_per_launch = init_sample_per_launch;

                double global_remain_time = time_limit - (frame_start_time - launch_time);
                int rest_frame = frame_count + frame_start - frame;
                double frame_time_limit = global_remain_time / rest_frame;

                updateFrame(movie_time);
                updateCamera();

                std::cout << "[info] frame: " << frame << "\tmovie_time:" << movie_time << "\tframe_time_limit:" << frame_time_limit << std::endl;

                // NOTE: 実際には2回ループの場合しかない
                for (int i = 0; !finalFrame; ++i)
                {
                    double now = sutil::currentTime();
                    double used_time = now - frame_start_time;
                    double delta_time = now - last_time;
                    double remain_time = frame_time_limit - used_time;
                    last_time = now;

                    std::cout << "[info] frame_global: " << frame << "\tframe_local:" << (frame - frame_start) << " / " << frame_count << "(" << ((double)(frame + 1 - frame_start) / frame_count * 100) << "%)\tlaunch : " << i
                        << "\tused_time:" << (now - launch_time) << "/" << time_limit << "(" << ((double)(now - launch_time) / time_limit * 100) << "%)" << std::endl;

                    // 1回目の結果から、時間切れしない sample_per_launch を決定する
                    if (i == 1)
                    {
                        // 初回launchは時間がかかるため、テストの描画時間を40%に補正する
                        if (frame == frame_start)
                        {
                            delta_time *= 0.4;
                        }

                        int new_sample_per_launch = (int)(remain_time / delta_time * sample_per_launch);

                        // 1以上にしないと真っ暗な結果になる
                        new_sample_per_launch = max(1, new_sample_per_launch);


                        std::cout << "[info] chnage sample_per_launch: " << sample_per_launch << " to " << new_sample_per_launch << std::endl;
                        sample_per_launch = new_sample_per_launch;
                        finalFrame = true;
                    }

                    context["sample_per_launch"]->setUint(sample_per_launch);
                    context["frame_number"]->setUint(frame_number);
                    context["total_sample"]->setUint(total_sample);

                    if (finalFrame)
                    {
                        {
                            double begin_time = sutil::currentTime();

                            commandListWithDenoiser->execute();

                            double end_time = sutil::currentTime();
                            std::cout << "[info] final_frame\trender_time:" << end_time - begin_time << "\tsample_per_launch: " << sample_per_launch << std::endl;
                        }

                        double thread_join_begin = sutil::currentTime();

                        for (std::thread& th : threads) {
                            th.join();
                        }

                        double thread_join_end = sutil::currentTime();
                        std::cout << "[info] thread_join_time: " << thread_join_end - thread_join_begin << " sec." << std::endl;

                        threads.clear();

                        char filename[50];
                        snprintf(filename, sizeof(filename), "%03d.png", frame + 1);

                        // displayBufferPNG(filename, denoisedBuffer);
                        threads.push_back(displayBufferPNG_task(filename, denoisedBuffer, &pix[0]));

                        if (flag_debug)
                        {
                            snprintf(filename, sizeof(filename), "%03d_original.png", frame + 1);
                            displayBufferPNG(filename, getOutputBuffer());

                            snprintf(filename, sizeof(filename), "%03d_albedo.png", frame + 1);
                            displayBufferPNG(filename, getAlbedoBuffer());

                            snprintf(filename, sizeof(filename), "%03d_normal.png", frame + 1);
                            displayBufferPNG(filename, getNormalBuffer());

                            snprintf(filename, sizeof(filename), "%03d_liner.png", frame + 1);
                            displayBufferPNG(filename, getLinerBuffer());

                            snprintf(filename, sizeof(filename), "%03d_depth.png", frame + 1);
                            displayBufferPNG(filename, getLinerDepthBuffer());
                        }

                        total_sample += sample_per_launch;
                        all_frame_total_sample += total_sample;
                        std::cout << "[info] total_sample: " << total_sample << "\tall_frame_total_sample: " << all_frame_total_sample << std::endl;
                    }
                    else
                    {
                        double begin_time = sutil::currentTime();

                        commandListWithoutDenoiser->execute();

                        double end_time = sutil::currentTime();
                        std::cout << "[info] test_frame\trender_time:" << end_time - begin_time << "\tsample_per_launch: " << sample_per_launch << std::endl;
                        render_time_tsv << frame << "\t" << movie_time << "\t" << end_time - begin_time << std::endl;

                        frame_number++;
                        total_sample += sample_per_launch;
                    }
                }
            }

            destroyContext();

            for (std::thread& th : threads) {
                th.join();
            }

            threads.clear();

            double finish_time = sutil::currentTime();
            double total_time = finish_time - launch_time;
            std::cout << "[info] Finish!\ttotal_time: " << total_time << " sec.\tall_frame_total_sample: " << all_frame_total_sample << std::endl;
        }
        // インタラクティブモード
        else if (out_file.empty())
        {
            animate_begin_time = sutil::currentTime();
            context["sample_per_launch"]->setUint(10);
            glutRun();
        }
        // 静止画モード
        else
        {
            setupPostprocessing();
            updateCamera();
            Variable(denoiserStage->queryVariable("blend"))->setFloat(denoiseBlend);

            if (denoiseMode > 0)
            {
                Variable albedoBuffer = denoiserStage->queryVariable("input_albedo_buffer");
                albedoBuffer->set(getAlbedoBuffer());
            }

            if (denoiseMode > 1)
            {
                Variable normalBuffer = denoiserStage->queryVariable("input_normal_buffer");
                normalBuffer->set(getNormalBuffer());
            }

            // print config
            std::cout << "[info] resolution: " << width << "x" << height << " px" << std::endl;
            std::cout << "[info] time_limit: " << time_limit << " sec." << std::endl;
            std::cout << "[info] sample_per_launch: " << sample_per_launch << std::endl;
            std::cout << "[info] auto_set_sample_per_launch: " << auto_set_sample_per_launch << std::endl;
            std::cout << "[info] auto_set_sample_per_launch_scale: " << auto_set_sample_per_launch_scale << std::endl;
            std::cout << "[info] last_frame_scale: " << last_frame_scale << std::endl;
            std::cout << "[info] tonemap_exposure: " << tonemap_exposure << std::endl;


            if (use_time_limit)
            {
                std::cout << "[info] sample: INF(" << sampleMax << ")" << std::endl;
            }
            else
            {
                std::cout << "[info] sample: " << sampleMax << std::endl;
            }

            double last_time = sutil::currentTime();

            bool finalFrame = false;

            // NOTE: time_limit が指定されていたら、サンプル数は無制限にする
            for (int i = 0; !finalFrame && (total_sample < sampleMax || use_time_limit); ++i)
            {
                // TODO: 動作確認
                finalFrame |= (!use_time_limit && total_sample == sampleMax - 1);

                double now = sutil::currentTime();
                double used_time = now - launch_time;
                double delta_time = now - last_time;
                double remain_time = time_limit - used_time;
                last_time = now;

                std::cout << "loop:" << i << "\tsample_per_launch\t:" << sample_per_launch << "\tdelta_time:" << delta_time << "\tdelta_time_per_sample:" << delta_time / sample_per_launch << "\tused_time:" << used_time << "\tremain_time:" << remain_time << "\tsample:" << total_sample << "\tframe_number:" << frame_number << std::endl;

                if (auto_set_sample_per_launch && i == 1)
                {
                    sample_per_launch = (int)(remain_time / delta_time * auto_set_sample_per_launch_scale * sample_per_launch);
                    std::cout << "[info] chnage sample_per_launch: " << sample_per_launch << " to " << sample_per_launch << std::endl;
                }

                // NOTE: 前フレームの所要時間から次のフレームが制限時間内に終るかを予測する。デノイズを考慮して last_frame_scale 倍に見積もる
                if (used_time + delta_time * last_frame_scale > time_limit)
                {
                    if (sample_per_launch == 1)
                    {
                        std::cout << "[info] reached time limit! used_time: " << used_time << " sec. remain_time: " << remain_time << " sec." << std::endl;
                        finalFrame = true;
                    }
                    else
                    {
                        std::cout << "[info] chnage sample_per_launch: " << sample_per_launch << " to 1" << std::endl;
                        sample_per_launch = 1;
                    }
                }

                context["sample_per_launch"]->setUint(sample_per_launch);
                context["frame_number"]->setUint(frame_number);
                context["total_sample"]->setUint(total_sample);

                if (finalFrame)
                {
                    if (denoiser_perf_mode)
                    {
                        for (int i = 0; i < denoiser_perf_iter; i++)
                        {
                            commandListWithDenoiser->execute();
                        }
                    }
                    else
                    {
                        commandListWithDenoiser->execute();
                    }
                }
                else
                {
                    commandListWithoutDenoiser->execute();
                }

                frame_number++;
                total_sample += sample_per_launch;
            }

            {
                double now = sutil::currentTime();
                std::cout << "[info] final_frame_rendering: " << (now - last_time) << " sec." << std::endl;
            }


            displayBufferPNG(out_file.c_str(), denoisedBuffer);

            if (flag_debug)
            {
                displayBufferPNG((out_file + "_original.png").c_str(), getOutputBuffer());
                displayBufferPNG((out_file + "_albedo.png").c_str(), getAlbedoBuffer());
                displayBufferPNG((out_file + "_normal.png").c_str(), getNormalBuffer());
                displayBufferPNG((out_file + "_depth.png").c_str(), getLinerDepthBuffer());
                displayBufferPNG((out_file + "_liner.png").c_str(), getLinerBuffer());
            }

            destroyContext();

            double finish_time = sutil::currentTime();
            double total_time = finish_time - launch_time;
            std::cout << "[info] total_time: " << total_time << " sec." << std::endl;
            std::cout << "[info] total_sample: " << total_sample << std::endl;
        }

        return 0;
    }
    SUTIL_CATCH(context->get())
}