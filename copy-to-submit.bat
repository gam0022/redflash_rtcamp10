set srcDir=D:\Dropbox\RTCamp\redflash_rtcamp10\build\
set srcCudaDir=D:\Dropbox\RTCamp\redflash_rtcamp10\redflash\
set dstDir=D:\Dropbox\RTCamp\gam0022_gpu\

cp %srcDir%bin\Release\run.ps1 %dstDir%
cp %srcDir%bin\Release\redflash.exe %dstDir%
cp %srcDir%bin\Release\sutil_sdk.dll %dstDir%
cp %srcCudaDir%redflash.h %dstDir%cuda\
cp %srcCudaDir%redflash.cu %dstDir%cuda\
cp %srcCudaDir%bsdf_diffuse.cu %dstDir%cuda\
cp %srcCudaDir%bsdf_disney.cu %dstDir%cuda\
cp %srcCudaDir%bsdf_portal.cu %dstDir%cuda\
cp %srcCudaDir%intersect_raymarching.cu %dstDir%cuda\
cp %srcCudaDir%intersect_sphere.cu %dstDir%cuda\
cp %srcCudaDir%random.h %dstDir%cuda\