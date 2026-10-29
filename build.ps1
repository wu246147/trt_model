# 构建统一 TensorRT DLL 和 Qt 界面；只调用本机编译器，不安装或修改系统依赖。
param(
    [string]$TensorRT = 'E:\env\dll\tensorRT\TensorRT-8.6.1.6',
    [string]$CUDA = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8',
    [string]$OpenCV = 'D:\opencv4.5.5\opencv\build\x64\vc15\lib',
    [string]$Qt = 'D:\Qt\5.15.2\msvc2019_64',
    [switch]$NoGui,
    [switch]$NoTests,
    [switch]$DeployRuntime
)
$ErrorActionPreference = 'Stop'
$taskVsWhere = Join-Path ([Environment]::GetEnvironmentVariable('ProgramFiles(x86)')) 'Microsoft Visual Studio\Installer\vswhere.exe'
$taskVs = & $taskVsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (!$taskVs) { throw '没有找到 Visual Studio x64 C++ 工具链。' }
$taskVcVars = Join-Path $taskVs 'VC\Auxiliary\Build\vcvars64.bat'
$taskEnvironment = & $env:COMSPEC /d /c ('call "{0}" >nul && set' -f $taskVcVars)
if ($LASTEXITCODE) { throw '初始化 MSVC 环境失败。' }
foreach ($taskLine in $taskEnvironment) {
    if ($taskLine -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process') }
}
$taskCmake = (Get-Command cmake.exe -ErrorAction SilentlyContinue).Source
if (!$taskCmake) { $taskCmake = Join-Path $taskVs 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' }
$taskNinja = Join-Path $taskVs 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
$taskBuild = Join-Path $PSScriptRoot 'build-release'
$taskDist = Join-Path $PSScriptRoot 'dist'
$taskGui = if ($NoGui) { 'OFF' } else { 'ON' }
$taskTests = if ($NoTests) { 'OFF' } else { 'ON' }
& $taskCmake -S $PSScriptRoot -B $taskBuild -G Ninja "-DCMAKE_MAKE_PROGRAM=$taskNinja" -DCMAKE_BUILD_TYPE=Release "-DTRT_DIR=$TensorRT" "-DCUDAToolkit_ROOT=$CUDA" "-DOpenCV_DIR=$OpenCV" "-DTRT_MODEL_QT_ROOT=$Qt" "-DTRT_MODEL_BUILD_GUI=$taskGui" "-DTRT_MODEL_BUILD_TESTS=$taskTests"
if ($LASTEXITCODE) { throw 'CMake 配置失败。' }
& $taskCmake --build $taskBuild --config Release --parallel 4
if ($LASTEXITCODE) { throw '编译失败。' }
& $taskCmake --install $taskBuild --config Release --prefix $taskDist
if ($LASTEXITCODE) { throw '安装发布文件失败。' }
if (!$NoGui) {
    foreach ($taskDestination in @($taskBuild, (Join-Path $taskDist 'bin'))) {
        & (Join-Path $Qt 'bin\windeployqt.exe') --release --no-translations --no-compiler-runtime (Join-Path $taskDestination 'defect_alg_trt.exe')
        if ($LASTEXITCODE) { throw 'Qt 运行库部署失败。' }
    }
}
# 第三方原生库体积较大，只有显式要求时复制；正常构建不重复复制这些文件。
if ($DeployRuntime) {
    $taskRuntimeFiles = @('nvinfer.dll','nvonnxparser.dll','nvinfer_plugin.dll','nvinfer_builder_resource.dll') | ForEach-Object { Join-Path $TensorRT "lib\$_" }
    $taskRuntimeFiles += @('cudart64_110.dll','cublas64_11.dll','cublasLt64_11.dll') | ForEach-Object { Join-Path $CUDA "bin\$_" }
    $taskRuntimeFiles += Join-Path (Split-Path -Parent $OpenCV) 'bin\opencv_world455.dll'
    foreach ($taskFile in $taskRuntimeFiles) {
        if (!(Test-Path -LiteralPath $taskFile)) { throw "缺少运行库: $taskFile" }
        foreach ($taskDestination in @($taskBuild, (Join-Path $taskDist 'bin'))) { Copy-Item -LiteralPath $taskFile -Destination $taskDestination -Force }
    }
}
Write-Host "构建完成: $taskDist"
