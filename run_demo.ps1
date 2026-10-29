# 使用本机依赖目录启动测试界面，避免将大型 TensorRT 运行库重复复制到开发输出目录。
param(
    [string]$TensorRT = 'E:\env\dll\tensorRT\TensorRT-8.6.1.6',
    [string]$CUDA = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8',
    [string]$OpenCVBin = 'D:\opencv4.5.5\opencv\build\x64\vc15\bin'
)
$ErrorActionPreference = 'Stop'
$taskExecutable = Join-Path $PSScriptRoot 'bin\defect_alg_trt.exe'
if (!(Test-Path -LiteralPath $taskExecutable)) { $taskExecutable = Join-Path $PSScriptRoot 'dist\bin\defect_alg_trt.exe' }
if (!(Test-Path -LiteralPath $taskExecutable)) { throw '请先运行 build.ps1 构建测试界面。' }
$env:PATH = (Join-Path $TensorRT 'lib') + ';' + (Join-Path $CUDA 'bin') + ';' + $OpenCVBin + ';' + $env:PATH
# 用户运行此脚本的目的就是打开交互式窗口，因此使用正常窗口显示方式。
Start-Process -FilePath $taskExecutable -WorkingDirectory $PSScriptRoot -WindowStyle Normal
