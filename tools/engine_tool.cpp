#include "trt_engine.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#ifdef _WIN32
#include <Windows.h>
#endif

/// 开发验证工具：在进程内转换 ONNX 或显示引擎信息，不调用外部 trtexec。
int runTool(int argc, char** argv) {
    try {
        if (argc >= 4 && std::string(argv[1]) == "build") {
            trt_model::buildEngine(argv[2], argv[3], argc > 4 && std::string(argv[4]) == "fp16");
            std::cout << "引擎已保存: " << argv[3] << '\n';
            return 0;
        }
        if (argc >= 3 && (std::string(argv[1]) == "inspect" || std::string(argv[1]) == "infer")) {
            trt_model::Engine engine;
            engine.load(argv[2]);
            const auto& info = engine.info();
            std::cout << "input " << info.inputName << " [" << info.batch << "," << info.channels << "," << info.height << "," << info.width << "] classes=" << info.classes << '\n';
            for (const auto& output : info.outputs) {
                std::cout << "output " << output.name;
                for (int d : output.shape) std::cout << " " << d;
                std::cout << '\n';
            }
            for (const auto& item : info.metadata) std::cout << "meta " << item.first << "=" << item.second << '\n';
            if (std::string(argv[1]) == "infer") {
                if (argc != 5) throw std::runtime_error("infer 需要输入 float32 二进制文件和输出文件前缀");
                std::vector<float> input(static_cast<size_t>(info.batch) * info.channels * info.height * info.width);
                std::ifstream file(std::filesystem::u8path(argv[3]), std::ios::binary);
                if (!file.read(reinterpret_cast<char*>(input.data()), input.size() * sizeof(float))) throw std::runtime_error("读取 NCHW 数据失败");
                const auto outputs = engine.infer(input);
                for (size_t i = 0; i < outputs.size(); ++i) {
                    const auto& output = outputs[i];
                    std::ofstream binary(std::filesystem::u8path(std::string(argv[4]) + "." + std::to_string(i) + ".bin"), std::ios::binary);
                    if (!binary.write(reinterpret_cast<const char*>(output.values.data()), output.values.size() * sizeof(float))) throw std::runtime_error("写输出数据失败");
                }
            }
            return 0;
        }
        std::cerr << "用法: trt_engine_tool build model.onnx model.engine [fp16]\n"
                  << "      trt_engine_tool inspect model.engine\n"
                  << "      trt_engine_tool infer model.engine input.bin output_prefix\n";
        return 2;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}

#ifdef _WIN32
/// Windows 命令行按 UTF-16 接收路径，再统一转换为 DLL 约定的 UTF-8，支持中文文件名。
int wmain(int argc, wchar_t** wideArguments) {
    try {
        std::vector<std::string> utf8;
        utf8.reserve(argc);
        for (int i = 0; i < argc; ++i) {
            const int length = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wideArguments[i], -1, nullptr, 0, nullptr, nullptr);
            if (length <= 0) throw std::runtime_error("命令行路径转换为 UTF-8 失败");
            std::string value(static_cast<size_t>(length), '\0');
            if (!WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wideArguments[i], -1, value.data(), length, nullptr, nullptr))
                throw std::runtime_error("命令行路径编码无效");
            value.pop_back(); utf8.push_back(std::move(value));
        }
        std::vector<char*> arguments;
        for (auto& value : utf8) arguments.push_back(value.data());
        return runTool(argc, arguments.data());
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
#else
/// 其他平台直接使用 UTF-8 命令行参数运行开发验证工具。
int main(int argc, char** argv) { return runTool(argc, argv); }
#endif
