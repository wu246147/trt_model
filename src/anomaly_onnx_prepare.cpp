#include "anomaly_onnx_prepare.h"
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace trt_model {
namespace {
/// 保存 protobuf 字段的原始编码和有效载荷视图；不复制模型权重。
struct Field { unsigned number = 0, wire = 0; std::string_view raw, value; std::uint64_t integer = 0; };

/// 读取变长整数并检查截断及溢出，offset 为输入输出游标。
std::uint64_t readVarint(std::string_view bytes, size_t& offset) {
    std::uint64_t value = 0;
    for (unsigned shift = 0; shift < 64 && offset < bytes.size(); shift += 7) {
        const auto byte = static_cast<unsigned char>(bytes[offset++]);
        if (shift == 63 && byte > 1) throw std::runtime_error("异常 ONNX 变长整数溢出");
        value |= static_cast<std::uint64_t>(byte & 127) << shift;
        if (!(byte & 128)) return value;
    }
    throw std::runtime_error("异常 ONNX protobuf 数据被截断");
}

/// 枚举一层 protobuf 字段，拒绝越界或未支持的编码类型。
std::vector<Field> fields(std::string_view bytes) {
    std::vector<Field> result;
    size_t offset = 0;
    while (offset < bytes.size()) {
        const size_t start = offset;
        const auto tag = readVarint(bytes, offset);
        Field f;
        f.number = static_cast<unsigned>(tag >> 3); f.wire = static_cast<unsigned>(tag & 7);
        if (f.number == 0) throw std::runtime_error("异常 ONNX 存在无效字段编号");
        if (f.wire == 0) f.integer = readVarint(bytes, offset);
        else {
            const auto count = f.wire == 2 ? readVarint(bytes, offset) : f.wire == 1 ? 8 : f.wire == 5 ? 4 : std::uint64_t(-1);
            if (count > bytes.size() - offset) throw std::runtime_error("异常 ONNX 字段长度无效");
            f.value = bytes.substr(offset, static_cast<size_t>(count)); offset += static_cast<size_t>(count);
        }
        f.raw = bytes.substr(start, offset - start); result.push_back(f);
    }
    return result;
}

/// 编码无符号变长整数，负 int64 由调用者按补码转换。
void appendVarint(std::string& output, std::uint64_t value) {
    while (value >= 128) { output += static_cast<char>((value & 127) | 128); value >>= 7; }
    output += static_cast<char>(value);
}

/// 追加长度编码字段，value 可引用较大的模型权重。
void appendBytes(std::string& output, unsigned number, std::string_view value) {
    appendVarint(output, (number << 3) | 2); appendVarint(output, value.size()); output.append(value.data(), value.size());
}

/// 追加整数 protobuf 字段，用于 ONNX 维度、属性和版本。
void appendInteger(std::string& output, unsigned number, std::uint64_t value) {
    appendVarint(output, number << 3); appendVarint(output, value);
}

/// 返回消息中指定字符串字段；不存在则返回空字符串。
std::string textField(std::string_view message, unsigned number) {
    for (const auto& f : fields(message)) if (f.number == number && f.wire == 2) return std::string(f.value);
    return {};
}

/// 解析 INT64 TensorProto 常量，归约轴只能来自静态常量以保证等价转换。
std::vector<std::int64_t> tensorIntegers(std::string_view tensor) {
    std::vector<std::int64_t> values;
    const auto tensorFields = fields(tensor);
    bool int64 = false;
    for (const auto& f : tensorFields) if (f.number == 2) int64 = f.integer == 7;
    // 权重通常为数百MB的浮点张量，仅解码真正的INT64轴常量，避免复制网络参数。
    if (!int64) return {};
    for (const auto& f : tensorFields) {
        if (f.number == 7 && f.wire == 0) values.push_back(static_cast<std::int64_t>(f.integer));
        if (f.number == 7 && f.wire == 2) { size_t cursor = 0; while (cursor < f.value.size()) values.push_back(static_cast<std::int64_t>(readVarint(f.value, cursor))); }
        if (f.number == 9) {
            if (f.value.size() % 8) throw std::runtime_error("异常 ONNX 归约轴常量长度错误");
            for (size_t i = 0; i < f.value.size(); i += 8) {
                std::uint64_t value = 0;
                for (unsigned j = 0; j < 8; ++j) value |= static_cast<std::uint64_t>(static_cast<unsigned char>(f.value[i + j])) << (8 * j);
                values.push_back(static_cast<std::int64_t>(value));
            }
        }
    }
    return values;
}

/// 根据元数据读取正整数尺寸，异常时保留具体键名。
int dimension(const std::map<std::string, std::string>& metadata, const std::string& key) {
    const auto found = metadata.find(key);
    if (found == metadata.end()) throw std::runtime_error("异常 ONNX 缺少尺寸元数据: " + key);
    size_t end = 0; const auto value = std::stoll(found->second, &end);
    if (end != found->second.size() || value <= 0 || value > std::numeric_limits<int>::max()) throw std::runtime_error("异常 ONNX 尺寸无效: " + key);
    return static_cast<int>(value);
}

/// 生成固定 FP32 NCHW 单批次输入，图内仍包含原有归一化与校准运算。
std::string makeInput(const std::string& name, int height, int width) {
    std::string shape;
    for (int value : {1, 3, height, width}) { std::string dim; appendInteger(dim, 1, value); appendBytes(shape, 1, dim); }
    std::string tensor; appendInteger(tensor, 1, 1); appendBytes(tensor, 2, shape);
    std::string type; appendBytes(type, 1, tensor);
    std::string input; appendBytes(input, 1, name); appendBytes(input, 2, type); return input;
}

/// 生成单元素INT64 Constant节点，供Shape属性展开后的Slice使用。
std::string integerConstant(const std::string& name, std::int64_t value) {
    std::string tensor; appendInteger(tensor, 1, 1); appendInteger(tensor, 2, 7); appendInteger(tensor, 7, static_cast<std::uint64_t>(value));
    std::string attribute; appendBytes(attribute, 1, "value"); appendBytes(attribute, 5, tensor); appendInteger(attribute, 20, 4);
    std::string node; appendBytes(node, 2, name); appendBytes(node, 3, name); appendBytes(node, 4, "Constant"); appendBytes(node, 5, attribute); return node;
}

/// 将TRT8.6忽略的Shape(start,end)改为无属性Shape再显式Slice，保持负轴和边界截断语义。
void appendShapeNodes(std::string& graph, std::string_view node) {
    const auto input = textField(node, 1), output = textField(node, 2), name = textField(node, 3);
    std::int64_t start = 0, end = std::numeric_limits<int>::max(); bool hasBounds = false;
    for (const auto& f : fields(node)) if (f.number == 5) {
        const auto key = textField(f.value, 1);
        if (key != "start" && key != "end") continue;
        hasBounds = true;
        for (const auto& a : fields(f.value)) if (a.number == 3) { if (key == "start") start = static_cast<std::int64_t>(a.integer); else end = static_cast<std::int64_t>(a.integer); }
    }
    if (!hasBounds) { appendBytes(graph, 1, node); return; }
    const auto base = output + "_trt86_shape";
    std::string full; appendBytes(full, 1, input); appendBytes(full, 2, base); appendBytes(full, 3, name + "_full"); appendBytes(full, 4, "Shape"); appendBytes(graph, 1, full);
    appendBytes(graph, 1, integerConstant(base + "_start", start));
    appendBytes(graph, 1, integerConstant(base + "_end", end));
    appendBytes(graph, 1, integerConstant(base + "_axis", 0));
    std::string slice;
    for (const auto& tensor : {base, base + "_start", base + "_end", base + "_axis"}) appendBytes(slice, 1, tensor);
    appendBytes(slice, 2, output); appendBytes(slice, 3, name + "_slice"); appendBytes(slice, 4, "Slice"); appendBytes(graph, 1, slice);
}

/// 读取节点的整数属性；未设置时返回ONNX协议规定的默认值。
std::int64_t integerAttribute(std::string_view node, const std::string& name, std::int64_t defaultValue) {
    for (const auto& f : fields(node)) if (f.number == 5 && textField(f.value, 1) == name)
        for (const auto& a : fields(f.value)) if (a.number == 3) return static_cast<std::int64_t>(a.integer);
    return defaultValue;
}

/// 读取节点所有输入或输出名称，number为NodeProto字段编号1或2。
std::vector<std::string> tensorNames(std::string_view node, unsigned number) {
    std::vector<std::string> names;
    for (const auto& f : fields(node)) if (f.number == number) names.emplace_back(f.value);
    return names;
}

/// 将“完整排序后只用前K个值”收缩为同序TopK，严格验证消费者以绕过TRT8.6的K<=3840限制。
std::int64_t limitedTopK(std::string_view node, const std::vector<Field>& graphFields,
    const std::map<std::string, std::vector<std::int64_t>>& constants) {
    const auto inputs = tensorNames(node, 1), outputs = tensorNames(node, 2);
    if (inputs.size() != 2 || outputs.size() != 2) return 0;
    const auto k = constants.find(inputs[1]);
    if (k == constants.end() || k->second.size() != 1 || k->second[0] <= 3840) return 0;
    if (integerAttribute(node, "sorted", 1) != 1) throw std::runtime_error("异常 ONNX 的大K无序TopK无法等价缩减");
    std::string_view slice;
    size_t valueConsumers = 0;
    for (const auto& f : graphFields) {
        if (f.number == 12 && (textField(f.value, 1) == outputs[0] || textField(f.value, 1) == outputs[1]))
            throw std::runtime_error("异常 ONNX 的大K排序结果直接对外输出，不能缩减");
        if (f.number != 1) continue;
        for (const auto& input : tensorNames(f.value, 1)) {
            if (input == outputs[1]) throw std::runtime_error("异常 ONNX 的大K排序索引仍被使用，不能缩减");
            if (input == outputs[0]) { ++valueConsumers; slice = f.value; }
        }
    }
    if (valueConsumers != 1 || textField(slice, 4) != "Slice") throw std::runtime_error("异常 ONNX 的大K排序必须只有一个前缀切片消费者");
    const auto sliceInputs = tensorNames(slice, 1);
    if (sliceInputs.size() < 4 || sliceInputs[0] != outputs[0]) throw std::runtime_error("异常 ONNX 大K排序切片参数不受支持");
    const auto scalar = [&](const std::string& name) -> std::int64_t {
        const auto found = constants.find(name);
        if (found == constants.end() || found->second.size() != 1) throw std::runtime_error("异常 ONNX 大K排序切片需要静态单轴参数");
        return found->second[0];
    };
    const auto start = scalar(sliceInputs[1]), end = scalar(sliceInputs[2]), axis = scalar(sliceInputs[3]);
    const auto step = sliceInputs.size() > 4 ? scalar(sliceInputs[4]) : 1;
    if (start != 0 || step != 1 || axis != integerAttribute(node, "axis", -1) || end <= 0 || end > 3840)
        throw std::runtime_error("异常 ONNX 大K排序只能等价优化为同轴前3840项以内的连续前缀");
    return end;
}

/// 写入收缩后的TopK及新的K常量，索引输出无人使用，前缀数值顺序保持不变。
void appendLimitedTopK(std::string& graph, std::string_view node, std::int64_t k) {
    const auto constantName = textField(node, 2) + "_trt86_prefix_k";
    appendBytes(graph, 1, integerConstant(constantName, k));
    std::string converted;
    int inputIndex = 0;
    for (const auto& f : fields(node)) {
        if (f.number == 1 && inputIndex++ == 1) appendBytes(converted, 1, constantName);
        else converted.append(f.raw.data(), f.raw.size());
    }
    appendBytes(graph, 1, converted);
}

/// 将 opset18 Reduce* 的常量 axes 输入转为 opset17 属性，避免 TRT8.6 静默忽略归约轴。
std::string convertNode(std::string_view node, const std::map<std::string, std::vector<std::int64_t>>& constants) {
    const auto type = textField(node, 4);
    const bool reduce = type.rfind("Reduce", 0) == 0 && type != "ReduceSum";
    if (!reduce && type != "Resize") return std::string(node);
    std::vector<std::string> inputs;
    for (const auto& f : fields(node)) if (f.number == 1) inputs.emplace_back(f.value);
    std::string converted;
    int inputIndex = 0;
    for (const auto& f : fields(node)) {
        if (f.number == 1 && reduce && inputIndex++ > 0) continue;
        if (f.number == 5) {
            const auto name = textField(f.value, 1);
            if (reduce && name == "noop_with_empty_axes") {
                for (const auto& a : fields(f.value)) if (a.number == 3 && a.integer != 0) throw std::runtime_error("异常 ONNX 不支持空归约轴 noop 语义");
                continue;
            }
            if (type == "Resize" && (name == "antialias" || name == "keep_aspect_ratio_policy" || name == "axes")) {
                if (name == "antialias") for (const auto& a : fields(f.value)) if (a.number == 3 && a.integer != 0) throw std::runtime_error("网络内部含抗锯齿 Resize，不能直接降级为 TensorRT8.6");
                if (name == "keep_aspect_ratio_policy" && textField(f.value, 4) != "stretch") throw std::runtime_error("异常 ONNX 内部 Resize 不支持非stretch纵横比策略");
                if (name == "axes") throw std::runtime_error("异常 ONNX 内部 Resize 含未支持的 opset18 axes属性");
                continue;
            }
        }
        converted.append(f.raw.data(), f.raw.size());
    }
    if (reduce && inputs.size() > 1 && !inputs[1].empty()) {
        const auto found = constants.find(inputs[1]);
        if (found == constants.end() || found->second.empty()) throw std::runtime_error("异常 ONNX 归约轴不是非空 INT64 常量: " + inputs[1]);
        std::string axes; appendBytes(axes, 1, "axes"); appendInteger(axes, 20, 7);
        for (auto axis : found->second) appendInteger(axes, 8, static_cast<std::uint64_t>(axis));
        appendBytes(converted, 5, axes);
    }
    return converted;
}

/// 外移部署时插入的首个 Resize，保留其输出名作为新输入，避免改写内部张量引用。
std::string convertGraph(std::string_view graph, int height, int width) {
    std::map<std::string, std::vector<std::int64_t>> constants;
    std::string newInputName;
    size_t resizeCount = 0;
    const auto graphFields = fields(graph);
    for (const auto& f : graphFields) {
        if (f.number == 5) { const auto values = tensorIntegers(f.value); if (!values.empty()) constants[textField(f.value, 8)] = values; }
        if (f.number == 1 && textField(f.value, 3) == "deployment_antialiased_resize") {
            if (textField(f.value, 4) != "Resize" || textField(f.value, 1) != "image") throw std::runtime_error("异常 ONNX 部署 Resize 结构不匹配");
            newInputName = textField(f.value, 2); ++resizeCount;
        }
        if (f.number == 1 && textField(f.value, 4) == "Constant") {
            for (const auto& a : fields(f.value)) if (a.number == 5 && textField(a.value, 1) == "value") {
                for (const auto& v : fields(a.value)) if (v.number == 5) constants[textField(f.value, 2)] = tensorIntegers(v.value);
            }
        }
    }
    if (resizeCount != 1 || newInputName.empty()) throw std::runtime_error("异常 ONNX 必须包含唯一的 deployment_antialiased_resize 节点，请使用已校准的导出模型");
    std::map<std::string, std::int64_t> topKLimits;
    std::map<std::string, bool> changedShapeNames;
    for (const auto& f : graphFields) if (f.number == 1 && textField(f.value, 4) == "TopK") {
        const auto limit = limitedTopK(f.value, graphFields, constants);
        if (limit) {
            topKLimits[textField(f.value, 3)] = limit;
            for (const auto& name : tensorNames(f.value, 2)) changedShapeNames[name] = true;
        }
    }
    std::string converted;
    for (const auto& f : graphFields) {
        if (f.number == 1) {
            if (textField(f.value, 3) == "deployment_antialiased_resize") continue;
            const auto topK = topKLimits.find(textField(f.value, 3));
            if (topK != topKLimits.end()) appendLimitedTopK(converted, f.value, topK->second);
            else if (textField(f.value, 4) == "Shape") appendShapeNodes(converted, f.value);
            else appendBytes(converted, 1, convertNode(f.value, constants));
        } else if (f.number == 11) appendBytes(converted, 11, makeInput(newInputName, height, width));
        // TopK被等价收缩后移除其过时的内部value_info；图对外输出的尺寸与含义保持原样。
        else if (f.number == 13 && changedShapeNames.count(textField(f.value, 1))) continue;
        else converted.append(f.raw.data(), f.raw.size());
    }
    return converted;
}
}

/// 仅适配合同版本1的已校准异常图，写入外部缩放约定后由公共构建器保存元数据。
bool prepareAnomalyOnnx(std::vector<char>& bytes, std::map<std::string, std::string>& metadata) {
    if (!metadata.count("anomalib.model_type")) return false;
    if (metadata["anomalib.contract_version"] != "1" || metadata["anomalib.preprocessing"] != "embedded") throw std::runtime_error("异常 ONNX 合同版本或预处理约定不受支持");
    const int height = dimension(metadata, "anomalib.resize_height"), width = dimension(metadata, "anomalib.resize_width");
    std::string converted;
    for (const auto& f : fields(std::string_view(bytes.data(), bytes.size()))) {
        if (f.number == 7) appendBytes(converted, 7, convertGraph(f.value, height, width));
        else if (f.number == 8 && textField(f.value, 1).empty()) { std::string opset; appendInteger(opset, 2, 17); appendBytes(converted, 8, opset); }
        else converted.append(f.raw.data(), f.raw.size());
    }
    metadata["anomalib.preprocessing"] = "external_antialias";
    metadata["trt_model.anomaly_compatibility"] = "opset18_reduce_shape_topk_to17_v2";
    bytes.assign(converted.begin(), converted.end()); return true;
}
}
