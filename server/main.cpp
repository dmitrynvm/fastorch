#include <cstdlib>
#include <torch/script.h>
#include <torch/serialize/tensor.h>
#include <torch/serialize.h>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>
#include "crow.hpp"
#include "base64.hpp"


int PORT = 8000;
 
template <typename T> T vectorProduct(const std::vector<T>& v) {
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

std::vector<std::string> readLabels(std::string& labelFilepath) {
    std::vector<std::string> labels;
    std::string line;
    std::ifstream fp(labelFilepath);
    while (std::getline(fp, line)) {
        labels.push_back(line);
    }
    return labels;
}


int main(int argc, char **argv) {
    int width = 224;
    int height = 224;
    std::vector<double> mean = {0.485, 0.456, 0.406};
    std::vector<double> std = {0.229, 0.224, 0.225};

    std::string labelPath = argv[2];
    std::string modelPath = argv[1];

    auto labels = readLabels(labelPath);

    const int64_t batchSize = 1;
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "server");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    Ort::Session session(env, "/data/mobile.onnx", options);
    Ort::AllocatorWithDefaultOptions allocator;
    size_t numInputNodes = session.GetInputCount();
    size_t numOutputNodes = session.GetOutputCount();
    const char* inputName = session.GetInputName(0, allocator);
    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
    if (inputDims.at(0) == -1) {
        std::cout << "Got dynamic batch size. Setting input batch size to "
        << batchSize << "." << std::endl;
        inputDims.at(0) = batchSize;
    }
    const char* outputName = session.GetOutputName(0, allocator);


    crow::SimpleApp app;

    CROW_ROUTE(app, "/").methods("GET"_method)
    ([](){
        crow::json::wvalue res;
        res["server"] = "works";
        std::ostringstream os;
        os << crow::json::dump(res);
        return crow::response{os.str()};
    });

    CROW_ROUTE(app, "/predict").methods("POST"_method)
    ([
        &width,
        &height,
        &mean,
        &std,
        &labels,
        &session,
        &inputDims,
        &allocator,
        &inputName,
        &outputName
    ](const crow::request& req) {
        crow::json::wvalue res;
        res["index"] = "";
        res["label"] = "";
        res["proba"] = "";
        std::ostringstream os;
        auto args = crow::json::load(req.body);

        cv::Mat imageBGR = cv::imread("/data/tench.jpg", cv::ImreadModes::IMREAD_COLOR);
        cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
        cv::resize(
            imageBGR, 
            resizedImageBGR, 
            cv::Size(inputDims.at(3), inputDims.at(2)), 
            cv::InterpolationFlags::INTER_CUBIC
        ); 
        cv::cvtColor(resizedImageBGR, resizedImageRGB, cv::ColorConversionCodes::COLOR_BGR2RGB);
        resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);
        cv::Mat channels[3];
        cv::split(resizedImage, channels);
        channels[0] = (channels[0] - 0.485) / 0.229;
        channels[1] = (channels[1] - 0.456) / 0.224;
        channels[2] = (channels[2] - 0.406) / 0.225;
        cv::merge(channels, 3, resizedImage);
        cv::dnn::blobFromImage(resizedImage, preprocessedImage);
        size_t inputTensorSize = vectorProduct(inputDims);
        std::vector<float> inputTensorValues(inputTensorSize);
        for (int64_t i = 0; i < batchSize; ++i) {
            std::copy(
                preprocessedImage.begin<float>(),
                preprocessedImage.end<float>(),
                inputTensorValues.begin() + i * inputTensorSize / batchSize
            );
        }
        Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
        std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
        size_t outputTensorSize = vectorProduct(outputDims);
        assert(("Output tensor size should equal to the label set size.", labels.size() * batchSize == outputTensorSize));
        std::vector<float> outputTensorValues(outputTensorSize);
        std::vector<const char*> inputNames{inputName};
        std::vector<const char*> outputNames{outputName};
        std::vector<Ort::Value> inputTensors;
        std::vector<Ort::Value> outputTensors;
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(), inputDims.size()));
        outputTensors.push_back(Ort::Value::CreateTensor<float>( memoryInfo, outputTensorValues.data(), outputTensorSize, outputDims.data(), outputDims.size()));

        session.Run(
            Ort::RunOptions{nullptr}, 
            inputNames.data(),
            inputTensors.data(), 
            1, 
            outputNames.data(),
            outputTensors.data(), 
            1
        );

        std::vector<int> predIds(batchSize, 0);
        std::vector<std::string> predLabels(batchSize);
        std::vector<float> confidences(batchSize, 0.0f);
        for (int64_t b = 0; b < batchSize; ++b) {
            float activation = 0;
            float maxActivation = std::numeric_limits<float>::lowest();
            float expSum = 0;
            for (int i = 0; i < labels.size(); i++) {
                activation = outputTensorValues.at(i + b * labels.size());
                expSum += std::exp(activation);
                if (activation > maxActivation) {
                predIds.at(b) = i;
                maxActivation = activation;
                }
            }
            predLabels.at(b) = labels.at(predIds.at(b));
            confidences.at(b) = std::exp(maxActivation) / expSum;
        }
        for (int64_t b = 0; b < batchSize; ++b) {
            assert(("Output predictions should all be identical.",
            predIds.at(b) == predIds.at(0)));
        }

        res["index"] = predIds.at(0);
        res["label"] = predLabels.at(0);
        res["proba"] = confidences.at(0);
    
        os << crow::json::dump(res);
        return crow::response{os.str()};
    });
    app.port(PORT).run();
    return 0;
}
