mkdir /tmp/onnx
cd /tmp/onnx
mkdir /onnxruntime
mkdir /onnxruntime/lib
wget -O onnx_archive.nupkg https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime/1.7.0
unzip onnx_archive.nupkg
cp runtimes/linux-x64/native/libonnxruntime.so /onnxruntime/lib/
cp -r build/native/include/ /onnxruntime/include
rm -rf /tmp/onnx