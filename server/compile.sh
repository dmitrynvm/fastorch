rm -rf build
mkdir -p build
cd build
cmake -Wno-dev -DCMAKE_PREFIX_PATH=/usr/local/lib/torch ..
make -j4
cd ..
mv build/predict .
rm -rf build

cp /usr/local/lib/onnx/lib/* /usr/local/lib
cp /usr/local/lib/torch/lib/* /usr/local/lib
ldconfig
