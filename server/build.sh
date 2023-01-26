rm -rf trash
mkdir -p trash
cd trash
cmake -Wno-dev -DCMAKE_PREFIX_PATH=/usr/local/lib/torch ..
make -j4
cd ..
mv trash/predict .
rm -rf trash

cp /usr/local/lib/onnx/lib/* /usr/local/lib
cp /usr/local/lib/torch/lib/* /usr/local/lib
ldconfig
