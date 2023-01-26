rm -rf env
rm -rf data/*.pt data/*.onnx
find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
docker container stop $(docker container ls -aq); docker system prune -af
