init: 
	./scripts/init.sh

build:
	docker build -t server -f docker/cpu.Dockerfile .

run:
	docker run -it -p 8000:8000 --entrypoint /bin/bash server

up:
	docker run -p 8000:8000 server

test:
	pytest --log-cli-level=INFO

ping:
	python3 test/ping.py 

clean: 
	./scripts/clean.sh
