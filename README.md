# Fast PyTorch Model Server

The guide explaines how to run inference server and create a virtual enviroment for manual testing of this server. 

1. Download and enter the repository
```bash
    git clone https://github.com/dmitrynvm/fastorch
    cd fastorch
```

2. Create enviroment and install dependencies
```bash
    make init
    source env/bin/activate
```

3. Build container and application
```bash
    make build
```

4. Start server
```bash
	make up
```
5. Run tests and manual pings
```bash
    make test
    make ping
```

6. Run the container and debug
```bash
	make run
```

7. Clean up the environment
```bash
    deactivate
    make clean
```
