CONTAINER = jesusgf23/simsapp

.PHONY: build exec push run go train release 
exec:
	docker exec -it $(CONTAINER) /bin/bash

build:
	docker build -t $(CONTAINER) .
	
push:
	docker push $(CONTAINER)

run:
	docker run -it $(CONTAINER) /bin/bash

go:
	make build && make push

train:
	python example.py


release:
	rm -rf dist/ && \
	python setup.py sdist bdist_wheel && \
	twine upload dist/* --verbose

clean:
	rm -rf dist/ build/ *.egg-info