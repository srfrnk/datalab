all:
	pass

build:
	docker-compose -f docker-compose.json build

run:
	export PROJECT_ID=$$(gcloud config get-value core/project) &&\
	docker-compose -f docker-compose.json up

stop:
	docker-compose -f docker-compose.json kill

setup-local:
	sudo pip install -r requirements.txt

interactive:
	docker-compose -f docker-compose.json exec datalab bash
