# ══════════════════════════════════════════════════════════════════════════════
#  Healthkaton — Makefile
#  Requires: Docker Desktop + make (Git Bash / WSL / Chocolatey: choco install make)
#
#  Quick reference:
#    make build     – build both images
#    make up        – build & start (foreground)
#    make start     – start detached (background)
#    make stop      – stop containers (keep them)
#    make down      – stop & remove containers
#    make clean     – down + remove images + prune build cache
#    make restart   – down then start detached
#    make logs      – tail all logs
#    make logs-be   – tail backend logs only
#    make logs-fe   – tail frontend logs only
#    make status    – show running containers
#    make shell-be  – open bash shell in backend container
#    make shell-fe  – open sh shell in frontend container
# ══════════════════════════════════════════════════════════════════════════════

COMPOSE  = docker compose
PROJECT  = healthkaton

.PHONY: build up start stop down clean restart logs logs-be logs-fe status shell-be shell-fe

## Build both images without starting
build:
	$(COMPOSE) build

## Build & start in foreground (Ctrl+C to stop)
up:
	$(COMPOSE) up --build

## Build & start detached (background)
start:
	$(COMPOSE) up --build -d
	@echo ""
	@echo "  App running at http://localhost"
	@echo "  Run 'make logs' to tail logs, 'make down' to stop."

## Stop containers without removing them
stop:
	$(COMPOSE) stop

## Stop & remove containers (images kept)
down:
	$(COMPOSE) down

## Full cleanup: containers + images + dangling build cache
clean:
	$(COMPOSE) down --rmi all --volumes --remove-orphans
	docker builder prune -f
	@echo "All $(PROJECT) containers, images and cache removed."

## Restart: tear down then start fresh in background
restart: down start

## Tail logs for all services
logs:
	$(COMPOSE) logs -f

## Tail backend logs only
logs-be:
	$(COMPOSE) logs -f backend

## Tail frontend logs only
logs-fe:
	$(COMPOSE) logs -f frontend

## Show container status
status:
	$(COMPOSE) ps

## Open a shell in the backend container
shell-be:
	$(COMPOSE) exec backend bash

## Open a shell in the frontend container (nginx uses sh, not bash)
shell-fe:
	$(COMPOSE) exec frontend sh
