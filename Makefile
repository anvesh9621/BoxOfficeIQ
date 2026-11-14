.PHONY: install run-backend run-frontend train db-init test clean

install:
	pip install -r backend/requirements.txt
	pip install -r ml/requirements.txt

run-backend:
	cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

run-frontend:
	cd frontend && npm install && npm run dev

train:
	python ml/training/train.py

db-init:
	cd backend && alembic upgrade head

test:
	pytest backend/tests/ -v --tb=short

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf dist/ build/ htmlcov/ .coverage
