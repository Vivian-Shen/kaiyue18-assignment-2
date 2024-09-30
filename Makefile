# Define the virtual environment directory and Flask app file
VENV = venv
FLASK_APP = app.py

# Create and activate the virtual environment, then install dependencies
install:
	python3 -m venv $(VENV)
	./$(VENV)/bin/pip install -r requirements.txt

# Run the Flask application on port 3000
run:
	FLASK_APP=$(FLASK_APP) FLASK_ENV=development ./$(VENV)/bin/flask run --host=0.0.0.0 --port=3000

# Clean the virtual environment
clean:
	rm -rf $(VENV)

# Reinstall all dependencies by cleaning and reinstalling
reinstall: clean install