# Academic-Risk-Engagement-Prediction-System
uv python install 3.12
uv python pin 3.12

# Addition dependencies (Alternate method):
uv pip install jupyter ipykernel
uv run python -m ipykernel install --user --name docscanner_env --display-name "Doc Scanner (uv)"

To install/freeze additional libraries with UV:

uv pip install -r requirements.txt
uv pip freeze > requirements.txt
uv pip install -r requirements_new.txt

To run files:

uv run python hello_world.py
uv run streamlit run app.py