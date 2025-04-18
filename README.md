# NearbyNLU

A natural language understanding (NLU) system that processes location-based queries and interfaces with Google Maps API to provide relevant results.

## Project Structure

```
tensorflow-maps-nlp/
├── app/                            # Main backend application logic
├── tests/                          # Unit tests
├── notebooks/                      # Jupyter notebooks for training/model dev
├── data/                           # Training data and vocab files
├── models/                         # Saved TensorFlow models
├── requirements.txt                # Python dependencies
└── .env                            # Environment variables
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your Google Maps API key:
```
GOOGLE_MAPS_API_KEY=your_api_key_here
```

4. Run the application:
```bash
python app/main.py
```

## Development

- Model development notebooks are in the `notebooks/` directory
- Unit tests can be run with `pytest tests/`
- The main application logic is in `app/`

## License

MIT License 