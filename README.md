# streamlit-leaf

Streamlit app for leaf segmentation.

To run locally:

```
docker build -t streamlit-leaf:latest .
docker run -p 8501:8501 streamlit-leaf:latest
```

To view python print statements during development, use:

```
docker run -e PYTHONUNBUFFERED=1 -p 8501:8501 streamlit-leaf:latest
```

The streamlit app can then be accessed at `http://localhost:8501`.