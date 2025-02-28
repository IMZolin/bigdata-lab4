# mle-template
Classic MLE template with CI/CD pipelines

```
python -m src.train
```

```
python -m src.preprocess
```

```
python -m src.predict --mode predict --message 'I love this product'
```

```
uvicorn src.app:app --reload
```

```
coverage run -m unittest src.unit_tests.test_preprocess

coverage report
```

```
curl -X 'POST' \
  'http://127.0.0.1:8000/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "message": "I love this product!"
}'
```