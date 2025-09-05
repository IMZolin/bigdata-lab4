# MLE lab2, Ivan Zolin M4145

### Build and Run Services

```bash
# Build Docker images from the Dockerfile
docker-compose build

# Start the REST API service
docker-compose up web

# Run the tests inside the test service
docker-compose up unit_test
```

### Example Request to the Web Service

```
curl -X POST \
  http://localhost:8000/predict/ \
  -H "Accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"message": "I love this product!"}'
```

Expected response:
```
{
  "sentiment": "Positive sentiment"
}
```

```
curl -X GET http://localhost:8000/health/ -H "Content-Type: application/json"
``` 

Expected response:
```
{
  "status": "OK"
}
```
