# Deployment Documentation

1. Create docker image (don't need docker file) e.g.:
```
langgraph build \  --platform linux/amd64 \
  -t {docker_username}/langgraph-agent:1.0
```

2. Pull the image to docker hub:
```
docker push {docker_username}/langgraph-agent:1.0
```

3. Use image url in deployment with necessary credentials for DB, Redis, LLM API.
```
docker.io/{docker_image}/langgraph-agent:1.0
```