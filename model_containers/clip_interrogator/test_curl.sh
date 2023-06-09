#!/bin/bash

set -euxo pipefail

if [ "${1+x}" == "x" ]; then
    container_name="$1"
else
    # The container name is the directory name with '_' replaced with '-'.
    container_name=$(echo $(basename "$(realpath "$(dirname "$0")")") | sed 's/_/-/g')
fi

docker_port=$(docker port "${container_name}" 8000/tcp)
ENDPOINT=localhost:${docker_port##*:}

# The expected output will be a JSON style response with {"transcription":[" And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."]}
# curl -X POST http://${ENDPOINT}/predict \
#     -H "Content-Type: application/json" \
#     --data '{
#         "image_path": "thierry.png"}' > response.json
echo "{\"mode\": \"classic\", " > req.json
echo "\"image\": \"" >> req.json
base64 thierry.png >> req.json
echo "\"}" >> req.json

curl -X POST http://${ENDPOINT}/predict \
    -H "Content-Type: application/json" \
    --data @req.json > response.json
