tag=${1}
operator_image="${operator_image:-quay.io/domino/domino-extensions-api}"
docker buildx build --platform=linux/amd64 -f ./Dockerfile -t ${operator_image}:${tag} .
docker push ${operator_image}:${tag}