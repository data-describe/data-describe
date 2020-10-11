# AI Platform Notebooks

This docker image provides data-describe installed for an AI Platform Notebook custom container.

## Building the image

To build the image, run from the root directory of the repository:

```
docker-compose build aiplatform
```

### Using AI Platform

You will need to push the docker image to a hosted repository such as Google Container Registry.

`<PROJECT_ID>` should be your GCP Project ID
`<IMAGE_NAME>` can be any name you choose, e.g. `data-describe`

```
docker tag data-describe_aiplatform gcr.io/<PROJECT_ID>/<IMAGE_NAME>
docker push gcr.io/<PROJECT_ID>/<IMAGE_NAME>
```

To use the image, create a new Notebook with a custom container configured using the container (`gcr.io/<PROJECT_ID>/<IMAGE_NAME>`)