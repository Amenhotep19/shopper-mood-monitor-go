# Building Docker Images in Microsoft Azure Cloud

This is a short step by step tutorial about how to build [Docker](https://docker.com) image for `shopper-mood-monitor-go` application in [Microsoft Azure](https://azure.microsoft.com/) Cloud and make it available in [Azure Container Registry (ACR)](https://docs.microsoft.com/en-us/azure/container-registry/).

## Prerequisities

* Create a free Azure account by following the guide [here](https://azure.microsoft.com/en-us/free/)
* Install `azure-cli` tool by followin the guid [here](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest)

You should now be ready to proceed with the next steps

## Create Azure Container Registry

Before you can start building `docker` images you need to create a container registry where the built images will be stored and available for download. Before you can create an Azure Container Registry (ACR) you first need to create a [Resource Group](https://docs.microsoft.com/en-us/azure/azure-resource-manager/resource-group-overview#resource-groups) with which you will then associate your ACR.

Go ahead and login to your Azure account and create a new resource group by running the commands below:

```shell
az login
```

This will open a browser window and prompts you for you Azure password. Once you've authenticated you are ready to proceed further.

Pick a geographic location that suits your current geography. You can get a list of location as follows:

```shell
az account list-locations
```

For this tutorial we will work with assumption youre based in `westeurope`. Go ahead and create Azure Resource Group now:

```shell
az group create --name myResourceGroup --location westeurope
```

Now that you have created resource group, you can proceed by creating Azure Container Registry which will store all docker images and make them available for download. Note that you need to pick a **unique** name for your registry:

```shell
az acr create --resource-group myResourceGroup --name myOpenVinoGocv --sku Standard
```

With Azure Container Registry running on your account you can now proceed with building the application docker image.


## Build docker image and push to ACR

You are now ready to build a `docker` image for `shopper-mood-monitor-go` application and stored it in the ACR you had built earlier. We assume you have already cloned the `shopper-mood-monitor-go` git repository as per instructions in [README](./README.md).

First you need to log in to the ACR you built earlier:

```shell
az acr login --name myOpenVinoGocv
```

You should see the output that your login was successful:

```shell
Login Succeeded
```

You can list all available ACRs in your Azure account:

```shell
az acr list --resource-group myResourceGroup --query "[].{acrLoginServer:loginServer}" --output table
```

You should see output that looks something like this:

```shell
AcrLoginServer
-------------------------
myopenvinogocv.azurecr.io
```

Before you are able to build and push new docker images into the ACR you need to allow access to it. There is a full documetation about Role Based Access Control using Azure AD which you can read about online. For the purpose of this guide we will grant ourselves **admin** privileges i.e. full read/write access:

```shell
az acr update --name myOpenVinoGocv --admin-enabled true
```

```shell
{
  "adminUserEnabled": true,
  "creationDate": "2018-12-04T13:35:38.338340+00:00",
  "id": "/subscriptions/someguidhere/resourceGroups/myResourceGroup/providers/Microsoft.ContainerRegistry/registries/myOpenVinoGocv",
  "location": "westeurope",
  "loginServer": "myopenvinogocv.azurecr.io",
  "name": "myOpenVinoGocv",
  "provisioningState": "Succeeded",
  "resourceGroup": "myResourceGroup",
  "sku": {
    "name": "Standard",
    "tier": "Standard"
  },
  "status": null,
  "storageAccount": null,
  "tags": {},
  "type": "Microsoft.ContainerRegistry/registries"
}
```

If you have have not yet not done so, obtain your own unique download URL for the Intel distribution of OpenVINO toolkit as described in the main README under "Docker".

Now for the final part, we can build a Docker image and automatically upload it to our ACR by running the command below, substituting your own unique download URL:


```shell
az acr build --resource-group myResourceGroup --registry myOpenVinoGocv --image shopper-mood-monitor-go --build-arg OPENVINO_DOWNLOAD_URL=[your unique OpenVINO download URL here] .
```

If everything went fine you should see the output similar to the one below:

```shell
The following dependencies were found:
- image:
    registry: myopenvinogocv.azurecr.io
    repository: shopper-mood-monitor-go
    tag: latest
    digest: sha256:fd1d337bf7384a8e33ed8a73a0948d02520ce0fada32ce24efac627c7de9de23
  runtime-dependency:
    registry: registry.hub.docker.com
    repository: library/ubuntu
    tag: "16.04"
    digest: sha256:e547ecaba7d078800c358082088e6cc710c3affd1b975601792ec701c80cdd39
  git: {}

2018/11/20 14:26:48 Successfully populated digests for step ID: build
2018/11/20 14:26:48 Step ID: push marked as successful (elapsed time in seconds: 131.494060)

Run ID: cb2 was successful after 7m3s
```

Now you are ready to run the example wherever you have `docker` cli available as follows. Note that the docker image tag contains a DNS name pointing to ACR printed in the output shown above under `registry` key:

```shell
docker run -it --rm myopenvinogocv.azurecr.io/shopper-mood-monitor-go -h
```

## Destroy Azure environment

If you no longer need ACR you can easily remove all the resources by deleting particular resource group as follows:

```shell
az group delete --name myResourceGroup
```
