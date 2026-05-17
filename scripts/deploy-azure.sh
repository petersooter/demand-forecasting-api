#!/usr/bin/env bash
# deploy-azure.sh
# Builds the Docker image, pushes to Azure Container Registry,
# and deploys to Azure App Service (Web App for Containers).
#
# Prerequisites:
#   az login
#   az extension add --name containerapp  (if using Container Apps)
#
# Usage:
#   ./scripts/deploy-azure.sh
#   RESOURCE_GROUP=my-rg ./scripts/deploy-azure.sh

set -euo pipefail

# ── Config (override via env vars) ──────────────────────────────────────────
RESOURCE_GROUP="${RESOURCE_GROUP:-demand-forecasting-rg}"
LOCATION="${LOCATION:-eastus}"
ACR_NAME="${ACR_NAME:-demandforecastingacr}"
APP_NAME="${APP_NAME:-demand-forecasting-api}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
APP_SERVICE_PLAN="${APP_SERVICE_PLAN:-demand-forecasting-plan}"

IMAGE="${ACR_NAME}.azurecr.io/${APP_NAME}:${IMAGE_TAG}"

echo "Deploying: ${IMAGE}"
echo "Resource group: ${RESOURCE_GROUP} (${LOCATION})"
echo ""

# ── 1. Resource group ────────────────────────────────────────────────────────
echo "Creating resource group..."
az group create \
  --name "${RESOURCE_GROUP}" \
  --location "${LOCATION}" \
  --output none

# ── 2. Container registry ────────────────────────────────────────────────────
echo "Creating container registry..."
az acr create \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${ACR_NAME}" \
  --sku Basic \
  --admin-enabled true \
  --output none

# ── 3. Build and push ────────────────────────────────────────────────────────
echo "Building and pushing image..."
az acr build \
  --registry "${ACR_NAME}" \
  --image "${APP_NAME}:${IMAGE_TAG}" \
  .

# ── 4. App Service plan ──────────────────────────────────────────────────────
echo "Creating App Service plan..."
az appservice plan create \
  --name "${APP_SERVICE_PLAN}" \
  --resource-group "${RESOURCE_GROUP}" \
  --is-linux \
  --sku B1 \
  --output none

# ── 5. Web App ───────────────────────────────────────────────────────────────
ACR_PASSWORD=$(az acr credential show --name "${ACR_NAME}" --query "passwords[0].value" -o tsv)

echo "Creating Web App..."
az webapp create \
  --resource-group "${RESOURCE_GROUP}" \
  --plan "${APP_SERVICE_PLAN}" \
  --name "${APP_NAME}" \
  --deployment-container-image-name "${IMAGE}" \
  --docker-registry-server-url "https://${ACR_NAME}.azurecr.io" \
  --docker-registry-server-user "${ACR_NAME}" \
  --docker-registry-server-password "${ACR_PASSWORD}" \
  --output none

# ── 6. Health check config ───────────────────────────────────────────────────
az webapp config set \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${APP_NAME}" \
  --generic-configurations '{"healthCheckPath": "/health"}' \
  --output none

echo ""
echo "Deployed: https://${APP_NAME}.azurewebsites.net"
echo "Docs:     https://${APP_NAME}.azurewebsites.net/docs"
