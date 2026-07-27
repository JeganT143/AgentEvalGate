// Secret-management infrastructure for the AgentEvalGate API: a Key Vault holding
// AEG_API_KEY, wired to the Container App via its own system-assigned managed
// identity and Container Apps' native Key Vault secret reference. The secret VALUE
// never lives in this file, any other committed file, or the container image built
// by docker/Dockerfile.api - see docs/build_log.md (Day 6 / Step 3) and
// internal/mentoring_notes.md for why that specific property matters for an image
// that gets publicly pulled.
//
// `min replicas: 0` below carries forward Day 6 / Step 1's scale-to-zero decision.
//
// NOT YET COMPILED against a real Azure subscription in this environment (no
// working `az` or `bicep` CLI available here - attempted, binary download came
// back corrupted). Run `az bicep build --file infra/main.bicep` (and
// `az deployment group validate --what-if`) before deploying this for real.

@description('Azure region for all resources.')
param location string = resourceGroup().location

@description('Globally-unique Key Vault name (3-24 chars, alphanumeric/hyphens).')
param keyVaultName string

@description('Container Apps managed environment name.')
param containerAppEnvName string = 'agentevalgate-env'

@description('Container App name.')
param containerAppName string = 'agentevalgate-api'

@description('Fully-qualified container image reference (registry/repo:tag). No default on purpose - the registry choice (ACR vs. Docker Hub) hasn\'t been made yet as of Day 6 / Step 3, so this forces an explicit value instead of silently deploying a placeholder.')
param containerImage string

@description('The real API key value, supplied ONLY at deploy time, e.g. `az deployment group create ... --parameters apiKeyValue=$AEG_API_KEY`. Never hardcode this, never put it in a committed .bicepparam file - @secure() keeps it out of Azure deployment history and command output, but it is still your job to never let it touch a file that gets committed.')
@secure()
param apiKeyValue string

// Built-in "Key Vault Secrets User" role - lets a principal read secret values but not
// manage the vault itself (least privilege: the Container App only ever needs to read).
var keyVaultSecretsUserRoleId = '4633458b-17de-408a-b874-0445c86b69e6'

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: keyVaultName
  location: location
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: subscription().tenantId
    // RBAC over access policies: role assignments are auditable, scoped, and
    // consistent with how every other Azure resource in this project grants access.
    enableRbacAuthorization: true
  }
}

resource apiKeySecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: keyVault
  name: 'aeg-api-key'
  properties: {
    value: apiKeyValue
  }
}

resource containerAppEnv 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: containerAppEnvName
  location: location
  properties: {}
}

resource containerApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: containerAppName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    environmentId: containerAppEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
      }
      // This is the Container Apps "secret reference" mechanism the task asked for
      // at minimum - backed by Key Vault (the task's preferred option) instead of a
      // plaintext value, so the two options aren't actually a tradeoff here: this
      // gets both at once. `identity: 'system'` means the platform reads the secret
      // using the Container App's own managed identity - no credential of any kind
      // is stored on the Container App resource itself.
      secrets: [
        {
          name: 'aeg-api-key'
          keyVaultUrl: apiKeySecret.properties.secretUri
          identity: 'system'
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'api'
          image: containerImage
          env: [
            {
              // The app reads this exactly like it does locally via .env
              // (AEG_API_KEY, src/config.py::Settings) - only the source of the
              // value changes between local dev and deployed, not the app code.
              name: 'AEG_API_KEY'
              secretRef: 'aeg-api-key'
            }
          ]
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 3
      }
    }
  }
}

resource keyVaultSecretsUserRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, containerApp.id, keyVaultSecretsUserRoleId)
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', keyVaultSecretsUserRoleId)
    principalId: containerApp.identity.principalId
    principalType: 'ServicePrincipal'
  }
}
