export type AppEnv = "prod" | "test" | "local";

export const ENV_CONFIG = {
  prod: {
    authUrl: "https://lucid-v3-prod.westeurope.cloudapp.azure.com/auth",
    apiUrl: "https://lucid-v3-prod.westeurope.cloudapp.azure.com/lucid",
    deploymentUrl: "https://api.agent.zypl.ai",
  },
  test: {
    authUrl: "https://test.lucid.zypl.ai/api/auth",
    apiUrl: "https://test.lucid.zypl.ai/api/lucid",
    deploymentUrl: "https://api.agent.test.zypl.ai",
  },
  // Local: uses test auth/api URLs, deployment URL is entered by the user
  local: {
    authUrl: "https://test.lucid.zypl.ai/api/auth",
    apiUrl: "https://test.lucid.zypl.ai/api/lucid",
    deploymentUrl: "", // set at runtime from LOCAL_DEPLOYMENT_URL_KEY
  },
} as const;

const ENV_STORAGE_KEY = "lucid_app_env";
export const LOCAL_DEPLOYMENT_URL_KEY = "lucid_local_deployment_url";

export function getAppEnv(): AppEnv {
  if (typeof window === "undefined") return "prod";
  return (sessionStorage.getItem(ENV_STORAGE_KEY) as AppEnv) ?? "prod";
}

export function setAppEnv(env: AppEnv): void {
  sessionStorage.setItem(ENV_STORAGE_KEY, env);
}

export function getLocalDeploymentUrl(): string {
  if (typeof window === "undefined") return "";
  return localStorage.getItem(LOCAL_DEPLOYMENT_URL_KEY) ?? "";
}

export function setLocalDeploymentUrl(url: string): void {
  localStorage.setItem(LOCAL_DEPLOYMENT_URL_KEY, url);
}

export const ML_API_CONFIG = {
  BASE_URL: "/ml-api",
  BASIC_AUTH_HEADER: "Basic enlwbF9tb2RlbDpLSnZ4amh5eWUz",
  PARTNER: "zypl",
  PROJECT_ID: "fc11354b-1188-4192-9189-6cba6edbbf38",
  UPLOAD_LABEL_PREFIX: "train-upload",
  BLIND_SERVICE_URL: "https://microservice-r2.onrender.com",
};
