import { ML_API_CONFIG, getAppEnv } from "./mlConfig";

function envHeaders(extra?: Record<string, string>): Record<string, string> {
  return { "x-lucid-env": getAppEnv(), ...extra };
}

export async function authenticate(
    login?: string,
    password?: string,
): Promise<string | null> {
    const url = `${ML_API_CONFIG.BASE_URL}/auth/login`;
    const headers = {
        ...envHeaders(),
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": ML_API_CONFIG.BASIC_AUTH_HEADER,
    };
    const credentials = {
        login: login ?? "",
        password: password ?? "",
    };

    try {
        const response = await fetch(url, {
            method: "POST",
            headers,
            body: JSON.stringify(credentials),
        });

        if (!response.ok) {
            return null;
        }

        const data = await response.json();
        return data.active_token || null;
    } catch (error) {
        console.error("Error during authentication:", error);
        return null;
    }
}

export async function getAgentKey(activeToken: string, partner?: string): Promise<string | null> {
    const url = `${ML_API_CONFIG.BASE_URL}/auth/agent?partner=${partner ?? ML_API_CONFIG.PARTNER}`;
    try {
        const response = await fetch(url, {
            method: "POST",
            headers: {
                ...envHeaders(),
                "Accept": "application/json",
                "Authorization": `Bearer ${activeToken}`,
            },
        });
        if (!response.ok) return null;
        const data = await response.json();
        return data.agent_key || null;
    } catch (error) {
        console.error("Error fetching agent key:", error);
        return null;
    }
}

export async function uploadTrain(file: File, token: string): Promise<string | null> {
    const partner = ML_API_CONFIG.PARTNER;
    const projectId = ML_API_CONFIG.PROJECT_ID;
    const url = `${ML_API_CONFIG.BASE_URL}/lucid/train/${projectId}/upload?partner=${partner}`;

    const headers = {
        ...envHeaders(),
        "Accept": "application/json",
        "Authorization": `Bearer ${token}`,
    };

    const formData = new FormData();
    formData.append("train", file);

    const now = new Date();
    const isoString = now.toISOString();
    const timestamp = isoString.split(".")[0].replace(/-/g, "").replace(/:/g, "").replace("T", "");
    const label = `${ML_API_CONFIG.UPLOAD_LABEL_PREFIX}_${timestamp}`;
    formData.append("label", label);

    try {
        const response = await fetch(url, {
            method: "POST",
            headers,
            body: formData,
        });

        if (!response.ok) {
            console.error("Upload failed:", response.status, response.statusText);
            return null;
        }

        const data = await response.json();
        return data.train_id || null;
    } catch (error) {
        console.error("Error during file upload:", error);
        return null;
    }
}

export interface BlindUploadResponse {
    upload_url: string;
    agent_download_url: string;
    s3_key: string;
}

export async function getBlindUploadUrl(filename: string): Promise<BlindUploadResponse | null> {
    const url = `${ML_API_CONFIG.BLIND_SERVICE_URL}/get-upload-url`;

    const formData = new FormData();
    formData.append("filename", filename);

    try {
        const response = await fetch(url, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            console.error("Failed to get blind upload URL:", response.status, response.statusText);
            return null;
        }

        return await response.json();
    } catch (error) {
        console.error("Error fetching blind upload URL:", error);
        return null;
    }
}

export interface ZganTaskStatusResponse {
    status: "done" | "failed" | "processing" | "pending" | string;
    result?: Record<string, unknown>;
}

export async function checkZganTaskStatus(
    task_id: string,
    token: string,
): Promise<ZganTaskStatusResponse | null> {
    const partner = ML_API_CONFIG.PARTNER;
    const url = `${ML_API_CONFIG.BASE_URL}/lucid/task/${task_id}/result?partner=${partner}`;

    try {
        const response = await fetch(url, {
            method: "GET",
            headers: {
                ...envHeaders(),
                Accept: "application/json",
                Authorization: `Bearer ${token}`,
            },
        });

        if (!response.ok) {
            return null;
        }

        return await response.json();
    } catch (error) {
        console.error("Error checking ZGAN task status:", error);
        return null;
    }
}

export async function uploadToR2(uploadUrl: string, file: File): Promise<boolean> {
    try {
        const response = await fetch(uploadUrl, {
            method: "PUT",
            body: file,
        });

        if (!response.ok) {
            console.error("Failed to upload to R2:", response.status, response.statusText);
            return false;
        }

        return true;
    } catch (error) {
        console.error("Error uploading to R2:", error);
        return false;
    }
}
